"""
run_synthetic_competing_risks.py
====================================
Experiment 2.1

Models: ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'survtrace', 'mensa', 'dgp']
"""

# 3rd party
import pandas as pd
import numpy as np
import config as cfg
import torch
import torch.optim as optim
import torch.nn as nn
import random
import warnings
import copy
import tqdm
import math
import argparse
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import CompetingRiskSyntheticDataLoader
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import dotdict
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from utility.data import format_data_deephit_cr, format_hierarchical_data, calculate_layer_size_hierarch, format_survtrace_data
from utility.evaluation import global_C_index, local_C_index

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                          make_deephit_cr, make_dsm_model, make_rsf_model, train_deepsurv_model,
                          make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction, train_mtlr_cr
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function
from hierarchical.data_settings import synthetic_cr_settings
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from torchmtlr.utils import encode_mtlr_format, reset_parameters, encode_mtlr_format_no_censoring
from torchmtlr.model import MTLRCR, mtlr_neg_log_likelihood, mtlr_risk, mtlr_survival
from utility.data import calculate_vocab_size
from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceMulti
from survtrace.train_utils import Trainer
from survtrace.config import STConfig
from utility.data import calculate_vocab_size
from pycox.models import DeepHit, DeepHitSingle

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ['mensa', 'dgp']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_tau', type=float, default=0.25)
    parser.add_argument('--copula_name', type=str, default="clayton")
    parser.add_argument('--linear', type=bool, default=True)
    
    args = parser.parse_args()
    seed = args.seed
    k_tau = args.k_tau
    copula_name = args.copula_name
    linear = args.linear
    
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic.yaml")
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config, k_tau=k_tau, copula_name=copula_name,
                                                      linear=linear, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    n_events = data_config['cr_n_events']
    dgps = dl.dgps
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Evaluate models
    model_results = pd.DataFrame()
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = DeepSurv(in_features=n_features, config=config)
                data_train = pd.DataFrame(train_dict['X'])
                data_train['time'] = train_dict['T']
                data_train['event'] = (train_dict['E'] == i)*1.0
                data_valid = pd.DataFrame(valid_dict['X'])
                data_valid['time'] = valid_dict['T']
                data_valid['event'] = (valid_dict['E'] == i)*1.0
                model = train_deepsurv_model(model, data_train, data_valid, time_bins,
                                             config=config, random_state=0,
                                             reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "deephit":
            config = dotdict(cfg.DEEPHIT_PARAMS)
            model = make_deephit_cr(in_features=n_features, out_features=len(time_bins),
                                    num_risks=n_events, duration_index=time_bins, config=config)
            train_data, valid_data, out_features, duration_index = format_data_deephit_cr(train_dict, valid_dict, time_bins)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_cr.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data(train_dict, valid_dict,
                                                                            test_dict, n_time_bins)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = len(time_bins)
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                                valid_data, config, hyperparams, verbose)
        elif model_name == "mtlrcr":
            train_times = np.digitize(train_dict['T'], bins=time_bins).astype(np.int64)
            train_events = train_dict['E'].type(torch.int64)
            valid_times = np.digitize(valid_dict['T'], bins=time_bins).astype(np.int64)
            valid_events = valid_dict['E'].type(torch.int64)
            y_train = encode_mtlr_format_no_censoring(train_times, train_events + 1, time_bins)
            y_valid = encode_mtlr_format_no_censoring(valid_times, valid_events + 1, time_bins)
            num_time_bins = len(time_bins) + 1
            config = dotdict(cfg.MTLRCR_PARAMS)
            model = MTLRCR(in_features=n_features, num_time_bins=num_time_bins, num_events=n_events)
            model = train_mtlr_cr(train_dict['X'], y_train, valid_dict['X'], y_valid,
                                    model, time_bins, num_epochs=config['num_epochs'],
                                    lr=config['lr'], batch_size=config['batch_size'],
                                    verbose=True, device=device, C1=config['c1'],
                                    early_stop=config['early_stop'], patience=config['patience'])
        elif model_name == "dsm":
            config = dotdict(cfg.DSM_PARAMS)
            model = make_dsm_model(config)
            X_train = pd.DataFrame(train_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            X_valid = pd.DataFrame(valid_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            y_train = pd.DataFrame({'event': train_dict['E'], 'time': train_dict['T']})
            y_valid = pd.DataFrame({'event': valid_dict['E'], 'time': valid_dict['T']})
            model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
        elif model_name == "survtrace":
            config = dotdict(cfg.SURVTRACE_PARAMS)
            X_train = pd.DataFrame(train_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            X_valid = pd.DataFrame(valid_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            cat_features = []
            num_features = [f'X{i}' for i in range(n_features)]
            y_train, y_valid, duration_index, out_features = format_survtrace_data(train_dict, valid_dict,
                                                                                    time_bins, n_events)
            config['vocab_size'] = calculate_vocab_size(X_train, cat_features)
            config['duration_index'] = duration_index
            config['out_feature'] = out_features
            config['num_numerical_feature'] = int(len(num_features))
            config['num_categorical_feature'] = int(len(cat_features))
            config['num_feature'] = n_features
            config['num_event'] = n_events
            config['in_features'] = n_features
            model = SurvTraceMulti(dotdict(config))
            trainer = Trainer(model)
            trainer.fit((X_train, y_train), (X_valid, y_valid),
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'],
                        val_batch_size=config['batch_size'])
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            model1, model2, model3, copula = make_mensa_model_3_events(n_features, start_theta=2.0, eps=1e-4,
                                                                       device=device, dtype=dtype)
            model1, model2, model3, copula = train_mensa_model_3_events(train_dict, valid_dict, model1, model2, model3,
                                                                        copula, n_epochs=5000, lr=0.001)
            print(f"NLL all events: {triple_loss(model1, model2, model3, valid_dict, copula)}")
            print(f"DGP loss: {triple_loss(dgps[0], dgps[1], dgps[2], valid_dict, copula)}")
        elif model_name == "dgp":
            continue
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]
        if model_name == "deepsurv":
            all_preds = []
            for trained_model in trained_models:
                preds, time_bins_model, _ = make_deepsurv_prediction(trained_model, test_dict['X'],
                                                                     config=config, dtype=dtype)
                spline = interp1d(time_bins_model, preds, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins), columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "deephit":
            cif = model.predict_cif(test_dict['X'])
            all_preds = []
            for i in range(n_events):
                cif_df = pd.DataFrame((1-cif[i]).T, columns=time_bins.numpy())
                all_preds.append(cif_df)
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(test_data[0], model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            all_preds = []
            for event_pred in event_preds:
                preds = pd.DataFrame(event_pred, columns=bin_locations)
                spline = interp1d(bin_locations, preds, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins), columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "mtlrcr":
            pred_prob = model(test_dict['X'])
            num_points = len(time_bins)
            preds_e1 = mtlr_survival(pred_prob[:,:num_time_bins]).detach().numpy()[:, 1:] # drop extra bin
            preds_e2 = mtlr_survival(pred_prob[:,num_time_bins:num_time_bins*2]).detach().numpy()[:, 1:]
            preds_e3 = mtlr_survival(pred_prob[:,num_time_bins*2:]).detach().numpy()[:, 1:]
            preds_e1 = pd.DataFrame(preds_e1, columns=time_bins.numpy())
            preds_e2 = pd.DataFrame(preds_e2, columns=time_bins.numpy())
            preds_e3 = pd.DataFrame(preds_e3, columns=time_bins.numpy())
            all_preds = [preds_e1, preds_e2, preds_e3]
        elif model_name == "dsm":
            X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            model_preds = model.predict_survival(X_test, times=list(time_bins.numpy()))
            model_preds = pd.DataFrame(model_preds, columns=time_bins.numpy())
            all_preds = [model_preds, model_preds]
        elif model_name == "survtrace":
            all_preds = []
            for i in range(n_events):
                preds = model.predict_surv(test_dict['X'], batch_size=32, event=i)[:, 1:] # drop extra bin
                preds = pd.DataFrame(preds, columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "mensa":
            all_preds = []
            models = [model1, model2, model3]
            for model in models:
                preds = predict_survival_function(model, test_dict['X'], time_bins).detach().numpy()
                preds = pd.DataFrame(preds, columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "dgp":
            all_preds = []
            for model in dgps:
                preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                for i in range(time_bins.shape[0]):
                    preds[:,i] = model.survival(time_bins[i], test_dict['X'])
                    preds = pd.DataFrame(preds, columns=time_bins.numpy())
                    all_preds.append(preds)
        else:
            raise NotImplementedError()
        
        # Test local and global CI
        y_test_time = np.stack([test_dict['T1'], test_dict['T2'], test_dict['T3']], axis=1)
        y_test_event = np.array(pd.get_dummies(test_dict['E']))
        all_preds_arr = [df.to_numpy() for df in all_preds]
        global_ci = global_C_index(all_preds_arr, y_test_time, y_test_event)
        local_ci = local_C_index(all_preds_arr, y_test_time, y_test_event)
    
        # Make evaluation for each event
        for event_id, surv_preds in enumerate(all_preds):
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict[f'T{event_id+1}']
            y_train_event = np.array([1] * n_train_samples)
            y_test_time = test_dict[f'T{event_id+1}']
            y_test_event = np.array([1] * n_test_samples)
            lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            ci =  lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score(num_points=len(time_bins))
            mae = lifelines_eval.mae(method='Uncensored')
            d_calib = lifelines_eval.d_calibration()[0]
            
            truth_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                truth_preds[:,i] = dgps[event_id].survival(time_bins[i], test_dict['X'])
            survival_l1 = float(compute_l1_difference(truth_preds, surv_preds.to_numpy(),
                                                      n_samples, steps=time_bins))
            
            metrics = [ci, ibs, mae, survival_l1, d_calib, global_ci, local_ci]
            print(metrics)
            res_sr = pd.Series([model_name, linear, copula_name, k_tau] + metrics,
                                index=["ModelName", "Linear", "Copula", "KTau",
                                        "CI", "IBS", "MAE", "L1", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
            model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")
                    