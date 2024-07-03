"""
run_synthetic_competing_risks.py
====================================
Experiment 2.1

Models: ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa', 'dgp']
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
import os
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import CompetingRiskSyntheticDataLoader
from copula import Clayton2D, Frank2D, NestedClayton
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import dotdict
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from utility.data import format_data_deephit_cr, format_hierarchical_data_cr, calculate_layer_size_hierarch, format_survtrace_data
from utility.evaluation import global_C_index, local_C_index
from mensa.model import CompetingMENSA

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
MODELS = ['mensa']

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
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_cr.yaml")
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config, k_tau=k_tau, copula_name=copula_name,
                                                      linear=linear, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    n_events = data_config['n_events']
    dgps = dl.dgps
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Evaluate models
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
            min_time = torch.tensor([dl.get_data()[1].min()], dtype=dtype)
            max_time = torch.tensor([dl.get_data()[1].max()], dtype=dtype)
            time_bins_dh = time_bins
            if min_time not in time_bins_dh:
                time_bins_dh = torch.concat([min_time, time_bins_dh], dim=0)
            if max_time not in time_bins_dh:
                time_bins_dh = torch.concat([time_bins_dh, max_time], dim=0)
            model = make_deephit_cr(in_features=n_features, out_features=len(time_bins_dh),
                                    num_risks=n_events, duration_index=time_bins_dh, config=config)
            train_data, valid_data, out_features, duration_index = format_data_deephit_cr(train_dict, valid_dict, time_bins_dh)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_cr.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                                                            n_time_bins, n_events)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = len(time_bins)
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
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
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            copula = NestedClayton(torch.tensor([2.0]), torch.tensor([2.0]),
                                   1e-4, 1e-4, device, dtype)
            n_epochs = config['n_epochs']
            lr = config['lr']
            model = CompetingMENSA(n_features, n_events, copula=copula, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr=lr)
        elif model_name == "dgp":
            pass
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
                cif_df = pd.DataFrame((1-cif[i]).T, columns=time_bins_dh.numpy())
                spline = interp1d(time_bins_dh, cif_df, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins), columns=time_bins.numpy())
                all_preds.append(preds)
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
            all_preds = []
            for i in range(n_events):
                start = i * num_time_bins
                end = start + num_time_bins
                preds = mtlr_survival(pred_prob[:, start:end]).detach().numpy()[:, 1:]
                preds = pd.DataFrame(preds, columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "dsm":
            X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            model_preds = model.predict_survival(X_test, times=list(time_bins.numpy()))
            model_preds = pd.DataFrame(model_preds, columns=time_bins.numpy())
            all_preds = [model_preds for _ in range(n_events)]
        elif model_name == "mensa":
            models = model.get_models()
            all_preds = []
            for i in range(n_events):
                model_preds = predict_survival_function(models[i], test_dict['X'], time_bins).detach().numpy()
                model_preds = pd.DataFrame(model_preds, columns=time_bins.numpy())
                all_preds.append(model_preds)
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
        
        # Calculate local and global CI
        y_test_time = np.stack([test_dict['T'] for _ in range(n_events)], axis=1)
        y_test_event = np.stack([np.array((test_dict['E'] == i)*1.0) for i in range(n_events)], axis=1)
        all_preds_arr = [df.to_numpy() for df in all_preds]
        global_ci = global_C_index(all_preds_arr, y_test_time, y_test_event)
        local_ci = local_C_index(all_preds_arr, y_test_time, y_test_event)
        
        # Make evaluation for each event
        model_results = pd.DataFrame()
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
            res_sr = pd.Series([model_name, seed, linear, copula_name, k_tau, event_id+1] + metrics,
                                index=["ModelName", "Seed", "Linear", "Copula", "KTau", "EventId",
                                        "CI", "IBS", "MAE", "L1", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/synthetic_cr.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)
                    