"""
run_synthetic_competing_risks.py
====================================
Experiment 2.1

Datasets: SEER, Rotterdam
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
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import (dotdict, format_data, format_data_as_dict_single)
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from utility.data import (format_data_deephit_cr, format_hierarchical_data, calculate_layer_size_hierarch,
                          format_survtrace_data, format_data_as_dict_single)
from utility.evaluation import global_C_index, local_C_index
from data_loader import SeerDataLoader

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
from pycox.models import DeepHit

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
MODELS = ['deephit']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    seed = args.seed
    
    # Load and split data
    dl = SeerDataLoader().load_data(n_samples=1000, device=device, dtype=dtype)
    df_train, df_valid, df_test = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                random_state=seed)
    n_events = dl.n_events
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    X_train = df_train.drop(['event', 'time'], axis=1)
    X_valid = df_valid.drop(['event', 'time'], axis=1)
    X_test = df_test.drop(['event', 'time'], axis=1)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict = format_data_as_dict_single(X_train, df_train['event'], df_train['time'], dtype)
    valid_dict = format_data_as_dict_single(X_valid, df_valid['event'], df_valid['time'], dtype)
    test_dict = format_data_as_dict_single(X_test, df_test['event'], df_test['time'], dtype)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Evaluate models
    model_results = pd.DataFrame()
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            model1 = DeepSurv(in_features=n_features, config=config)
            model2 = DeepSurv(in_features=n_features, config=config)
            data_train1 = pd.DataFrame(train_dict['X'])
            data_train1['time'] = train_dict['T']
            data_train1['event'] = (train_dict['E'] == 0)*1.0
            data_train2 = pd.DataFrame(train_dict['X'])
            data_train2['time'] = train_dict['T']
            data_train2['event'] = (train_dict['E'] == 1)*1.0
            model1 = train_deepsurv_model(model1, data_train1, time_bins, config=config, random_state=0,
                                          reset_model=True, device=device, dtype=dtype)
            model2 = train_deepsurv_model(model2, data_train2, time_bins, config=config, random_state=0,
                                          reset_model=True, device=device, dtype=dtype)
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
            y_train = encode_mtlr_format(train_times, train_events, time_bins)
            y_valid = encode_mtlr_format(valid_times, valid_events, time_bins)
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
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]
        if model_name == "deepsurv":
            preds_e1, time_bins_model1, _ = make_deepsurv_prediction(model1, test_dict['X'],
                                                                     config=config, dtype=dtype)
            preds_e2, time_bins_model2, _ = make_deepsurv_prediction(model2, test_dict['X'],
                                                                     config=config, dtype=dtype)
            spline1 = interp1d(time_bins_model1, preds_e1, kind='linear', fill_value='extrapolate')
            spline2 = interp1d(time_bins_model2, preds_e2, kind='linear', fill_value='extrapolate')
            all_preds = [spline1(time_bins), spline2(time_bins)]
        elif model_name == "deephit":
            cif = model.predict_cif(test_dict['X'])
            cif1 = pd.DataFrame((1-cif[0]).T, columns=time_bins.numpy())
            cif2 = pd.DataFrame((1-cif[1]).T, columns=time_bins.numpy())
            all_preds = [cif1.values, cif2.values]
        elif model_name == "hierarch": #TODO
            event_preds = util.get_surv_curves(test_data[0], model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            preds_e1 = pd.DataFrame(event_preds[0], columns=bin_locations)
            preds_e2 = pd.DataFrame(event_preds[1], columns=bin_locations)
            spline1 = interp1d(bin_locations, preds_e1, kind='linear', fill_value='extrapolate') 
            spline2 = interp1d(bin_locations, preds_e2, kind='linear', fill_value='extrapolate')
            all_preds = [spline1(time_bins), spline2(time_bins)]
        elif model_name == "mtlrcr":
            pred_prob = model(test_dict['X'])
            num_points = len(time_bins)
            preds_e1 = mtlr_survival(pred_prob[:,:num_time_bins]).detach().numpy()[:, 1:] # drop extra bin
            preds_e2 = mtlr_survival(pred_prob[:,num_time_bins:num_time_bins*2]).detach().numpy()[:, 1:]
            all_preds = [preds_e1, preds_e2]
        elif model_name == "dsm":
            X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(n_features)])
            model_preds = model.predict_survival(X_test, times=list(time_bins.numpy()))
            all_preds = [model_preds.copy(), model_preds.copy()]
        elif model_name == "survtrace":
            preds_e1 = model.predict_surv(test_dict['X'], batch_size=32, event=0)[:, 1:] # drop extra bin
            preds_e2 = model.predict_surv(test_dict['X'], batch_size=32, event=1)[:, 1:]
            all_preds = [preds_e1, preds_e2]
        elif model_name == "mensa":
            preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy() # censoring
            preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
            preds_e3 = predict_survival_function(model3, test_dict['X'], time_bins).detach().numpy()
            all_preds = [preds_e2, preds_e3]
        else:
            raise NotImplementedError()
        
        # Test local and global CI
        y_test_time = np.stack([test_dict['T'], test_dict['T']], axis=1)
        y_test_event = np.stack([np.array((test_dict['E'] == 1)*1.0),
                                 np.array((test_dict['E'] == 2)*1.0)], axis=1)
        global_ci = global_C_index(all_preds, y_test_time, y_test_event)
        local_ci = local_C_index(all_preds, y_test_time, y_test_event)
    
        # Make evaluation for each event
        for event_id, surv_preds in enumerate(all_preds):
            surv_preds_df = pd.DataFrame(surv_preds, columns=time_bins.numpy())
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict['T']
            y_train_event = (train_dict['E'] == event_id+1)*1.0
            y_test_time = test_dict['T']
            y_test_event = (test_dict['E'] == event_id+1)*1.0
            lifelines_eval = LifelinesEvaluator(surv_preds_df.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            ci = lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score()
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_margin = lifelines_eval.mae(method="Margin")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            d_calib = lifelines_eval.d_calibration()[0]
            
            metrics = [ci, ibs, mae_hinge, mae_margin, mae_pseudo, d_calib, global_ci, local_ci]
            print(metrics)
            res_sr = pd.Series([model_name] + metrics,
                                index=["ModelName", "CI", "IBS", "MAE-H", "MAE-M",
                                       "MAE-P", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
            model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")
                    