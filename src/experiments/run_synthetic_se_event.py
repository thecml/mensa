"""
run_synthetic_se_event.py
====================================
Experiment 1.1
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
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import SingleEventSyntheticDataLoader
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function)
from utility.data import (dotdict, format_data, format_data_as_dict_single)
from utility.config import load_config
from mensa.model import train_mensa_model_2_events, make_mensa_model_2_events

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                         make_deephit_model, make_dsm_model, make_rsf_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set up precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Set up device
device = torch.device("cpu")

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"]
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.25] # between 0 and 0.8
#MODELS = ["cox", "coxnet", "coxboost", "rsf", "dsm", "deepsurv", "mtlr", "dcsurvival", "mensa"]
MODELS = ["deepsurv"]
N_SAMPLES = 1000
N_FEATURES = 10

if __name__ == "__main__":
    model_results = pd.DataFrame()
    
    for dataset_version in DATASET_VERSIONS:
        for copula_name in COPULA_NAMES:
            for k_tau in KENDALL_TAUS:
                # Load and split data
                data_config = load_config(cfg.DATA_CONFIGS_DIR, f"synthetic.yaml")
                dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                                linear=dataset_version,
                                                                n_samples=N_SAMPLES,
                                                                copula_name=copula_name,
                                                                k_tau=k_tau, n_features=N_FEATURES)
                dgps = dl.dgps
                train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
                
                # Make time bins
                time_bins = make_time_bins(train_dict['T'], event=train_dict['E'], dtype=dtype)
                
                # Format data to work easier with sksurv API
                X_train = pd.DataFrame(train_dict['X'], columns=[f'X{i}' for i in range(N_FEATURES)])
                X_valid = pd.DataFrame(valid_dict['X'], columns=[f'X{i}' for i in range(N_FEATURES)])
                X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(N_FEATURES)])
                y_train = convert_to_structured(train_dict['T'], train_dict['E'])
                y_valid = convert_to_structured(valid_dict['T'], valid_dict['E']) 
                y_test = convert_to_structured(test_dict['T'], test_dict['E'])
                
                 # Evaluate each model
                for model_name in MODELS:
                    if model_name == "cox":
                        config = load_config(cfg.COX_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_cox_model(config)
                        model.fit(X_train, y_train)
                    elif model_name == "coxnet":
                        config = load_config(cfg.COXNET_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_coxnet_model(config)
                        model.fit(X_train, y_train)
                    elif model_name == "coxboost":
                        config = load_config(cfg.COXBOOST_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_coxboost_model(config)
                        model.fit(X_train, y_train)
                    elif model_name == "rsf":
                        config = load_config(cfg.RSF_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_rsf_model(config)
                        model.fit(X_train, y_train)
                    elif model_name == "dsm":
                        config = load_config(cfg.DSM_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_dsm_model(config)
                        model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
                    elif model_name == "deepsurv":
                        config = dotdict(cfg.DEEPSURV_PARAMS)
                        model = DeepSurv(in_features=N_FEATURES, config=config)
                        data_train = pd.DataFrame(train_dict['X'])
                        data_train['time'] = train_dict['T']
                        data_train['event'] = train_dict['E']
                        model = train_deepsurv_model(model, data_train, time_bins, config=config, random_state=0,
                                                     reset_model=True, device=device, dtype=dtype)
                    elif model_name == "mtlr":
                        data_train = X_train.copy()
                        data_train["time"] = pd.Series(y_train['time'])
                        data_train["event"] = pd.Series(y_train['event']).astype(int)
                        data_valid = X_valid.copy()
                        data_valid["time"] = pd.Series(y_valid['time'])
                        data_valid["event"] = pd.Series(y_valid['event']).astype(int)
                        config = dotdict(load_config(cfg.MTLR_CONFIGS_DIR, f"synthetic.yaml"))
                        num_time_bins = len(time_bins)
                        model = mtlr(in_features=N_FEATURES, num_time_bins=num_time_bins, config=config)
                        model = train_mtlr_model(model, data_train, data_valid, time_bins,
                                                 config, random_state=0, dtype=dtype,
                                                 reset_model=True, device=device)
                    elif model_name == "dcsurvival":
                        config = load_config(cfg.DCSURVIVAL_CONFIGS_DIR, f"synthetic.yaml")
                        depth = config['depth']
                        widths = config['widths']
                        lc_w_range = config['lc_w_range']
                        shift_w_range = config['shift_w_range']
                        phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol=1e-14).to(device)
                        model = DCSurvival(phi, device, num_features=N_FEATURES, tol=1e-14).to(device)
                        model = train_dcsurvival_model(model, train_dict['X'], valid_dict['X'],
                                                       train_dict['T'], train_dict['E'],
                                                       valid_dict['T'], valid_dict['E'],
                                                       num_epochs=10, batch_size=32, device=device)
                    elif model_name == "mensa":
                        config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
                        model1, model2, copula = make_mensa_model_2_events(N_FEATURES, start_theta=2.0, eps=1e-4,
                                                                           device=device, dtype=dtype)
                        model1, model2, copula = train_mensa_model_2_events(train_dict, valid_dict, model1, model2,
                                                                            copula, n_epochs=100, lr=0.001)
                    else:
                        raise NotImplementedError()
                    
                    # Compute survival function
                    if model_name in ['cox', 'coxnet', "coxboost", 'rsf']:
                        model_preds = model.predict_survival_function(X_test)
                        model_preds = np.row_stack([fn(time_bins) for fn in model_preds])
                    elif model_name == 'dsm':
                        model_preds = model.predict_survival(X_test, times=list(time_bins.numpy()))
                    elif model_name == "deepsurv":
                        model_preds, time_bins_deepsurv, _ = make_deepsurv_prediction(model, test_dict['X'],
                                                                                      config=config, dtype=dtype)
                        spline = interp1d(time_bins_deepsurv, model_preds, kind='linear', fill_value='extrapolate')
                        model_preds = spline(time_bins)
                    elif model_name == "mtlr":
                        data_test = X_test.copy()
                        data_test["time"] = pd.Series(y_test['time'])
                        data_test["event"] = pd.Series(y_test['event']).astype('int')
                        mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                      dtype=dtype, device=device)
                        survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, time_bins, config)
                        model_preds = survival_outputs[:, 1:].numpy()
                    elif model_name == "mensa":
                        model_preds = predict_survival_function(model1, test_dict['X'], time_bins).numpy()
                    elif model_name == "dcsurvival":
                        model_preds = predict_survival_function(model, test_dict['X'], time_bins).numpy()
                    else:
                        raise NotImplementedError()
                        
                    # Compute L1 (Truth vs. Model) - event only
                    n_samples = test_dict['X'].shape[0]
                    truth_preds_e = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_e[:,i] = dgps[0].survival(time_bins[i], test_dict['X'])
                    l1_e = float(compute_l1_difference(truth_preds_e, model_preds,
                                                       n_samples, steps=time_bins))
                    
                    print(f"Evaluated {model_name} - {dataset_version} - {round(k_tau, 3)} - {round(l1_e, 3)}")
                    res_sr = pd.Series([model_name, dataset_version, copula_name, k_tau, l1_e],
                                       index=["ModelName", "DatasetVersion", "Copula", "KTau", "L1"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    model_results.to_csv(f"{cfg.RESULTS_DIR}/synthetic_se_event.csv")
                    