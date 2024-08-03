"""
run_synthetic_se_event.py
====================================
Experiment 1.1

Models: ["deepsurv", "deephit", "mtlr", "dsm", "dcsurvival", "mensa", "mensa-nocop", "dgp"]
"""

# 3rd party
import pandas as pd
import numpy as np
import config as cfg
import torch
import random
import warnings
import argparse
import os
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import SingleEventSyntheticDataLoader
from utility.survival import (make_time_bins, convert_to_structured,
                              compute_l1_difference, predict_survival_function)
from utility.data import dotdict
from utility.config import load_config
from mensa.model import MENSA
from utility.data import format_data_deephit_single
from copula import Convex_bivariate

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model,
                         make_dsm_model, make_rsf_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv, make_deephit_single, train_deephit_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ["deepsurv", "deephit", "mtlr", "dsm", "dcsurvival", "mensa", "mensa-nocop", "dgp"]

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
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_se.yaml")
    dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                    linear=linear, copula_name=copula_name,
                                                    k_tau=k_tau, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    dgps = dl.dgps
    
    # Make time bins    
    min_time = dl.get_data()[1].min()
    max_time = dl.get_data()[1].max()
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)

    # Format data to work easier with sksurv API
    X_train = pd.DataFrame(train_dict['X'], columns=[f'X{i}' for i in range(n_features)])
    X_valid = pd.DataFrame(valid_dict['X'], columns=[f'X{i}' for i in range(n_features)])
    X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(n_features)])
    y_train = convert_to_structured(train_dict['T'], train_dict['E'])
    y_valid = convert_to_structured(valid_dict['T'], valid_dict['E'])
    y_test = convert_to_structured(test_dict['T'], test_dict['E'])
    
    # Evaluate each model
    for model_name in MODELS:
        if model_name == "cox":
            config = dotdict(cfg.COX_PARAMS)
            model = make_cox_model(config)
            model.fit(X_train, y_train)
        elif model_name == "coxnet":
            config = dotdict(cfg.COXNET_PARAMS)
            model = make_coxnet_model(config)
            model.fit(X_train, y_train)
        elif model_name == "coxboost":
            config = dotdict(cfg.COXBOOST_PARAMS)
            model = make_coxboost_model(config)
            model.fit(X_train, y_train)
        elif model_name == "rsf":
            config = dotdict(cfg.RSF_PARAMS)
            model = make_rsf_model(config)
            model.fit(X_train, y_train)
        elif model_name == "dsm":
            config = dotdict(cfg.DSM_PARAMS)
            model = make_dsm_model(config)
            model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
        elif model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            model = DeepSurv(in_features=n_features, config=config)
            data_train = pd.DataFrame(train_dict['X'])
            data_train['time'] = train_dict['T']
            data_train['event'] = train_dict['E']
            data_valid = pd.DataFrame(valid_dict['X'])
            data_valid['time'] = valid_dict['T']
            data_valid['event'] = valid_dict['E']
            model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                         random_state=0, reset_model=True, device=device, dtype=dtype)
        elif model_name == "deephit":
            config = dotdict(cfg.DEEPHIT_PARAMS)
            model = make_deephit_single(in_features=n_features, out_features=len(time_bins),
                                        time_bins=time_bins.cpu().numpy(), device=device, config=config)
            labtrans = model.label_transform
            train_data, valid_data, out_features, duration_index = format_data_deephit_single(train_dict, valid_dict, labtrans)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "mtlr":
            data_train = X_train.copy()
            data_train["time"] = pd.Series(y_train['time'])
            data_train["event"] = pd.Series(y_train['event']).astype(int)
            data_valid = X_valid.copy()
            data_valid["time"] = pd.Series(y_valid['time'])
            data_valid["event"] = pd.Series(y_valid['event']).astype(int)
            config = dotdict(cfg.MTLR_PARAMS)
            num_time_bins = len(time_bins)
            model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
            model = train_mtlr_model(model, data_train, data_valid, time_bins,
                                     config, random_state=0, dtype=dtype,
                                     reset_model=True, device=device)
        elif model_name == "dcsurvival":
            config = dotdict(cfg.DCSURVIVAL_PARAMS)
            depth = config['depth']
            widths = config['widths']
            lc_w_range = config['lc_w_range']
            shift_w_range = config['shift_w_range']
            learning_rate = 1e-4
            phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol=1e-14).to(device)
            model = DCSurvival(phi, device, num_features=n_features, tol=1e-14).to(device)
            model = train_dcsurvival_model(model, train_dict['X'], valid_dict['X'],
                                           train_dict['T'], train_dict['E'],
                                           valid_dict['T'], valid_dict['E'],
                                           num_epochs=1000, learning_rate=learning_rate, device=device)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            dropout = config['dropout']
            copula = Convex_bivariate(copulas=['cl'], dtype=dtype, device=device)
            model = MENSA(n_features=n_features, n_events=2, hidden_layers=layers,
                          dropout=dropout, copula=copula, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs,
                      lr_dict={'network': lr, 'copula': 0.01})
        elif model_name == "mensa-nocop":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            dropout = config['dropout']
            model = MENSA(n_features=n_features, n_events=2, hidden_layers=layers,
                          dropout=dropout, copula=None, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr_dict={'network': lr})
        elif model_name == "dgp":
            pass
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]
        if model_name in ['cox', 'coxnet', "coxboost", 'rsf']:
            model_preds = model.predict_survival_function(X_test)
            model_preds = np.row_stack([fn(time_bins) for fn in model_preds])
        elif model_name == 'dsm':
            model_preds = model.predict_survival(X_test, times=list(time_bins.cpu().numpy()))
        elif model_name == "deepsurv":
            model_preds, time_bins_deepsurv = make_deepsurv_prediction(model, test_dict['X'].to(device),
                                                                       config=config, dtype=dtype)
            spline = interp1d(time_bins_deepsurv.cpu().numpy(), model_preds.cpu().numpy(),
                              kind='linear', fill_value='extrapolate')
            model_preds = spline(time_bins.cpu().numpy())
        elif model_name == "mtlr":
            data_test = X_test.copy()
            data_test["time"] = pd.Series(y_test['time'])
            data_test["event"] = pd.Series(y_test['event']).astype('int')
            mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                          dtype=dtype, device=device)
            survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, time_bins, config)
            model_preds = survival_outputs[:, 1:].cpu().numpy()
        elif model_name == "deephit":
            model_preds = model.predict_surv(test_dict['X']).cpu().numpy()
        elif model_name == "dcsurvival":
            model_preds = predict_survival_function(model, test_dict['X'].to(device),
                                                    time_bins, device=device).cpu().numpy()
        elif model_name in ["mensa", "mensa-nocop"]:
            model_preds = model.predict(test_dict['X'], time_bins)[1].cpu().detach().numpy() # use event preds
        elif model_name == "dgp":
            model_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                model_preds[:,i] = dgps[0].survival(time_bins[i], test_dict['X'].to(device))
            model_preds = model_preds.cpu().numpy()
        else:
            raise NotImplementedError()
            
        # Compute L1 (Truth vs. Model) - event only
        truth_preds_e = torch.zeros((n_samples, time_bins.shape[0]), device=device)
        for i in range(time_bins.shape[0]):
            truth_preds_e[:,i] = dgps[0].survival(time_bins[i], test_dict['X'].to(device))
        model_preds_th = torch.tensor(model_preds, device=device, dtype=dtype)
        l1_e = float(compute_l1_difference(truth_preds_e, model_preds_th, n_samples,
                                           steps=time_bins, device=device))
        
        # Compute prediction metrics
        surv_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        n_train_samples = len(train_dict['X'])
        n_test_samples= len(test_dict['X'])
        y_train_time = train_dict['T']
        y_train_event = np.array([1] * n_train_samples)
        y_test_time = test_dict[f'T']
        y_test_event = np.array([1] * n_test_samples)
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        ci =  lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score(num_points=len(time_bins))
        mae = lifelines_eval.mae(method='Uncensored')
        
        metrics = [ci, ibs, mae, l1_e]
        print(metrics)
        result_row = pd.Series([model_name, seed, linear, copula_name, k_tau] + metrics,
                               index=["ModelName", "Seed", "Linear", "Copula", "KTau",
                                      "CI", "IBS", "MAE", "L1"])
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/synthetic_se.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=result_row.keys())
        results = results.append(result_row, ignore_index=True)
        results.to_csv(filename, index=False)
        