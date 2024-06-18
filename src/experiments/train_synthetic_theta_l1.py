import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_mensa_model
import torch
import random
import warnings
from models import Mensa
from multi_evaluator import MultiEventEvaluator
from data_loader import *
from utility.survival import preprocess_data
from utility.data import dotdict, array_to_tensor
import torch.optim as optim
import torch.nn as nn
from copulas import Clayton
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival, MultiNDESurvival, SurvNDE
from tqdm import tqdm
from utility.evaluator import LifelinesEvaluator
import copy
from models import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from torch.utils.data import DataLoader, TensorDataset
import math
from utility.data import format_data
from utility.config import load_config
from sota_builder import * # Import all SOTA models
from utility.survival import risk_fn, compute_l1_difference, predict_survival_curve
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device("cpu")

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.8]
MODELS = ["weibull"] #"coxnet", "coxboost", "rsf", "dsm", "dcph", "weibull"
N_SAMPLES = 10000
N_FEATURES = 10

if __name__ == "__main__":
    model_results = pd.DataFrame()
    
    for dataset_version in DATASET_VERSIONS:
        for copula_name in COPULA_NAMES:
            for k_tau in KENDALL_TAUS:
                # Load and split data
                if dataset_version == "linear":
                    dl = LinearSyntheticDataLoader().load_data(n_samples=N_SAMPLES,
                                                               copula_name=copula_name,
                                                               k_tau=k_tau, n_features=N_FEATURES)
                else:
                    dl = NonlinearSyntheticDataLoader().load_data(n_samples=N_SAMPLES,
                                                                  copula_name=copula_name,
                                                                  k_tau=k_tau, n_features=N_FEATURES)
                num_features, cat_features = dl.get_features()
                (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                                         valid_size=0.5)
                
                # Make time bins
                time_bins = make_time_bins(y_train['time'], event=y_train['event'], dtype=dtype)
                
                # Scale data
                X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test,
                                                           cat_features, num_features)
                
                # Format data
                #(covariates, times, events)
                X_train_th, times_train_th, events_train_th = format_data(X_train, y_train, dtype, device)
                X_valid_th, times_valid_th, events_valid_th = format_data(X_valid, y_valid, dtype, device)
                X_test_th, times_test_th, events_test_th = format_data(X_test, y_test, dtype, device)
                
                # Make truth model
                n_features = X_test.shape[1]
                beta_e, shape_e, scale_e, beta_c, shape_c, scale_c, u_e = dl.params
                if dataset_version == "linear":
                    truth_model_e = Weibull_linear(n_features, alpha=scale_e, gamma=shape_e,
                                                   beta=beta_e, device=device, dtype=dtype)
                    truth_model_c = Weibull_linear(n_features, alpha=scale_c, gamma=shape_c,
                                                   beta=beta_c, device=device, dtype=dtype)
                else:
                    truth_model = Weibull_nonlinear(n_features, alpha=shape_e, gamma=scale_e,
                                                    beta=beta_e, risk_function=risk_fn, device=device,
                                                    dtype=dtype)
                                    
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
                    elif model_name == "dcph":
                        config = load_config(cfg.DCPH_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_dcph_model(config)
                        model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
                    elif model_name == "mtlr":
                        data_train = X_train.copy()
                        data_train["time"] = pd.Series(y_train['time'])
                        data_train["event"] = pd.Series(y_train['event']).astype(int)
                        data_valid = X_valid.copy()
                        data_valid["time"] = pd.Series(y_valid['time'])
                        data_valid["event"] = pd.Series(y_valid['event']).astype(int)
                        config = dotdict(load_config(cfg.MTLR_CONFIGS_DIR, f"synthetic.yaml"))
                        num_time_bins = len(time_bins)
                        model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
                        model = train_mtlr_model(model, data_train, data_valid, time_bins,
                                                 config, random_state=0, reset_model=True, device=device)
                    elif model_name == "mensa":
                        raise NotImplementedError()
                    elif model_name == "weibull":
                        model1 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                        model2 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                        train_dict = dict()
                        train_dict['X'] = torch.tensor(X_train.to_numpy(), dtype=dtype)
                        train_dict['T'] = torch.tensor(y_train['time'].copy(), dtype=dtype)
                        train_dict['E'] = torch.tensor(y_train['event'].copy(), dtype=dtype)
                        valid_dict = dict()
                        valid_dict['X'] = torch.tensor(X_valid.to_numpy(), dtype=dtype)
                        valid_dict['T'] = torch.tensor(y_valid['time'].copy(), dtype=dtype)
                        valid_dict['E'] = torch.tensor(y_valid['event'].copy(), dtype=dtype)
                        
                        # Calculate loss for truth models
                        from trainer import loss_function
                        copula = Clayton.Clayton(4, device)
                        truth_loss = loss_function(truth_model_e, truth_model_c, valid_dict, copula=copula)
                        print(truth_loss)
                        model1, model2 = independent_train_loop_linear(model1, model2,
                                                                       train_dict, valid_dict, 1000)
                        print(0)
                        
                    else:
                        raise NotImplementedError()
                    
                    # Compute survival function
                    if model_name in ['cox', 'coxnet', "coxboost", 'rsf']:
                        model_preds = model.predict_survival_function(X_test)
                        model_preds = np.row_stack([fn(time_bins) for fn in model_preds])
                    elif model_name in ['dsm', 'dcph']:
                        model_preds = model.predict_survival(X_test, times=list(time_bins.numpy()))
                    elif model_name == "mtlr":
                        data_test = X_test.copy()
                        data_test["time"] = pd.Series(y_test['time'])
                        data_test["event"] = pd.Series(y_test['event']).astype(int)
                        mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                      dtype=dtype, device=device)
                        survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, time_bins, config)
                        survival_outputs = survival_outputs[:, 1:]
                        model_preds = survival_outputs.numpy()
                    elif model_name == "mensa":
                        raise NotImplementedError()
                    elif model_name == "weibull":
                        model_preds_e, _, _ = predict_survival_curve(model1, X_test_th, time_bins)
                        model_preds_e = model_preds_e.numpy()
                        model_preds_c, _, _ = predict_survival_curve(model2, X_test_th, time_bins)
                        model_preds_c = model_preds_c.numpy()
                        
                    # Compute L1 (Truth vs. Model) - event/censoring
                    n_samples = X_test_th.shape[0]
                    truth_preds_e = torch.zeros((X_test_th.shape[0], time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_e[:,i] = truth_model_e.survival(time_bins[i], X_test_th)
                    l1_e = float(compute_l1_difference(truth_preds_e, model_preds_e,
                                                       n_samples, steps=time_bins))
                    truth_preds_c = torch.zeros((X_test_th.shape[0], time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_c[:,i] = truth_model_c.survival(time_bins[i], X_test_th)
                    n_samples = X_test_th.shape[0]
                    l1_c = float(compute_l1_difference(truth_preds_c, model_preds_c,
                                                       n_samples, steps=time_bins))
                    
                    print(f"Evaluated {model_name} - {dataset_version} - {round(k_tau, 3)} - {round(l1_e, 3)} - {round(l1_c, 3)}")
                    res_sr = pd.Series([model_name, dataset_version, copula_name, k_tau, l1_e, l1_c],
                                       index=["ModelName", "DatasetVersion", "Copula", "KTau", "L1E", "L1C"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")
                    