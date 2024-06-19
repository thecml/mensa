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
from utility.data import format_data, format_data_as_dict
from utility.config import load_config
from sota_builder import * # Import all SOTA models
from utility.survival import risk_fn, compute_l1_difference, predict_survival_curve
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0, 0.25, 0.5, 0.8]
MODELS = ["weibull-nocop", "weibull-cop"]
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
                (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(frac_train=0.7,
                                                                                         frac_valid=0.1,
                                                                                         frac_test=0.2)
                
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
                    truth_model_e = Weibull_nonlinear(n_features, alpha=shape_e, gamma=scale_e,
                                                      beta=beta_e, risk_function=risk_fn, device=device,
                                                      dtype=dtype)
                    truth_model_c = Weibull_nonlinear(n_features, alpha=shape_c, gamma=scale_c,
                                                      beta=beta_c, risk_function=risk_fn, device=device,
                                                      dtype=dtype)
                                    
                 # Evaluate each model
                for model_name in MODELS:
                    if model_name == "weibull-nocop":
                        model1 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                        model2 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                        train_dict = format_data_as_dict(X_train, y_train, dtype)
                        valid_dict = format_data_as_dict(X_valid, y_valid, dtype)
                        model1, model2 = independent_train_loop_linear(model1, model2, train_dict,
                                                                       valid_dict, 15000)
                    elif model_name == "weibull-cop":
                        model1 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                        model2 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                        train_dict = format_data_as_dict(X_train, y_train, dtype)
                        valid_dict = format_data_as_dict(X_valid, y_valid, dtype)
                        copula = Clayton.Clayton(torch.tensor([1], dtype=dtype), device)
                        #copula_dgp = Clayton.Clayton(torch.tensor([8], dtype=dtype), device)
                        #truth_loss = loss_function(truth_model_e, truth_model_c, valid_dict, copula=copula_dgp)
                        #print(truth_loss)
                        model1, model2 = dependent_train_loop_linear(model1, model2, train_dict,
                                                                     valid_dict, 15000, copula=copula)
                    else:
                        raise NotImplementedError()
                    
                    # Compute survival function
                    if model_name in ["weibull-nocop", "weibull-cop"]:
                        event_preds = predict_survival_curve(model1, X_test_th, time_bins).numpy()
                        cens_preds = predict_survival_curve(model2, X_test_th, time_bins).numpy()
                        
                    # Compute L1 (Truth vs. Model) - event/censoring
                    n_samples = X_test_th.shape[0]
                    truth_preds_e = torch.zeros((X_test_th.shape[0], time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_e[:,i] = truth_model_e.survival(time_bins[i], X_test_th)
                    l1_e = float(compute_l1_difference(truth_preds_e, event_preds,
                                                       n_samples, steps=time_bins))
                    
                    truth_preds_c = torch.zeros((X_test_th.shape[0], time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_c[:,i] = truth_model_c.survival(time_bins[i], X_test_th)
                    n_samples = X_test_th.shape[0]
                    l1_c = float(compute_l1_difference(truth_preds_c, cens_preds,
                                                       n_samples, steps=time_bins))
                    
                    print(f"Evaluated {model_name} - {dataset_version} - {round(k_tau, 3)} - {round(l1_e, 3)} - {round(l1_c, 3)}")
                    res_sr = pd.Series([model_name, dataset_version, copula_name, k_tau, l1_e, l1_c],
                                       index=["ModelName", "DatasetVersion", "Copula", "KTau", "L1E", "L1C"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results2.csv")
                    