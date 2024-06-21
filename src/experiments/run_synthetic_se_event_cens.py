"""
run_synthetic_se_event_censoring.py
====================================
Experiment 1.2
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
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import SingleEventSyntheticDataLoader
from copula import Clayton2D, Frank2D
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
                          make_deephit_multi, make_dsm_model, make_rsf_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.25]
MODELS = ["mensa-cop", "mensa-nocop"]
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

                 # Evaluate each model
                for model_name in MODELS:
                    if model_name == "mensa-nocop":
                        model1 = Weibull_log_linear(N_FEATURES, 2, 1, device, dtype)
                        model2 = Weibull_log_linear(N_FEATURES, 2, 1, device, dtype)
                        model1, model2 = independent_train_loop_linear(model1, model2, train_dict,
                                                                       valid_dict, 1000)
                    elif model_name == "mensa-cop":
                        model1 = Weibull_log_linear(N_FEATURES, 2, 1, device, dtype)
                        model2 = Weibull_log_linear(N_FEATURES, 2, 1, device, dtype)
                        copula = Clayton2D(torch.tensor([1], dtype=dtype), device, dtype)
                        model1, model2, copula = dependent_train_loop_linear(model1, model2, train_dict,
                                                                             valid_dict, 1000, copula=copula)
                    else:
                        raise NotImplementedError()
                    
                    # Compute survival function
                    if model_name in ["mensa-nocop", "mensa-cop"]:
                        event_preds = predict_survival_function(model1, test_dict['X'], time_bins).numpy()
                        cens_preds = predict_survival_function(model2, test_dict['X'], time_bins).numpy()
                        
                    # Compute L1 (Truth vs. Model) - event/censoring
                    n_samples = test_dict['X'].shape[0]
                    truth_preds_e = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_e[:,i] = dgps[0].survival(time_bins[i], test_dict['X'])
                    l1_e = float(compute_l1_difference(truth_preds_e, event_preds,
                                                       n_samples, steps=time_bins))
                    
                    truth_preds_c = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                    for i in range(time_bins.shape[0]):
                        truth_preds_c[:,i] = dgps[1].survival(time_bins[i], test_dict['X'])
                    l1_c = float(compute_l1_difference(truth_preds_c, cens_preds,
                                                       n_samples, steps=time_bins))
                    
                    print(f"Evaluated {model_name} - {dataset_version} - {round(k_tau, 3)} " +
                          f"- {round(l1_e, 3)} - {round(l1_c, 3)}")
                    res_sr = pd.Series([model_name, dataset_version, copula_name, k_tau, l1_e, l1_c],
                                       index=["ModelName", "DatasetVersion", "Copula", "KTau", "L1-E", "L1-C"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results2.csv")
                    