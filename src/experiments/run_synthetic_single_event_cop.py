"""
run_synthetic_se_event_censoring.py
====================================
Experiment 1.2

Models: ["mensa-nocop", "mensa-cop", "dgp"]
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
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import SingleEventSyntheticDataLoader
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function)
from utility.data import (dotdict, format_data, format_data_as_dict_single)
from utility.config import load_config
from mensa.model import SingleMENSA

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                          make_deephit_cr, make_dsm_model, make_rsf_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Define models
MODELS = ["mensa-nocop", "mensa-cop", "dgp"]

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
    dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config, linear=linear,
                                                    copula_name=copula_name, k_tau=k_tau)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    n_events = data_config['n_events']
    dgps = dl.dgps
    
    # Make time bins
    min_time = dl.get_data()[1].min()
    max_time = dl.get_data()[1].max()
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    time_bins = torch.concat([torch.tensor([min_time], device=device, dtype=dtype), 
                              time_bins, torch.tensor([max_time], device=device, dtype=dtype)])

    # Evaluate each model
    for model_name in MODELS:
        if model_name == "mensa-nocop":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            model = SingleMENSA(n_features=n_features, copula=None, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr=lr)
        elif model_name == "mensa-cop":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            copula = Clayton2D(torch.tensor([2.0], dtype=dtype), device, dtype)
            model = SingleMENSA(n_features=n_features, copula=copula, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr=lr)
        elif model_name == "dgp":
            pass
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]
        if model_name in ['mensa-nocop', 'mensa-cop']:
            model_c, model_e = model.get_models()[0], model.get_models()[1]
            cens_preds = predict_survival_function(model_c, test_dict['X'], time_bins).detach().numpy()
            event_preds = predict_survival_function(model_e, test_dict['X'], time_bins).detach().numpy()
        elif model_name == "dgp":
            event_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                event_preds[:,i] = dgps[0].survival(time_bins[i], test_dict['X'])
            cens_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                cens_preds[:,i] = dgps[1].survival(time_bins[i], test_dict['X'])
        else:
            raise NotImplementedError()
            
        # Compute L1 (Truth vs. Model) - event/censoring
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
        
        print(f"Evaluated {model_name} - {linear} - {round(k_tau, 3)} " +
                f"- {round(l1_e, 3)} - {round(l1_c, 3)}")
        result_row = pd.Series([model_name, seed, linear, copula_name, k_tau, l1_e, l1_c],
                               index=["ModelName", "Seed", "DatasetVersion",
                                      "Copula", "KTau", "L1-E", "L1-C"])
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/synthetic_se_cop.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=result_row.keys())
        results = results.append(result_row, ignore_index=True)
        results.to_csv(filename, index=False)