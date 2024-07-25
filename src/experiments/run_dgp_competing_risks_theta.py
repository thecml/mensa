"""
run_synthetic_competing_risks_theta.py
====================================
Experiment 2.1

Models: ["mensa"]
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
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import dotdict
from utility.config import load_config
from utility.loss import triple_loss
from utility.data import format_data_deephit_cr, format_hierarchical_data_cr, calculate_layer_size_hierarch, format_survtrace_data
from utility.evaluation import global_C_index, local_C_index
from mensa.model import MENSA
from Copula2 import Convex_Nested

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
from utility.data import calculate_vocab_size
from pycox.models import DeepHit, DeepHitSingle

from utility.data import theta_to_kendall_tau

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--copula_name', type=str, default="clayton")
    parser.add_argument('--linear', type=bool, default=True)
    
    args = parser.parse_args()
    seed = args.seed
    copula_name = args.copula_name
    linear = args.linear
    
    for k_tau in [0, 0.2, 0.4, 0.6, 0.8]:
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
        
        # Train model
        config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
        n_epochs = config['n_epochs']
        lr = config['lr']
        batch_size = config['batch_size']
        copula = Convex_Nested(2, 2, 1e-3, 1e-3, device)
        model = MENSA(n_features, n_events, copula=copula, device=device)
        model.fit(train_dict, valid_dict, n_epochs=1000, lr=0.005, batch_size=1024)
        
        # Get k_taus
        estimated_thetas = model.thetas
        estimated_k_taus = [tuple(theta_to_kendall_tau(copula_name, value) for value in tpl)
                            for tpl in estimated_thetas]
        model_results = pd.DataFrame(estimated_k_taus, columns=[f'KTau_{i+1}' for i in range(4)])
        model_results['KTau'] = k_tau
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/synthetic_cr_theta.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)
                    