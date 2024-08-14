"""
run_real_single_event.py
====================================
Datasets: seer_se, support_se, mimic_se
Models: ["deepsurv", "deephit", "dsm", "mtlr", "dcsurvival", "mensa", "mensa-nocop"]
"""
import sys, os
sys.path.append(os.path.abspath('../'))
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
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              predict_survival_function)
from utility.data import dotdict
from utility.config import load_config
from mensa.model import MENSA
from utility.data import format_data_deephit_single
from copula import Convex_bivariate
from data_loader import get_data_loader

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_dsm_model, make_rsf_model, train_deepsurv_model,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='seer_cr')
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    
    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    n_events = dl.n_events
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    n_samples = train_dict['X'].shape[0]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    # Format data to work easier with sksurv API
    n_features = train_dict['X'].shape[1]
    X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    X_valid = pd.DataFrame(valid_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    y_train = convert_to_structured(train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy())
    y_valid = convert_to_structured(valid_dict['T'].cpu().numpy(), valid_dict['E'].cpu().numpy())
    y_test = convert_to_structured(test_dict['T'].cpu().numpy(), test_dict['E'].cpu().numpy())
    
    config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
    n_epochs = config['n_epochs']
    n_dists = config['n_dists']
    lr = config['lr']
    batch_size = config['batch_size']
    layers = config['layers']
    model = MENSA(n_features, layers=layers, n_dists=n_dists, n_events=n_events+1,
                  copula=None, device=device)
    lr_dict = {'network': lr, 'copula': lr}
    model.fit(train_dict, valid_dict, optimizer='adam', verbose=False, n_epochs=n_epochs,
                patience=10, batch_size=batch_size, lr_dict=lr_dict)
        
    # Predict
    model_preds = model.predict(test_dict['X'], time_bins, risk=0) # use event preds
        
    # Make evaluation
    model_results = pd.DataFrame()
    surv_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
    
    y_train_time = train_dict['T']
    y_train_event = (train_dict['E'])*1.0
    y_test_time = test_dict['T']
    y_test_event = (test_dict['E'])*1.0
    lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                        y_train_time, y_train_event)
    
    ci = lifelines_eval.concordance()[0]
    ibs = lifelines_eval.integrated_brier_score()
    mae_hinge = lifelines_eval.mae(method="Hinge")
    mae_margin = lifelines_eval.mae(method="Margin")
    mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
    d_calib = lifelines_eval.d_calibration()[0]
    
    metrics = [ci, ibs, mae_hinge, mae_margin, mae_pseudo, d_calib]
    print(metrics)