"""
run_real_competing_risks.py
====================================
Models: ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa']
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
from utility.survival import (make_time_bins, preprocess_data)
from utility.data import dotdict
from utility.config import load_config
from utility.data import (format_data_deephit_competing, format_hierarchical_data_cr, calculate_layer_size_hierarch)
from utility.evaluation import global_C_index, local_C_index
from data_loader import get_data_loader
from mensa.model import MENSA

# SOTA
from sota_models import (make_deephit_cr, make_dsm_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)
from utility.mtlr import train_mtlr_cr
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from torchmtlr.utils import encode_mtlr_format
from torchmtlr.model import MTLRCR, mtlr_survival

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float32
torch.set_default_dtype(dtype)

SEED = 0
DATASET = "seer_cr"

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Load and split data
    dl = get_data_loader(DATASET)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=SEED)
    n_events = dl.n_events
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    n_features = train_dict['X'].shape[1]
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

    # Train model
    config = load_config(cfg.MENSA_CONFIGS_DIR, f"{DATASET.partition('_')[0]}.yaml")
    n_epochs = config['n_epochs']
    n_dists = config['n_dists']
    lr = config['lr']
    batch_size = config['batch_size']
    layers = config['layers']
    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']
    model = MENSA(n_features, layers=layers, dropout_rate=dropout_rate,
                  n_events=n_events, n_dists=n_dists, device=device)
    model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
              weight_decay=weight_decay, patience=20,
              batch_size=batch_size, verbose=True)
    
    # Make predictions
    all_preds = []
    for i in range(n_events):
        model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=i+1)
        model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        all_preds.append(model_preds)
        
    # Calculate local and global CI
    y_test_time = np.stack([test_dict['T'].cpu().numpy() for _ in range(n_events)], axis=1)
    y_test_event = np.stack([np.array((test_dict['E'].cpu().numpy() == i+1)*1.0)
                            for i in range(n_events)], axis=1)
    all_preds_arr = [df.to_numpy() for df in all_preds]
    global_ci = global_C_index(all_preds_arr, y_test_time, y_test_event)
    local_ci = local_C_index(all_preds_arr, y_test_time, y_test_event)
            
    # Make evaluation for each event
    model_results = pd.DataFrame()
    for event_id, surv_preds in enumerate(all_preds):
        n_train_samples = len(train_dict['X'])
        n_test_samples= len(test_dict['X'])
        y_train_time = train_dict['T']
        y_train_event = (train_dict['E'] == event_id+1)*1.0
        y_test_time = test_dict['T']
        y_test_event = (test_dict['E'] == event_id+1)*1.0
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        mae = lifelines_eval.mae(method="Margin")
        d_calib = lifelines_eval.d_calibration()[0]
        
        metrics = [ci, ibs, mae, d_calib, global_ci, local_ci]
        print(metrics)
        