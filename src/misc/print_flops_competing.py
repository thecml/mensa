"""
run_real_multi_event.py
====================================
Models: ['deepsurv', 'hierarch', 'mensa']
"""

# 3rd party
import pandas as pd
import numpy as np
import sys, os

from torchmtlr.model import MTLRCR
from utility.mtlr import mtlr
sys.path.append(os.path.abspath('../'))

import config as cfg
import torch
import random
import warnings
import argparse
import os
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from utility.survival import (convert_to_structured, make_time_bins, preprocess_data)
from utility.data import dotdict
from utility.config import load_config
from utility.data import calculate_layer_size_hierarch
from utility.evaluation import global_C_index, local_C_index
from mensa.model import MENSA

# SOTA
from sota_models import (make_deephit_cr, make_deephit_single, make_dsm_model, train_deepsurv_model, make_deepsurv_prediction, DeepSurv)
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from utility.data import (format_hierarchical_data_me, calculate_layer_size_hierarch)
from data_loader import CompetingRiskSyntheticDataLoader, get_data_loader
from fvcore.nn import FlopCountAnalysis

import logging
logging.basicConfig(level=logging.ERROR)

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float32
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0

if __name__ == "__main__":
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_cr.yaml")
    data_config['n_samples'] = 1000
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config=data_config,
                                                      linear=True, copula_name=None,
                                                      k_tau=0, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=SEED)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    n_events = 2
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Format data to work easier with sksurv API
    X_train = pd.DataFrame(train_dict['X'], columns=[f'X{i}' for i in range(n_features)])
    X_valid = pd.DataFrame(valid_dict['X'], columns=[f'X{i}' for i in range(n_features)])
    X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(n_features)])
    y_train = convert_to_structured(train_dict['T'], train_dict['E'])
    y_valid = convert_to_structured(valid_dict['T'], valid_dict['E'])
    y_test = convert_to_structured(test_dict['T'], test_dict['E'])
    
    # DeepSurv
    config = dotdict(cfg.DEEPSURV_PARAMS)
    models = []
    for i in range(n_events):
        model = DeepSurv(in_features=n_features, config=config)
        model.to(device)
        models.append(model)
    flops_total = 0
    for model in models:
        model.eval()
        flops = FlopCountAnalysis(model, test_dict['X'][0].unsqueeze(0).to(device))
        flops_total += flops.total()
    print(f"DeepSurv FLOPs: {flops_total}")
    
    # DeepHit
    config = dotdict(cfg.DEEPHIT_PARAMS)
    max_time = torch.tensor([dl.get_data()[1].max()], dtype=dtype, device=device)
    time_bins_dh = time_bins
    if max_time not in time_bins_dh:
        time_bins_dh = torch.concat([time_bins_dh, max_time], dim=0)
    model = make_deephit_cr(in_features=n_features, out_features=len(time_bins_dh),
                            num_risks=n_events, duration_index=time_bins_dh, config=config)
    flops_total = 0
    model.net.eval()
    flops = FlopCountAnalysis(model.net, test_dict['X'][0].unsqueeze(0).to(device))
    flops_total += flops.total()
    print(f"DeepHit FLOPs: {flops_total}")
    
    # MTLRCR
    config = dotdict(cfg.MTLR_PARAMS)
    num_time_bins = len(time_bins.cpu().numpy()) + 1
    model = MTLRCR(in_features=n_features, num_time_bins=num_time_bins, num_events=n_events)
    flops_total = 0
    model.to(device)
    model.eval()
    flops = FlopCountAnalysis(model, test_dict['X'][0].unsqueeze(0).to(device))
    flops_total += flops.total()
    print(f"MTLR FLOPs: {flops_total}")
    
    # DSM
    config = dotdict(cfg.DSM_PARAMS)
    n_iter = config['n_iter']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    model = make_dsm_model(config)
    model.fit(train_dict['X'].cpu().numpy(), train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy(),
              val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'].cpu().numpy(), valid_dict['T'].cpu().numpy()),
              learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
    flops_total = 0
    flops = FlopCountAnalysis(model.torch_model, test_dict['X'][0].unsqueeze(0).to("cpu"))
    flops_total += flops.total()
    print(f"DSM FLOPs : {flops_total}")

    # MENSA
    config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
    n_epochs = config['n_epochs']
    n_dists = config['n_dists']
    lr = config['lr']
    batch_size = config['batch_size']
    layers = config['layers']
    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']
    model = MENSA(n_features, layers=layers, dropout_rate=dropout_rate,
                  n_events=n_events, n_dists=n_dists, trajectories=None,
                  device=device)
    model.model.to(device)
    model.model.eval()
    flops = FlopCountAnalysis(model.model, test_dict['X'][0].unsqueeze(0).to(device))
    print(f"MENSA FLOPs: {flops.total()}")
    