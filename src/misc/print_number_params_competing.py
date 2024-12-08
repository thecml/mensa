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
from torchmtlr.utils import encode_mtlr_format
from utility.mtlr import make_mtlr_prediction, mtlr, train_mtlr_cr, train_mtlr_model
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
from utility.data import dotdict, format_data_deephit_competing, format_data_deephit_multi, format_data_deephit_single, format_hierarchical_data_cr
from utility.config import load_config
from utility.data import calculate_layer_size_hierarch
from utility.evaluation import global_C_index, local_C_index
from mensa.model import MENSA

# SOTA
from sota_models import (make_cox_model, make_coxboost_model, make_deephit_cr, make_deephit_single, make_dsm_model,
                         make_rsf_model, train_deephit_model, train_deepsurv_model, make_deepsurv_prediction, DeepSurv)
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from utility.data import (format_hierarchical_data_me, calculate_layer_size_hierarch)
from data_loader import CompetingRiskSyntheticDataLoader, get_data_loader

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ["deepsurv", "deephit", "mtlrcr", "dsm" , "hierarch", 'mensa']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    seed = args.seed
    
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_cr.yaml")
    data_config['n_samples'] = 1000
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config=data_config,
                                                      linear=True, copula_name=None,
                                                      k_tau=0, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
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
    
    # Train models
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            trained_models = []
            for i in range(n_events):
                data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
                data_train['time'] = train_dict['T'].cpu().numpy()
                data_train['event'] = (train_dict['E'].cpu().numpy() == i+1)*1.0
                data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
                data_valid['time'] = valid_dict['T'].cpu().numpy()
                data_valid['event'] = (valid_dict['E'].cpu().numpy() == i+1)*1.0
                model = DeepSurv(in_features=n_features, config=config)
                model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                             random_state=0, reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "deephit":
            config = dotdict(cfg.DEEPHIT_PARAMS)
            max_time = torch.tensor([dl.get_data()[1].max()], dtype=dtype, device=device)
            time_bins_dh = time_bins
            if max_time not in time_bins_dh:
                time_bins_dh = torch.concat([time_bins_dh, max_time], dim=0)
            model = make_deephit_cr(in_features=n_features, out_features=len(time_bins_dh),
                                    num_risks=n_events, duration_index=time_bins_dh, config=config)
            train_data, valid_data, out_features, duration_index = format_data_deephit_competing(train_dict, valid_dict, time_bins_dh)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "mtlrcr":
            train_events = train_dict['E'].type(torch.int64).cpu().numpy()
            valid_events = valid_dict['E'].type(torch.int64).cpu().numpy()
            y_train = encode_mtlr_format(train_dict['T'], train_events, time_bins.cpu().numpy())
            y_valid = encode_mtlr_format(valid_dict['T'], valid_events, time_bins.cpu().numpy())            
            num_time_bins = len(time_bins.cpu().numpy()) + 1
            config = dotdict(cfg.MTLRCR_PARAMS)
            model = MTLRCR(in_features=n_features, num_time_bins=num_time_bins, num_events=n_events)
            model = train_mtlr_cr(train_dict['X'], y_train, valid_dict['X'], y_valid,
                                  model, time_bins, num_epochs=config['num_epochs'],
                                  lr=config['lr'], batch_size=config['batch_size'],
                                  verbose=True, device=device, C1=config['c1'],
                                  early_stop=config['early_stop'], patience=config['patience'])
        elif model_name == "dsm":
            config = dotdict(cfg.DSM_PARAMS)
            n_iter = config['n_iter']
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
            model = make_dsm_model(config)
            model.fit(train_dict['X'].cpu().numpy(), train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy(),
                      val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'].cpu().numpy(), valid_dict['T'].cpu().numpy()),
                      learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_cr.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                                                            n_time_bins, n_events, censoring_event=False)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = len(time_bins)
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = 10
            n_dists = config['n_dists']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            weight_decay = config['weight_decay']
            dropout_rate = config['dropout_rate']
            model = MENSA(n_features, layers=layers, dropout_rate=dropout_rate,
                          n_events=n_events, n_dists=n_dists, device=device)
            model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
                      weight_decay=weight_decay, patience=10,
                      batch_size=batch_size, verbose=False)
        else:
            raise NotImplementedError()
    
        # Print number of trainable params.
        sum_params = 0
        if model_name == "deepsurv":
            for trained_model in trained_models:
                total_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
                sum_params += total_params
        elif model_name == "deephit":
            total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
            sum_params += total_params
        elif model_name == "mtlrcr":
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            sum_params += total_params
        elif model_name == "dsm":
            total_params = sum(p.numel() for p in model.torch_model.parameters() if p.requires_grad)
            sum_params += total_params
        elif model_name == "hierarch":
            sum_params += sum(p.numel() for p in model.main_layers[0].parameters() if p.requires_grad)
            for i in range(n_events):
                sum_params += sum(p.numel() for p in model.event_networks[i].parameters() if p.requires_grad)
        elif model_name == "mensa":
            sum_params += sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        else:
            raise NotImplementedError()
        
        print(f"{model_name}: {sum_params}")
        