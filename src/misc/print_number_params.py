"""
run_real_multi_event.py
====================================
Models: ['deepsurv', 'hierarch', 'mensa']
"""

# 3rd party
import pandas as pd
import numpy as np
import sys, os

from utility.mtlr import make_mtlr_prediction, mtlr, train_mtlr_model
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
from utility.data import dotdict, format_data_deephit_multi, format_data_deephit_single
from utility.config import load_config
from utility.data import calculate_layer_size_hierarch
from utility.evaluation import global_C_index, local_C_index
from mensa.model import MENSA
from mensa.model_trajectory import MENSA_trajectory

# SOTA
from sota_models import (make_cox_model, make_coxboost_model, make_deephit_single, make_dsm_model,
                         make_rsf_model, train_deephit_model, train_deepsurv_model, make_deepsurv_prediction, DeepSurv)
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from utility.data import (format_hierarchical_data_me, calculate_layer_size_hierarch)
from data_loader import get_data_loader

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
MODELS = ["coxph", "coxboost", "rsf", "deepsurv", "deephit", "mtlr", "dsm" , "hierarch", 'mensa']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='proact_me')
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    
    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    n_events = dl.n_events
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    event_cols = [f'e{i+1}' for i in range(n_events)]
    time_cols = [f't{i+1}' for i in range(n_events)]
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    train_dict['E'] = torch.tensor(train_dict['E'], device=device, dtype=torch.int64)
    train_dict['T'] = torch.tensor(train_dict['T'], device=device, dtype=torch.int64)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    valid_dict['E'] = torch.tensor(valid_dict['E'], device=device, dtype=torch.int64)
    valid_dict['T'] = torch.tensor(valid_dict['T'], device=device, dtype=torch.int64)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    test_dict['E'] = torch.tensor(test_dict['E'], device=device, dtype=torch.int64)
    test_dict['T'] = torch.tensor(test_dict['T'], device=device, dtype=torch.int64)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Train models
    for model_name in MODELS:
        if model_name == "coxph":
            config = dotdict(cfg.COXPH_PARAMS)
            trained_models = []
            for i in range(n_events):
                train_times = train_dict['T'][:,i].cpu().numpy()
                train_events = train_dict['E'][:,i].cpu().numpy()
                y_train = convert_to_structured(train_times, train_events)
                model = make_cox_model(config)
                model.fit(train_dict['X'].cpu(), y_train)
                trained_models.append(model)
        elif model_name == "coxboost":
            config = dotdict(cfg.COXBOOST_PARAMS)
            trained_models = []
            for i in range(n_events):
                train_times = train_dict['T'][:,i].cpu().numpy()
                train_events = train_dict['E'][:,i].cpu().numpy()
                y_train = convert_to_structured(train_times, train_events)
                model = make_coxboost_model(config)
                model.fit(train_dict['X'].cpu(), y_train)
                trained_models.append(model)
        elif model_name == "rsf":
            config = dotdict(cfg.RSF_PARAMS)
            trained_models = []
            for i in range(n_events):
                train_times = train_dict['T'][:,i].cpu().numpy()
                train_events = train_dict['E'][:,i].cpu().numpy()
                y_train = convert_to_structured(train_times, train_events)
                model = make_rsf_model(config)
                model.fit(train_dict['X'].cpu(), y_train)
                trained_models.append(model)
        elif model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = DeepSurv(in_features=n_features, config=config)
                data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
                data_train['time'] = train_dict['T'][:,i].cpu().numpy()
                data_train['event'] = train_dict['E'][:,i].cpu().numpy()
                data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
                data_valid['time'] = valid_dict['T'][:,i].cpu().numpy()
                data_valid['event'] = valid_dict['E'][:,i].cpu().numpy()
                model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                             random_state=0, reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "deephit":
            config = dotdict(cfg.DEEPHIT_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = make_deephit_single(in_features=n_features, out_features=len(time_bins),
                                            time_bins=time_bins.cpu().numpy(), device=device, config=config)
                labtrans = model.label_transform
                train_data, valid_data, out_features, duration_index = format_data_deephit_multi(train_dict, valid_dict, labtrans, risk=i)
                model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                            (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
                trained_models.append(model)
        elif model_name == "mtlr":
            config = dotdict(cfg.MTLR_PARAMS)
            trained_models = []
            for i in range(n_events):
                data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
                data_train['time'] = train_dict['T'][:,i].cpu().numpy()
                data_train['event'] = train_dict['E'][:,i].cpu().numpy()
                data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
                data_valid['time'] = valid_dict['T'][:,i].cpu().numpy()
                data_valid['event'] = valid_dict['E'][:,i].cpu().numpy()
                num_time_bins = len(time_bins)
                model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
                model = train_mtlr_model(model, data_train, data_valid, time_bins.cpu().numpy(),
                                         config, random_state=0, dtype=dtype,
                                         reset_model=True, device=device)
                trained_models.append(model)
        elif model_name == "dsm":
            config = dotdict(cfg.DSM_PARAMS)
            n_iter = config['n_iter']
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
            trained_models = []
            for i in range(n_events):
                model = make_dsm_model(config)
                model.fit(train_dict['X'].cpu().numpy(), train_dict['T'][:,i].cpu().numpy(), train_dict['E'][:,i].cpu().numpy(),
                          val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'][:,i].cpu().numpy(), valid_dict['T'][:,i].cpu().numpy()),
                          learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
                trained_models.append(model)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name}.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_me(train_dict, valid_dict, test_dict, n_time_bins)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = n_time_bins
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            n_epochs = config['n_epochs']
            n_dists = config['n_dists']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            model = MENSA(n_features, layers=layers, n_events=n_events,
                          n_dists=n_dists, device=device)
            model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
                      patience=10, batch_size=batch_size, verbose=False)
        else:
            raise NotImplementedError()
        
    
        # Print number of trainable params.
        sum_params = 0
        if model_name == "coxph":
            for trained_model in trained_models:
                sum_params += len(trained_model.coef_)
        elif model_name == "coxboost":
            for trained_model in trained_models:
                num_trees = trained_model.n_estimators
                max_depth = trained_model.max_depth
                params_per_tree = (2 ** (max_depth + 1)) - 1
                total_params = num_trees * params_per_tree
                sum_params += total_params
        elif model_name == "rsf":
            for trained_model in trained_models:
                num_trees = trained_model.n_estimators
                max_depth = trained_model.max_depth
                params_per_tree = (2 ** (max_depth + 1)) - 1
                total_params = num_trees * params_per_tree
                sum_params += total_params
        elif model_name == "deepsurv":
            for trained_model in trained_models:
                total_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
                sum_params += total_params
        elif model_name == "deephit":
            for trained_model in trained_models:
                total_params = sum(p.numel() for p in trained_model.net.parameters() if p.requires_grad)
                sum_params += total_params
        elif model_name == "mtlr":
            for trained_model in trained_models:
                total_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
                sum_params += total_params
        elif model_name == "dsm":
            for trained_model in trained_models:
                total_params = sum(p.numel() for p in trained_model.torch_model.parameters() if p.requires_grad)
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
        