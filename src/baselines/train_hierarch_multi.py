import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_multi_model
import torch
import random
import warnings
from hierarchical.data_settings import synthetic_settings
from hierarchical import util
from utility.data import dotdict
import config as cfg
from utility.hierarchical import format_hyperparams
from utility.config import load_config
from data_loader import CompetingRiskSyntheticDataLoader
from utility.survival import make_times_hierarchical

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set up precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Set up device
device = "cpu" # use CPU
device = torch.device(device)

approach = 'direct_full' #'hierarch_full'
dataset_name = 'Synthetic'

if __name__ == "__main__":
    # Load data
    data_config = load_config(cfg.DATA_CONFIGS_DIR, f"synthetic.yaml")
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config, k_tau=0.25,
                                                      n_samples=1000, n_features=10)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    
    # Format data for Hierarchical model
    data_settings = synthetic_settings # if Synthetic
    num_bins = data_settings['num_bins']
    train_times = np.stack([train_dict['T1'], train_dict['T2'], train_dict['T3']], axis=1)
    valid_times = np.stack([valid_dict['T1'], valid_dict['T2'], valid_dict['T3']], axis=1)
    test_times = np.stack([test_dict['T1'], test_dict['T2'], test_dict['T3']], axis=1)
    train_event_bins = make_times_hierarchical(train_times, num_bins=num_bins)
    valid_event_bins = make_times_hierarchical(valid_times, num_bins=num_bins)
    test_event_bins = make_times_hierarchical(test_times, num_bins=num_bins)
    train_events = np.array(pd.get_dummies(train_dict['E']))
    valid_events = np.array(pd.get_dummies(valid_dict['E']))
    test_events = np.array(pd.get_dummies(test_dict['E']))
    
    train_data = [train_dict['X'], train_event_bins, train_events]
    valid_data = [valid_dict['X'], valid_event_bins, valid_events]
    test_data = [test_dict['X'], test_event_bins, test_events]
    
    data_settings['min_time'], data_settings['max_time'] = train_event_bins.min(), train_event_bins.max()
    
    if 'direct_full':
        params = cfg.DIRECT_FULL_PARAMS
    else:
        params = cfg.HIERARCH_FULL_PARAMS
    hyperparams = format_hyperparams(params)
    verbose = params['verbose']
    test_curves = util.get_model_and_output(approach, train_data, test_data, valid_data,
                                            data_settings, hyperparams, verbose)
    # Evaluation
    # test_curves = List of arrays with shape (n_samples, n_bins) with len n_events
    
    print(approach, hyperparams)