import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_multi_model
import torch
import random
import warnings
from models import MultiEventCoxPH
from multi_evaluator import MultiEventEvaluator
from data_loader import SyntheticDataLoader
from utility.survival import impute_and_scale
from hierarchical.data_settings import all_settings
from hierarchical.hyperparams import all_hyperparams
from hierarchical import util
from utility.data import dotdict
import config as cfg
from utility.hierarch import format_hyperparams

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

approach = 'direct_full' #'hierarch_full'
dataset_name = 'Synthetic'

if __name__ == "__main__":
    # Load data
    dl = SyntheticDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    data_packages = dl.split_data()
    n_events = 2
    
    train_data = [data_packages[0][0], data_packages[0][1], data_packages[0][2]]
    test_data = [data_packages[1][0], data_packages[1][1], data_packages[1][2]]
    valid_data = [data_packages[2][0], data_packages[2][1], data_packages[2][2]]

    # Scale data
    train_data[0] = impute_and_scale(train_data[0].values, norm_mode='standard')
    test_data[0] = impute_and_scale(test_data[0].values, norm_mode='standard')
    valid_data[0] = impute_and_scale(valid_data[0].values, norm_mode='standard')
    
    data_settings = cfg.SYNTHETIC_DATA_SETTINGS # if Synthetic
    data_settings['min_time'], data_settings['max_time'] = dl.min_time, dl.max_time
    
    if 'direct_full':
        params = cfg.PARAMS_DIRECT_FULL
    else:
        params = cfg.PARAMS_HIERARCH_FULL
    hyperparams = format_hyperparams(params)
    verbose = params['verbose']
    test_curves = util.get_model_and_output(approach, train_data, test_data, valid_data,
                                            data_settings, hyperparams, verbose)
    # Evaluation
    # test_curves = List of arrays with shape (n_samples, n_bins) with len n_events
    
    print(dataset_kname + ',', approach, hyperparams)