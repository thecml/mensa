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
from utility.survival import scale_data
from hierarchical.data_settings import all_settings
from hierarchical.hyperparams import all_hyperparams
from hierarchical import util

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
    train_data[0] = scale_data(train_data[0].values, norm_mode='standard')
    test_data[0] = scale_data(test_data[0].values, norm_mode='standard')
    valid_data[0] = scale_data(valid_data[0].values, norm_mode='standard')
    
    data_settings = all_settings[dataset_name]
    data_settings['min_time'], data_settings['max_time'] = dl.min_time, dl.max_time
    
    hyperparams = all_hyperparams[dataset_name][approach]
    verbose = True
    mod = util.get_model_and_output(approach, train_data, test_data, valid_data, data_settings, hyperparams, verbose)
    
    print(dataset_name + ',', approach, hyperparams)