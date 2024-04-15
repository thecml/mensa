import tensorflow as tf
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from tools.baysurv_trainer import Trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, split_time_event
from tools.baysurv_builder import make_mlp_model, make_vi_model, make_mcd_model, make_sngp_model
from pathlib import Path
from utility.survival import (calculate_event_times, calculate_percentiles, convert_to_structured,
                              compute_survival_curve, compute_nondeterministic_survival_curve)
from utility.training import make_stratified_split
from time import time
from pycox.evaluation import EvalSurv
import math
from utility.survival import coverage
from scipy.stats import chisquare
import torch
import config as cfg
from data_loader import get_data_loader
from utility.survival import make_time_bins
from utility.data import dotdict
from multi_evaluator import MultiEventEvaluator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

DATASETS = ["als", "mimic", "seer", "rotterdam"]
MODELS = ["mensa"]
N_EPOCHS = 1

test_results = pd.DataFrame()
training_results = pd.DataFrame()

if __name__ == "__main__":
    # For each dataset, train models and plot scores
    for dataset_name in DATASETS:
        # Load training parameters
        config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        activation_fn = config['activiation_fn']
        layers = config['network_layers']
        l2_reg = config['l2_reg']
        batch_size = config['batch_size']
        early_stop = config['early_stop']
        patience = config['patience']
        n_samples_train = config['n_samples_train']
        n_samples_valid = config['n_samples_valid']
        n_samples_test = config['n_samples_test']
        
        # Load data
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        data_packages = dl.split_data()
        n_events = 2
        
        # Split data
        train_data = [data_packages[0][0], data_packages[0][1], data_packages[0][2]]
        test_data = [data_packages[1][0], data_packages[1][1], data_packages[1][2]]
        valid_data = [data_packages[2][0], data_packages[2][1], data_packages[2][2]]

        # Scale data TODO: Use the Preprocessor here
        train_data[0] = scale_data(train_data[0].values, norm_mode='standard')
        test_data[0] = scale_data(test_data[0].values, norm_mode='standard')
        valid_data[0] = scale_data(valid_data[0].values, norm_mode='standard')

        # Make event times
        time_bins = make_time_bins(train_data[1], event=train_data[2])
        
        # Format data
        data_train = pd.DataFrame(train_data[0])
        data_train["y1_time"] = pd.Series(train_data[1][:,0])
        data_train["y2_time"] = pd.Series(train_data[1][:,1])
        data_train["y1_event"] = pd.Series(train_data[2][:,0])
        data_train["y2_event"] = pd.Series(train_data[2][:,1])
        data_valid = pd.DataFrame(valid_data[0])
        data_valid["y1_time"] = pd.Series(valid_data[1][:,0])
        data_valid["y2_time"] = pd.Series(valid_data[1][:,1])
        data_valid["y1_event"] = pd.Series(valid_data[2][:,0])
        data_valid["y2_event"] = pd.Series(valid_data[2][:,1])
        data_test = pd.DataFrame(test_data[0])
        data_test["y1_time"] = pd.Series(test_data[1][:,0])
        data_test["y2_time"] = pd.Series(test_data[1][:,1])
        data_test["y1_event"] = pd.Series(test_data[2][:,0])
        data_test["y2_event"] = pd.Series(test_data[2][:,1])
        
        # Make model
        model = None
        
        # Train model
        config = dotdict(cfg.PARAMS_MENSA)
        num_features = train_data[0].shape[1]
        num_time_bins = len(time_bins)
        #model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
        #model = train_mtlr_model(model, data_train, data_valid, time_bins,
        #                         config, random_state=0, reset_model=True, device=device)
        
        test_data = torch.tensor(test_data[0], dtype=torch.float, device=device)
        #survival_outputs, _, ensemble_outputs = make_ensemble_mtlr_prediction(model, test_data,
        #                                                                      time_bins, config)
        #surv_preds = survival_outputs.numpy()
        surv_preds = None
        
        evaluator = MultiEventEvaluator(test_data[0], train_data[0], model, config, device)
        surv_preds = evaluator.predict_survival_curves()
        for event_id in range(n_events):
            ci = evaluator.calculate_ci(surv_preds[event_id], event_id)
            mae = evaluator.calculate_mae(surv_preds[event_id], event_id, method="Hinge")
            print(f"Event {event_id} - CI={round(ci, 2)} - MAE={round(mae, 2)}")