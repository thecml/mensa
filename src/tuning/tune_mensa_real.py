import numpy as np
import os
import argparse
import pandas as pd
import config as cfg
import torch
from utility.tuning import get_mensa_sweep_cfg
from utility.data import dotdict
import data_loader
from utility.config import load_config
from utility.survival import make_time_bins, preprocess_data, convert_to_structured
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.evaluation import LifelinesEvaluator
from data_loader import SingleEventSyntheticDataLoader
from copula import Clayton2D
from mensa.model import MENSA
from utility.survival import compute_l1_difference
import warnings
import random
from data_loader import get_data_loader

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 100
PROJECT_NAME = "mensa"

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global dataset_name
    
    dataset_name = "seer_se"
    sweep_config = get_mensa_sweep_cfg()

    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}')
    wandb.agent(sweep_id, train_mensa_model, count=N_RUNS)

def train_mensa_model():
    # Initialize a new wandb run
    config_defaults = cfg.MENSA_PARAMS
    wandb.init(config=config_defaults, group=dataset_name)
    config = wandb.config
    
    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=0)
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    n_features = train_dict['X'].shape[1]

    # Train model
    n_epochs = config['n_epochs']
    lr = config['lr']
    layers = config['layers']
    dropout = config['dropout']
    batch_size = config['batch_size']
    copula = Clayton2D(torch.tensor([2.0]).type(dtype), device, dtype)
    model = MENSA(n_features=n_features, n_events=2, layers=layers, dropout=dropout,
                  copula=copula, device=device)
    model.fit(train_dict, valid_dict, n_epochs=n_epochs,
              lr=lr, batch_size=batch_size, use_wandb=True) # log to wandb
    
if __name__ == "__main__":
    main()