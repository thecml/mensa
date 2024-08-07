import numpy as np
import os
import config as cfg
import torch
from utility.tuning import get_mensa_sweep_cfg
from utility.config import load_config
from utility.survival import preprocess_data
from data_loader import get_data_loader
from mensa.model import MENSA
import warnings
import random
import pandas as pd

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 1
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
    layers = config['layers']
    lr = config['lr']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    k = config['k']
    lr_dict = {'network': lr, 'copula': 0.01}
    model = MENSA(n_features, n_events=2, n_dists=k, layers=layers, device=device)
    model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr_dict=lr_dict,
              batch_size=batch_size, use_wandb=True)
    
if __name__ == "__main__":
    main()