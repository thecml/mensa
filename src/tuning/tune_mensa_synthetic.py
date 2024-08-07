import numpy as np
import os
import config as cfg
import torch
from utility.tuning import get_mensa_sweep_cfg
from utility.config import load_config
from data_loader import SingleEventSyntheticDataLoader
from mensa.model import MENSA
import warnings
import random

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 10
PROJECT_NAME = "mensa"

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global dataset_name
    
    dataset_name = "synthetic"
    sweep_config = get_mensa_sweep_cfg()

    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}')
    wandb.agent(sweep_id, train_mensa_model, count=N_RUNS)

def train_mensa_model():
    # Initialize a new wandb run
    config_defaults = cfg.MENSA_PARAMS
    wandb.init(config=config_defaults, group=dataset_name)
    config = wandb.config
    
    # Load data
    linear = False
    k_tau = 0.25
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_se.yaml")
    dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                    linear=linear, copula_name="clayton",
                                                    k_tau=k_tau, device=device, dtype=dtype)
    train_dict, valid_dict, _ = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
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