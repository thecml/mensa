import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.abspath('../'))

import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
import torch
import random
import warnings
from data_loader import *
from utility.survival import preprocess_data
from utility.data import dotdict, array_to_tensor
import torch.optim as optim
import torch.nn as nn
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from tqdm import tqdm
from SurvivalEVAL.Evaluator import LifelinesEvaluator
import copy
from torch.utils.data import DataLoader, TensorDataset
from mensa.model import MensaNDE
from utility.config import load_config
from utility.survival import predict_survival_function, compute_l1_difference
from copula import Clayton2D, Frank2D

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
# device = "cpu" # use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = torch.device(device)

if __name__ == "__main__":
    # Load and split data
    k_tau = 0
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic.yaml")
    dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                    linear=True, copula_name="clayton",
                                                    k_tau=k_tau, device='cpu', dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    n_events = data_config['se_n_events']
    dgps = dl.dgps
    
    print(f"Goal theta: {kendall_tau_to_theta('clayton', k_tau)}")
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Define params
    batch_size = 128
    num_epochs = 100
    early_stop_epochs = 100
    
    # Make model
    model = MensaNDE(hidden_size=32, hidden_surv=32, dropout_rate=0.25,
                     device=device, n_features=train_dict['X'].shape[1], tol=1e-14).to(device)
    copula = Clayton2D(torch.tensor([2.0], dtype=dtype), device, dtype)
    # optimizer = optim.Adam([{"params": model.sumo.parameters(), "lr": 0.005},
    #                         {"params": copula.parameters(), "lr": 0.005}])
    optimizer = optim.Adam([{"params": model.sumo_e.parameters(), "lr": 0.005},
                            {"params": model.sumo_c.parameters(), "lr": 0.005},
                            {"params": copula.parameters(), "lr": 0.005}])

    # Make data loaders
    train_loader = DataLoader(TensorDataset(train_dict['X'],
                                            train_dict['T'],
                                            train_dict['E']),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_dict['X'],
                                            valid_dict['T'],
                                            valid_dict['E']),
                              batch_size=batch_size, shuffle=False)
        
    # Train model
    #copula.enable_grad()
    
    best_valid_logloss = float('-inf')
    epochs_no_improve = 0
    for epoch in tqdm(range(num_epochs), disable=True):
        for xi, ti, ei in train_loader:
            optimizer.zero_grad()
            xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
            loss = model(xi, ti, ei, None, max_iter=10000)
            loss.backward()
            # for p in copula.parameters():
            #     p.grad = p.grad * 1000.0
            #     p.grad.clamp_(torch.tensor([-1.0]).to(device), torch.tensor([1.0]).to(device))
            
            optimizer.step()
        
            # for p in copula.parameters():
            #     if p <= 0.01:
            #         with torch.no_grad():
            #             p[:] = torch.clamp(p, 0.01, 100)
                        
        print('loss:', loss.item(), '\t copula:', copula.theta.item())
        
        if epoch % 10 == 0:
            total_val_logloss = 0
            for xi, ti, ei in valid_loader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                val_logloss = model(xi, ti, ei, None, max_iter=10000)
                total_val_logloss += val_logloss
            total_val_logloss /= len(valid_loader)
            
            #print(f"{total_val_logloss} - {copula.theta}")
            print(f"{total_val_logloss}")
            
            if total_val_logloss > (best_valid_logloss + 1):
                best_valid_logloss = total_val_logloss
                epochs_no_improve = 0
            else:
                if total_val_logloss > best_valid_logloss:
                    best_valid_logloss = total_val_logloss
                epochs_no_improve = epochs_no_improve + 10
            
        if epochs_no_improve == early_stop_epochs:
            break
        
    print("Done training")
    
    # Predict survial funcion
    model_preds = predict_survival_function(model, test_dict['X'], time_bins).numpy()
    
    # Compute L1 (Truth vs. Model) - event only
    n_samples = test_dict['X'].shape[0]
    truth_preds_e = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    for i in range(time_bins.shape[0]):
        truth_preds_e[:,i] = dgps[0].survival(time_bins[i], test_dict['X'])
    l1_e = float(compute_l1_difference(truth_preds_e, model_preds,
                                       n_samples, steps=time_bins))
    
    print(f"Evaluated mensa - {True} - {round(k_tau, 3)} - {round(l1_e, 3)}")