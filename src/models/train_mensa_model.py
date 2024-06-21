import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_mensa_model
import torch
import random
import warnings
from dgp import Mensa
from multi_evaluator import MultiEventEvaluator
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

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
torch.set_default_dtype(torch.float64)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # Load data
    dl = SingleEventSyntheticDataLoader().load_data(n_samples=10000)
    num_features, cat_features = dl.get_features()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                             valid_size=0.1,
                                                                             test_size=0.2)
    
    # Make time bins
    time_bins = make_time_bins(y_train['time'], event=y_train['event'])

    # Scale data
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test,
                                               cat_features, num_features,
                                               as_array=True)
    
    # Format data
    times_tensor_train = array_to_tensor(y_train['time'], torch.float32)
    event_indicator_tensor_train = array_to_tensor(y_train['event'], torch.float32)
    covariate_tensor_train = torch.tensor(X_train).to(device)
    times_tensor_val = array_to_tensor(y_valid['time'], torch.float32)
    event_indicator_tensor_val = array_to_tensor(y_valid['event'], torch.float32)
    covariate_tensor_val = torch.tensor(X_valid).to(device)
    times_tensor_test = array_to_tensor(y_test['time'], torch.float32)
    event_indicator_tensor_test = array_to_tensor(y_test['event'], torch.float32)
    covariate_tensor_test = torch.tensor(X_test).to(device)
    
    # Define params
    batch_size = 32
    num_epochs = 1000
    early_stop_epochs = 100
    
    # Make model
    model = MensaNDE(device=device, n_features=X_train.shape[1], tol=1e-14).to(device)
    optimizer = optim.Adam([{"params": model.sumo.parameters(), "lr": 1e-3}])

    # Make data loaders
    train_loader = DataLoader(TensorDataset(covariate_tensor_train,
                                            times_tensor_train,
                                            event_indicator_tensor_train),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(covariate_tensor_val,
                                            times_tensor_val,
                                            event_indicator_tensor_val),
                              batch_size=batch_size, shuffle=False)
        
    # Train model
    best_valid_logloss = float('-inf')
    epochs_no_improve = 0
    for epoch in tqdm(range(num_epochs), disable=True):
        for xi, ti, ei in train_loader:
            optimizer.zero_grad()
            logloss = model(xi, ti, ei, max_iter=10000)
            (-logloss).backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            total_val_logloss = 0
            for xi, ti, ei in valid_loader:
                val_logloss = model(xi, ti, ei, max_iter=1000)
                total_val_logloss += val_logloss
            total_val_logloss /= len(valid_loader)
            
            print(f"Valid NLL: {-total_val_logloss}")
            
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