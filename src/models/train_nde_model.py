import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_mensa_model
import torch
import random
import warnings
from models import Mensa
from multi_evaluator import MultiEventEvaluator
from data_loader import *
from utility.survival import preprocess_data
from utility.data import dotdict, array_to_tensor
import torch.optim as optim
import torch.nn as nn
from copulas import Clayton
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival, MultiNDESurvival, SurvNDE
from tqdm import tqdm
from utility.evaluator import LifelinesEvaluator
import copy
from models import Weibull_linear, Weibull_nonlinear
from torch.utils.data import DataLoader, TensorDataset
import math

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
torch.set_default_dtype(torch.float32)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # Load data
    dl = LinearSyntheticDataLoader().load_data(n_samples=20000, copula_name="Clayton")
    num_features, cat_features = dl.get_features()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                             valid_size=0.5)
    #beta, beta_shape_e, beta_scale_e = dl.params
    beta_e, beta_c = dl.params
    
    # Make time bins
    time_bins = make_time_bins(y_train['time'], event=y_train['event'])

    # Scale data
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test,
                                               cat_features, num_features,
                                               as_array=True)
    
    # Format data
    times_tensor_train = array_to_tensor(y_train['time'], torch.float32)
    event_indicator_tensor_train = array_to_tensor(y_train['event'], torch.float32)
    covariate_tensor_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    times_tensor_val = array_to_tensor(y_valid['time'], torch.float32)
    event_indicator_tensor_val = array_to_tensor(y_valid['event'], torch.float32)
    covariate_tensor_val = torch.tensor(X_valid, dtype=torch.float32).to(device)
    times_tensor_test = array_to_tensor(y_test['time'], torch.float32)
    event_indicator_tensor_test = array_to_tensor(y_test['event'], torch.float32)
    covariate_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Define model
    num_epochs = 1000 # 5000
    early_stop_epochs = 10
    hidden_size = 64
    hidden_surv = 64
    dropout_rate = 0.25
    batch_size = 32
    model = MultiNDESurvival(device=device, num_features=X_train.shape[1], tol=1e-14,
                             hidden_size=hidden_size, hidden_surv=hidden_surv,
                             dropout_rate=dropout_rate).to(device)
    learning_rate = 1e-3
    optimizer = optim.Adam([{"params": model.sumo.parameters(), "lr": learning_rate}])
    
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
            
            print(-total_val_logloss)
            
            if total_val_logloss > (best_valid_logloss + 1):
                best_valid_logloss = total_val_logloss
                epochs_no_improve = 0
            else:
                if total_val_logloss > best_valid_logloss:
                    best_valid_logloss = total_val_logloss
                epochs_no_improve = epochs_no_improve + 10
            
        if epochs_no_improve == early_stop_epochs:
            break
        
    # Evaluate
    dcs_surv_pred, _, _ = predict_survival_curve(model, X_test, time_bins)
    dcs_surv_pred = pd.DataFrame(dcs_surv_pred, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(dcs_surv_pred.T, y_test['time'], y_test['event'],
                                        y_train['time'], y_train['event'])
    ci = lifelines_eval.concordance()[0]
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ibs = lifelines_eval.integrated_brier_score()
    print(f"DCSurvial: CI={round(ci, 2)} - MAE={round(mae_hinge, 2)} - IBS={round(ibs, 2)}")
    
    # Make truth model
    truth_model = Weibull_nonlinear(n_features=X_test.shape[1],
                                    alpha=beta_shape_e,
                                    gamma=beta_scale_e,
                                    beta=beta,
                                    risk_function=risk_fn,
                                    device=torch.device("cpu"))
    truth_surv_pred, _, _ = predict_survival_curve(truth_model, X_test, time_bins, truth=True)
    truth_surv_pred = pd.DataFrame(truth_surv_pred, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(truth_surv_pred.T, y_test['time'], y_test['event'],
                                        y_train['time'], y_train['event'])
    ci = lifelines_eval.concordance()[0]
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ibs = lifelines_eval.integrated_brier_score()
    print(f"Truth: CI={round(ci, 2)} - MAE={round(mae_hinge, 2)} - IBS={round(ibs, 2)}")
    
    # Plot DCSurvival and truth survival
    import matplotlib.pyplot as plt
    plt.plot(time_bins, dcs_surv_pred.mean(axis=0), label="DCSurvival No Cop")
    plt.plot(time_bins, truth_surv_pred.mean(axis=0), label="Truth model Theta = 0.01")
    plt.legend()
    plt.show()
    
    # Calculate NLL of DCSurvival and truth model
    model.eval()
    performance = surv_diff(truth_model, model, covariate_tensor_test, steps=time_bins)
    print(performance)
    