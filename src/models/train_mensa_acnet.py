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
from copula import Clayton
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from tqdm import tqdm
from utility.evaluator import LifelinesEvaluator
import copy
from dcsurvival.truth_net import Weibull_linear

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
torch.set_default_dtype(torch.float64)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

def predict_survival_curve(model, x_test, time_bins, truth=False):
    device = torch.device("cpu")
    if truth == False:
        model = copy.deepcopy(model).to(device)
    surv_estimate = torch.zeros((x_test.shape[0], time_bins.shape[0]), device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    time_bins = torch.tensor(time_bins)
    for i in range(time_bins.shape[0]):
        surv_estimate[:,i] = model.survival(time_bins[i], x_test)
    return surv_estimate, time_bins, time_bins.max()

if __name__ == "__main__":
    # Load data
    dl = LinearSyntheticDataLoader().load_data(n_samples=1000)
    num_features, cat_features = dl.get_features()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                             valid_size=0.5)
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
    covariate_tensor_train = torch.tensor(X_train).to(device)
    times_tensor_val = array_to_tensor(y_valid['time'], torch.float32)
    event_indicator_tensor_val = array_to_tensor(y_valid['event'], torch.float32)
    covariate_tensor_val = torch.tensor(X_valid).to(device)
    
    # Define ACNet, model
    depth = 2 # depth of ACNet
    widths = [100, 100] # number of units of ACNet
    lc_w_range = (0, 1.0) # Phi_B
    shift_w_range = (0., 2.0) # Phi_B
    num_epochs = 100 # 5000
    batch_size = 128
    early_stop_epochs = 10
    phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol=1e-14).to(device)
    model = DCSurvival(phi, device = device, num_features=X_train.shape[1], tol=1e-14).to(device)
    optimizer = optim.Adam([{"params": model.sumo_e.parameters(), "lr": 1e-3},
                            {"params": model.sumo_c.parameters(), "lr": 1e-3},
                            {"params": model.phi.parameters(), "lr": 1e-4}])
    
    # Train model
    best_valid_logloss = float('-inf')
    epochs_no_improve = 0
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        logloss = model(covariate_tensor_train, times_tensor_train,
                        event_indicator_tensor_train, max_iter=10000)
        (-logloss).backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            val_logloss = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=1000)
            if val_logloss > (best_valid_logloss + 1):
                best_valid_logloss = val_logloss
                epochs_no_improve = 0
            else:
                if val_logloss > best_valid_logloss:
                    best_valid_logloss = val_logloss
                epochs_no_improve = epochs_no_improve + 10
        
        if epochs_no_improve == early_stop_epochs:
            break
    
    # Evaluate
    surv_pred, _, _ = predict_survival_curve(model, X_test, time_bins)
    surv_pred = pd.DataFrame(surv_pred, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test['time'], y_test['event'],
                                        y_train['time'], y_train['event'])
    ci = lifelines_eval.concordance()[0]
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ibs = lifelines_eval.integrated_brier_score()
    print(f"DCSurvial: CI={round(ci, 2)} - MAE={round(mae_hinge, 2)} - IBS={round(ibs, 2)}")
    
    # Train and evaluate truth model
    truth_model = Weibull_linear(num_feature=X_test.shape[1], shape=4,
                                 scale=14, device=torch.device("cpu"),
                                 coeff=beta_e)
    surv_pred, _, _ = predict_survival_curve(truth_model, X_test, time_bins, truth=True)
    surv_pred = pd.DataFrame(surv_pred, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test['time'], y_test['event'],
                                        y_train['time'], y_train['event'])
    ci = lifelines_eval.concordance()[0]
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ibs = lifelines_eval.integrated_brier_score()
    print(f"Truth: CI={round(ci, 2)} - MAE={round(mae_hinge, 2)} - IBS={round(ibs, 2)}")
    