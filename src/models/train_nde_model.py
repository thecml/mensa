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
from tqdm import tqdm
from utility.evaluation import LifelinesEvaluator
import copy
from dgp import Weibull_linear, Weibull_nonlinear
from torch.utils.data import DataLoader, TensorDataset
import math
from utility.data import format_data, format_data_as_dict_single
from NDE import survival_net_basic

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    dl = SingleEventSyntheticDataLoader().load_data(n_samples=10000,
                                               copula_name="clayton",
                                               k_tau=0, n_features=10)
    num_features, cat_features = dl.get_features()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                             valid_size=0.1,
                                                                             test_size=0.2)

    # Make time bins
    time_bins = make_time_bins(y_train['time'], event=y_train['event'], dtype=dtype)

    # Scale data
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features, num_features)

    # Make true models
    beta_e, shape_e, scale_e, beta_c, shape_c, scale_c, u_e = dl.params
    dgp1 = Weibull_linear(n_features=10, alpha=scale_e, gamma=shape_e, beta=beta_e, device=device, dtype=dtype)
    dgp2 = Weibull_linear(n_features=10, alpha=scale_c, gamma=shape_c, beta=beta_c, device=device, dtype=dtype)
    
    # Format data
    train_dict = format_data_as_dict_single(X_train, y_train, dtype)
    valid_dict = format_data_as_dict_single(X_valid, y_valid, dtype)
    
    # Train model
    num_epochs = 100
    sn1 = survival_net_basic(n_features=10, 
                            cat_size_list=[],
                            d_in_y=2,
                            d_out=1,
                            layers_x=[32,32,32],
                            layers_t=[],
                            layers=[32,32,32,32],
                            dropout=0.25, eps=1e-3)

    optimizer = torch.optim.Adam(sn1.parameters(), lr=1e-3)
    log_f = torch.log(1e-10+dgp1.PDF(valid_dict['T'], valid_dict['X']))
    log_s = torch.log(1e-10+dgp1.survival(valid_dict['T'], valid_dict['X']))
    print(-(log_f * valid_dict['E'] + log_s * (1-valid_dict['E'])).mean())
    for i in range(5000):
        optimizer.zero_grad()
        log_f = sn1.forward_f(train_dict['X'], train_dict['T'].reshape(-1,1))
        log_s = sn1.forward_S(train_dict['X'], train_dict['T'].reshape(-1,1), mask=0)
        loss = -(log_f * train_dict['E'].reshape(-1,1) + log_s * (1-train_dict['E'].reshape(-1,1))).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            with torch.no_grad():
                log_f = sn1.forward_f(valid_dict['X'], valid_dict['T'].reshape(-1,1))
                log_s = sn1.forward_S(valid_dict['X'], valid_dict['T'].reshape(-1,1), mask=0)
                loss_val = -(log_f * valid_dict['E'].reshape(-1,1) + log_s * (1-valid_dict['E'].reshape(-1,1))).mean()
                print(loss, loss_val)