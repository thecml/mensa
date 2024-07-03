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
from utility.survival import preprocess_data
from utility.data import dotdict, array_to_tensor
import torch.optim as optim
import torch.nn as nn
from copula import Clayton
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from SurvivalEVAL.Evaluator import LifelinesEvaluator
import copy
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear, Exp_linear, LogNormal_linear
from torch.utils.data import DataLoader, TensorDataset
import math
from utility.data import format_data, format_data_as_dict_multi
from utility.config import load_config
from utility.survival import risk_fn, compute_l1_difference, predict_survival_function
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function
from copula import NestedClayton, NestedFrank, ConvexCopula
from SurvivalEVAL.Evaluations.util import predict_median_survival_time
from SurvivalEVAL.Evaluations.MeanError import mean_error
from utility.loss import single_loss, double_loss, triple_loss
from mensa.model import train_mensa_model_3_events, train_mensa_model_2_events
from pycop import simulation
from data_loader import *
from copula import Clayton2D

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.25]
MODELS = ["weibull-nocop", "weibull-cop"]
N_SAMPLES = 10000
N_FEATURES = 10

if __name__ == "__main__":
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_se.yaml")
    dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                    linear=True, copula_name="clayton",
                                                    k_tau=KENDALL_TAUS[0], device='cpu', dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    n_events = data_config['n_events']
    dgps = dl.dgps
    
    print(f"Goal theta: {kendall_tau_to_theta('clayton', KENDALL_TAUS[0])}") # 0.8571
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=train_dict['E']) # Use first event for time bins
    
    # Make models
    eps = 1e-4
    n_features = 10
    copula_start_point = 2.0
    copula = Clayton2D(torch.tensor([copula_start_point], device=device, dtype=dtype), device, dtype)
    #copula = NestedFrank(torch.tensor([copula_start_point]),
    #                     torch.tensor([copula_start_point]), eps, eps, device, dtype)
    #c1 = NestedFrank(torch.tensor([2.0]), torch.tensor([2.0]), 1e-4, 1e-4, device, dtype)
    #copula = NestedClayton(torch.tensor([2.0]), torch.tensor([2.0]), 1e-4, 1e-4, device, dtype)
    #copula = ConvexCopula(c1, c2, beta=10000, device=device, dtype=dtype)
    
    model1 = LogNormal_linear(n_features, device)
    model2 = LogNormal_linear(n_features, device)
    #model3 = Weibull_log_linear(n_features, 2, 1, device, dtype)
    model1, model2, copula = train_mensa_model_2_events(train_dict, valid_dict, model1, model2,
                                                        copula, n_epochs=5000, lr=0.001, model_type='LogNormal_linear')

    # Print NLL of all events together
    print(f"NLL all events: {double_loss(model1, model2, valid_dict, copula)}")
    
    # Check the dgp performance
    #copula.theta = torch.tensor([5.0])
    print(f"DGP loss: {double_loss(dgps[0], dgps[1], valid_dict, copula)}")

    # Evaluate the L1
    preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy()
    preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
    #preds_c = predict_survival_function(model3, test_dict['X'], time_bins).detach().numpy()
    
    n_samples = test_dict['X'].shape[0]
    truth_preds_e1 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    for i in range(time_bins.shape[0]):
        truth_preds_e1[:,i] = dgps[0].survival(time_bins[i], test_dict['X'])
    l1_e1 = float(compute_l1_difference(truth_preds_e1, preds_e1, n_samples, steps=time_bins))
        
    truth_preds_e2 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    for i in range(time_bins.shape[0]):
        truth_preds_e2[:,i] = dgps[1].survival(time_bins[i], test_dict['X'])
    l1_e2 = float(compute_l1_difference(truth_preds_e2, preds_e2, n_samples, steps=time_bins))

    #truth_preds_c = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    #for i in range(time_bins.shape[0]):
    #    truth_preds_c[:,i] = dgps[2].survival(time_bins[i], test_dict['X'])
    #l1_c = float(compute_l1_difference(truth_preds_c, preds_c, n_samples, steps=time_bins))
    
    print(f"L1 E1: {l1_e1}")
    print(f"L1 E2: {l1_e2}")
    #print(f"L1 C: {l1_c}")
    
    for event_id, surv_preds in enumerate([preds_e1, preds_e2]):
        surv_preds = pd.DataFrame(surv_preds, columns=time_bins.numpy())
        y_test_time = test_dict['T']
        y_test_event = (test_dict['E'] == event_id)*1.0
        y_train_time = train_dict['T']
        y_train_event = (train_dict['E'] == event_id)*1.0
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        mae_hinge = lifelines_eval.mae(method="Hinge")
        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
        median_survs = lifelines_eval.predict_time_from_curve(predict_median_survival_time)
        event_indicators = np.array([1] * len(y_test_time))
        true_mae = float(mean_error(median_survs, y_test_time, event_indicators, method='Uncensored'))
        metrics = [ci, ibs, mae_hinge, mae_pseudo, true_mae]
        print(metrics)
                