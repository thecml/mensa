import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
import torch
import random
import warnings
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
from SurvivalEVAL.Evaluator import LifelinesEvaluator
import copy
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
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
from utility.loss import single_loss, triple_loss
from mensa.model import train_mensa_model_3_events

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.25]
MODELS = ["weibull-nocop", "weibull-cop"]
N_SAMPLES = 10000
N_FEATURES = 10

def generate_events(dgp1, dgp2, dgp3, x, device,theta=2.0, family='clayton'):
    if family is None:
        uv = torch.randn((x.shape[0], 3))#sample idnependent 
    else:
        u,v,w = simulation.simu_archimedean(family, 3, x.shape[0], theta=theta)
        u = torch.from_numpy(u).type(torch.float32).reshape(-1,1)
        v = torch.from_numpy(v).type(torch.float32).reshape(-1,1)
        w = torch.from_numpy(w).type(torch.float32).reshape(-1,1)
        uv = torch.cat([u,v,w], axis=1)
    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    t3 = dgp3.rvs(x, uv[:,2])
    T = np.concatenate([t1.reshape(-1,1),t2.reshape(-1,1),t3.reshape(-1,1)], axis=1)
    E = np.argmin(T,axis=1)
    obs_T = T[np.arange(T.shape[0]), E]
    T = torch.from_numpy(T).type(torch.float32)
    E = torch.from_numpy(E).type(torch.float32)
    obs_T = torch.from_numpy(obs_T).type(torch.float32)

    return {'X':x,'E':E, 'T':obs_T, 'T1':t1, 'T2':t2, 'T3':t3}

def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {"x_train":x_train, "x_val":x_val, "x_test":x_test}

def generate_data(x_dict, dgp1, dgp2,dgp3,device, copula='clayton', theta=2.0):
    train_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_train'],device, theta, copula)
    val_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_val'],device, theta, copula)
    test_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_test'],device, theta, copula)
    return train_dict, val_dict, test_dict

if __name__ == "__main__":
    model_results = pd.DataFrame()
    
    for dataset_version in DATASET_VERSIONS:
        for copula_name in COPULA_NAMES:
            for k_tau in KENDALL_TAUS:
                # Load and split data
                #dl = CompetingRiskSyntheticDataLoader().load_data(n_samples=N_SAMPLES,
                #                                                  linear=True, n_features=N_FEATURES)
                #num_features, cat_features = dl.get_features()
                #train_data, valid_data, test_data = dl.split_data(train_size=0.7, valid_size=0.3)
                nf = 10
                n_train = 10000
                n_val = 5000
                n_test = 5000
                x_dict = synthetic_x(n_train, n_val, n_test, nf, device)
                beta = torch.rand((nf,), device=device).type(dtype)
                dgp1 = Weibull_linear(10, alpha=17, gamma=8, beta=beta, device=device, dtype=dtype)
                dgp2 = Weibull_linear(10, alpha=16, gamma=8, beta=beta, device=device, dtype=dtype)
                dgp3 = Weibull_linear(10, alpha=17, gamma=9, beta=beta, device=device, dtype=dtype)
                dgp1.coeff = torch.rand((10,), device=device)
                dgp2.coeff = torch.rand((10,), device=device)
                dgp3.coeff = torch.rand((10,), device=device)
                train_dict, valid_dict, test_dict = generate_data(x_dict, dgp1, dgp2, dgp3, device, 'clayton', 5.0)
                
                # Make time bins
                time_bins = make_time_bins(train_dict['T'], event=train_dict['E']) # Use first event for time bins
                
                # Format data
                #train_dict = format_data_as_dict_multi(train_data[0], train_data[1], train_data[2], dtype)
                #valid_dict = format_data_as_dict_multi(valid_data[0], valid_data[1], valid_data[2], dtype)
                #test_dict = format_data_as_dict_multi(test_data[0], test_data[1], test_data[2], dtype)
                
                # Make truth models
                """
                n_features = nf
                beta, alpha_e1, gamma_e1, alpha_e2, gamma_e2, alpha_c, gamma_c = dl.params
                truth_model_e1 = Weibull_linear(n_features, alpha=alpha_e1, gamma=gamma_e1,
                                                beta=beta.squeeze(1), device=device, dtype=dtype)
                truth_model_e2 = Weibull_linear(n_features, alpha=alpha_e2, gamma=gamma_e2,
                                                beta=beta.squeeze(1), device=device, dtype=dtype)
                truth_model_c = Weibull_linear(n_features, alpha=alpha_c, gamma=gamma_c,
                                               beta=beta.squeeze(1), device=device, dtype=dtype)
                """
                
                # Make models
                eps = 1e-4
                n_features = nf
                copula_start_point = 2.0
                #copula = Clayton.Clayton3(torch.tensor([2.0], dtype=dtype), eps, dtype, device)
                #copula = NestedFrank(torch.tensor([copula_start_point]),
                #                     torch.tensor([copula_start_point]), eps, eps, device, dtype)
                #c1 = NestedFrank(torch.tensor([2.0]), torch.tensor([2.0]), 1e-4, 1e-4, device, dtype)
                copula = NestedClayton(torch.tensor([2.0]), torch.tensor([2.0]), 1e-4, 1e-4, device, dtype)
                #copula = ConvexCopula(c1, c2, beta=10000, device=device, dtype=dtype)
                
                model1 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                model2 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                model3 = Weibull_log_linear(n_features, 2, 1, device, dtype)
                model1, model2, model3 = train_mensa_model_3_events(train_dict, model1, model2, model3,
                                                                    copula, n_epochs=5000, lr=0.005)
            
                # Print NLL of all events together
                print(f"NLL all events: {triple_loss(model1, model2, model3, valid_dict, copula)}")
                
                # Check the dgp performance
                #copula.theta = torch.tensor([5.0])
                print(f"DGP loss: {triple_loss(dgp1, dgp2, dgp3, valid_dict, copula)}")

                # Evaluate the L1
                preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy()
                preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
                preds_c = predict_survival_function(model3, test_dict['X'], time_bins).detach().numpy()
                
                n_samples = test_dict['X'].shape[0]
                truth_preds_e1 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                for i in range(time_bins.shape[0]):
                    truth_preds_e1[:,i] = dgp1.survival(time_bins[i], test_dict['X'])
                l1_e1 = float(compute_l1_difference(truth_preds_e1, preds_e1, n_samples, steps=time_bins))
                    
                truth_preds_e2 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                for i in range(time_bins.shape[0]):
                    truth_preds_e2[:,i] = dgp2.survival(time_bins[i], test_dict['X'])
                l1_e2 = float(compute_l1_difference(truth_preds_e2, preds_e2, n_samples, steps=time_bins))

                truth_preds_c = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                for i in range(time_bins.shape[0]):
                    truth_preds_c[:,i] = dgp3.survival(time_bins[i], test_dict['X'])
                l1_c = float(compute_l1_difference(truth_preds_c, preds_c, n_samples, steps=time_bins))
                
                print(f"L1 E1: {l1_e1}")
                print(f"L1 E2: {l1_e2}")
                print(f"L1 C: {l1_c}")
                
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
                         
                break