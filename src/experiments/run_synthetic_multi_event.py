"""
run_synthetic_me_three_events.py
====================================
Experiment 3.1

Models: ["deepsurv", 'hierarch', 'mensa', 'dgp']
"""

# 3rd party
import pandas as pd
import numpy as np
import config as cfg
import torch
import torch.optim as optim
import torch.nn as nn
import random
import warnings
import copy
import tqdm
import math
import argparse
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import MultiEventSyntheticDataLoader
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import dotdict
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from utility.data import format_data_deephit_cr, format_hierarch_data_multi_event, calculate_layer_size_hierarch
from utility.evaluation import global_C_index, local_C_index

# SOTA
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                          make_deephit_cr, make_dsm_model, make_rsf_model, train_deepsurv_model,
                          make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ['deepsurv']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_tau', type=float, default=0.25)
    parser.add_argument('--copula_name', type=str, default="clayton")
    parser.add_argument('--linear', type=bool, default=True)
    
    args = parser.parse_args()
    seed = args.seed
    k_tau = args.k_tau
    copula_name = args.copula_name
    linear = args.linear
    
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic.yaml")
    dl = MultiEventSyntheticDataLoader().load_data(data_config, k_taus=[k_tau, k_tau, k_tau],
                                                   linear=linear, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    n_events = data_config['me_n_events']
    dgps = dl.dgps

    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Evaluate models
    model_results = pd.DataFrame()
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            model1 = DeepSurv(in_features=n_features, config=config)
            model2 = DeepSurv(in_features=n_features, config=config)
            model3 = DeepSurv(in_features=n_features, config=config)
            data_train1 = pd.DataFrame(train_dict['X'])
            data_train1['time'] = train_dict['T'][:,0]
            data_train1['event'] = train_dict['E'][:,0]
            data_train2 = pd.DataFrame(train_dict['X'])
            data_train2['time'] = train_dict['T'][:,1]
            data_train2['event'] = train_dict['E'][:,1]
            data_train3 = pd.DataFrame(train_dict['X'])
            data_train3['time'] = train_dict['T'][:,2]
            data_train3['event'] = train_dict['E'][:,2]
            model1 = train_deepsurv_model(model1, data_train1, time_bins, config=config, random_state=0,
                                          reset_model=True, device=device, dtype=dtype)
            model2 = train_deepsurv_model(model2, data_train2, time_bins, config=config, random_state=0,
                                          reset_model=True, device=device, dtype=dtype)
            model3 = train_deepsurv_model(model3, data_train3, time_bins, config=config, random_state=0,
                                          reset_model=True, device=device, dtype=dtype)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_me.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarch_data_multi_event(train_dict, valid_dict,
                                                                                 test_dict, n_time_bins)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = n_time_bins
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            model1, model2, model3, copula = make_mensa_model_3_events(n_features, start_theta=2.0, eps=1e-4,
                                                                       device=device, dtype=dtype)
            model1, model2, model3, copula = train_mensa_model_3_events(train_dict, valid_dict, model1, model2, model3,
                                                                        copula, n_epochs=5000, lr=0.001)
            print(f"NLL all events: {triple_loss(model1, model2, model3, valid_dict, copula)}")
            print(f"DGP loss: {triple_loss(dgps[0], dgps[1], dgps[2], valid_dict, copula)}")
        elif model_name == "dgp":
            model1 = dgps[0]
            model2 = dgps[1]
            model3 = dgps[2]
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]                    
        if model_name == "deepsurv":
            preds_e1, time_bins_model1, _ = make_deepsurv_prediction(model1, test_dict['X'],
                                                                        config=config, dtype=dtype)
            preds_e2, time_bins_model2, _ = make_deepsurv_prediction(model2, test_dict['X'],
                                                                        config=config, dtype=dtype)
            preds_e3, time_bins_model3, _ = make_deepsurv_prediction(model3, test_dict['X'],
                                                                        config=config, dtype=dtype)
            spline1 = interp1d(time_bins_model1, preds_e1, kind='linear', fill_value='extrapolate')
            spline2 = interp1d(time_bins_model2, preds_e2, kind='linear', fill_value='extrapolate')
            spline3 = interp1d(time_bins_model3, preds_e3, kind='linear', fill_value='extrapolate')
            all_preds = [spline1(time_bins), spline2(time_bins), spline3(time_bins)]
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(test_data[0], model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            preds_e1 = pd.DataFrame(event_preds[0], columns=bin_locations)
            preds_e2 = pd.DataFrame(event_preds[1], columns=bin_locations)
            preds_e3 = pd.DataFrame(event_preds[2], columns=bin_locations)
            spline1 = interp1d(bin_locations, preds_e1, kind='linear', fill_value='extrapolate')
            spline2 = interp1d(bin_locations, preds_e2, kind='linear', fill_value='extrapolate')
            spline3 = interp1d(bin_locations, preds_e3, kind='linear', fill_value='extrapolate')
            all_preds = [spline1(time_bins), spline2(time_bins), spline3(time_bins)]
        elif model_name == "mensa":
            preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy()
            preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
            preds_e3 = predict_survival_function(model3, test_dict['X'], time_bins).detach().numpy()
            all_preds = [preds_e1, preds_e2, preds_e3]
        elif model_name == "dgp":
            preds_e1 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                preds_e1[:,i] = model1.survival(time_bins[i], test_dict['X'])
            preds_e2 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                preds_e2[:,i] = model2.survival(time_bins[i], test_dict['X'])
            preds_e3 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                preds_e3[:,i] = model3.survival(time_bins[i], test_dict['X'])
            all_preds = [preds_e1, preds_e2, preds_e3]
        else:
            raise NotImplementedError()
        
        # Test local and global CI
        global_ci = global_C_index(all_preds, test_dict['T'].numpy(), test_dict['E'].numpy())
        local_ci = local_C_index(all_preds, test_dict['T'].numpy(), test_dict['E'].numpy())
    
        # Make evaluation for each event
        pred_time_bins = torch.cat([torch.tensor([0], device=device, dtype=dtype), time_bins])
        for event_id, surv_preds in enumerate(all_preds):
            surv_preds = np.hstack((np.ones((surv_preds.shape[0], 1)), surv_preds))
            surv_preds_df = pd.DataFrame(surv_preds, columns=pred_time_bins.numpy())
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict['T'][:,event_id]
            y_train_event = np.array([1] * n_train_samples)
            y_test_time = test_dict['T'][:,event_id]
            y_test_event = np.array([1] * n_test_samples)
            lifelines_eval = LifelinesEvaluator(surv_preds_df.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            ci =  lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score(num_points=len(pred_time_bins))
            mae = lifelines_eval.mae(method='Uncensored')
            d_calib = lifelines_eval.d_calibration()[0]
            
            truth_preds = torch.zeros((n_samples, pred_time_bins.shape[0]), device=device)
            for i in range(pred_time_bins.shape[0]):
                truth_preds[:,i] = dgps[event_id].survival(pred_time_bins[i], test_dict['X'])
            survival_l1 = float(compute_l1_difference(truth_preds, surv_preds, n_samples, steps=pred_time_bins))
            
            metrics = [ci, ibs, mae, survival_l1, d_calib, global_ci, local_ci]
            print(metrics)
            res_sr = pd.Series([model_name, linear, copula_name, k_tau] + metrics,
                                index=["ModelName", "Linear", "Copula", "KTau",
                                        "CI", "IBS", "MAE", "L1", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
            model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")
            