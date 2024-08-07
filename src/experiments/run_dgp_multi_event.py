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
import random
import warnings
import argparse
import os
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import MultiEventSyntheticDataLoader
from utility.survival import (make_time_bins, compute_l1_difference)
from utility.data import dotdict
from utility.config import load_config
from utility.data import format_hierarchical_data_me, calculate_layer_size_hierarch
from utility.evaluation import global_C_index, local_C_index
from mensa.model import MENSA

# SOTA
from sota_models import (train_deepsurv_model, make_deepsurv_prediction, DeepSurv)
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
MODELS = ["deepsurv", 'hierarch', 'mensa', 'dgp']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_tau', type=float, default=0.25)
    parser.add_argument('--copula_name', type=str, default="clayton")
    parser.add_argument('--linear', type=bool, default=False)
    
    args = parser.parse_args()
    seed = args.seed
    k_tau = args.k_tau
    copula_name = args.copula_name
    linear = args.linear
    
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_me.yaml")
    dl = MultiEventSyntheticDataLoader().load_data(data_config, k_taus=[k_tau, k_tau, k_tau],
                                                   linear=linear, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    n_events = data_config['n_events']
    dgps = dl.dgps

    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Evaluate models
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = DeepSurv(in_features=n_features, config=config)
                data_train = pd.DataFrame(train_dict['X'])
                data_train['time'] = train_dict['T'][:,i]
                data_train['event'] = train_dict['E'][:,i]
                data_valid = pd.DataFrame(valid_dict['X'])
                data_valid['time'] = valid_dict['T'][:,i]
                data_valid['event'] = valid_dict['E'][:,i]
                model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                             random_state=0, reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_me.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_me(train_dict, valid_dict, test_dict, n_time_bins)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = n_time_bins
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            batch_size = config['batch_size']
            copula = Convex_Nested(2, 2, 1e-3, 1e-3, device)
            model = MENSA(n_features, n_events, copula=copula, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr=lr, batch_size=1024)
        elif model_name == "dgp":
            pass
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]
        if model_name == "deepsurv":
            all_preds = []
            for trained_model in trained_models:
                preds, time_bins_model, _ = make_deepsurv_prediction(trained_model, test_dict['X'],
                                                                     config=config, dtype=dtype)
                spline = interp1d(time_bins_model, preds, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins), columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(test_data[0], model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            all_preds = []
            for event_pred in event_preds:
                preds = pd.DataFrame(event_pred, columns=bin_locations)
                spline = interp1d(bin_locations, preds, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins), columns=time_bins.numpy())
                all_preds.append(preds)
        elif model_name == "mensa":
            model_preds = model.predict(test_dict['X'], time_bins.numpy())
            all_preds = []
            for model_pred in model_preds:
                model_pred = pd.DataFrame(model_pred.detach().numpy(), columns=time_bins.numpy())
                all_preds.append(model_pred)
        elif model_name == "dgp":
            all_preds = []
            for model in dgps:
                preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                for i in range(time_bins.shape[0]):
                    preds[:,i] = model.survival(time_bins[i], test_dict['X'])
                preds_df = pd.DataFrame(preds, columns=time_bins.numpy())
                all_preds.append(preds_df)
        else:
            raise NotImplementedError()
        
        # Test local and global CI
        all_preds_arr = [df.to_numpy() for df in all_preds]
        global_ci = global_C_index(all_preds_arr, test_dict['T'].numpy(), test_dict['E'].numpy())
        local_ci = local_C_index(all_preds_arr, test_dict['T'].numpy(), test_dict['E'].numpy())
        
        # Make evaluation for each event
        model_results = pd.DataFrame()
        for event_id, surv_preds in enumerate(all_preds):
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict['T'][:,event_id]
            y_train_event = np.array([1] * n_train_samples)
            y_test_time = test_dict['T'][:,event_id]
            y_test_event = np.array([1] * n_test_samples)
            lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            ci =  lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score(num_points=len(time_bins))
            mae = lifelines_eval.mae(method='Uncensored')
            d_calib = lifelines_eval.d_calibration()[0]
            
            truth_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                truth_preds[:,i] = dgps[event_id].survival(time_bins[i], test_dict['X'])
            survival_l1 = float(compute_l1_difference(truth_preds, surv_preds.to_numpy(),
                                                      n_samples, steps=time_bins))
            
            metrics = [ci, ibs, mae, survival_l1, d_calib, global_ci, local_ci]
            print(metrics)
            res_sr = pd.Series([model_name, seed, linear, copula_name, k_tau, event_id+1] + metrics,
                                index=["ModelName", "Seed", "Linear", "Copula", "KTau", "EventId",
                                        "CI", "IBS", "MAE", "L1", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)

        # Save results
        filename = f"{cfg.RESULTS_DIR}/synthetic_me.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)
                