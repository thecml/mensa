"""
run_synthetic_competing_risks.py
====================================
Experiment 2.1

Models: ['deepsurv', 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa', 'mensa-cop', 'dgp']
"""
import sys, os
sys.path.append(os.path.abspath('../'))
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
from data_loader import CompetingRiskSyntheticDataLoader
from utility.survival import (make_time_bins, compute_l1_difference)
from utility.data import dotdict
from utility.config import load_config
from utility.data import format_data_deephit_cr, format_hierarchical_data_cr, calculate_layer_size_hierarch
from utility.evaluation import global_C_index, local_C_index
from mensa.model import MENSA
from copula import Nested_Convex_Copula

# SOTA
from sota_models import (make_deephit_cr, make_dsm_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)
from utility.mtlr import train_mtlr_cr
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from torchmtlr.utils import encode_mtlr_format
from torchmtlr.model import MTLRCR, mtlr_survival

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
MODELS = ['mensa-nocop', 'dgp']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_tau', type=float, default=0.5)
    parser.add_argument('--copula_name', type=str, default="clayton")
    parser.add_argument('--linear', type=bool, default=False)
    
    args = parser.parse_args()
    seed = args.seed
    k_tau = args.k_tau
    copula_name = args.copula_name
    linear = args.linear
    
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic_cr.yaml")
    dl = CompetingRiskSyntheticDataLoader().load_data(data_config, k_tau=k_tau, copula_name=copula_name,
                                                      linear=linear, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    n_features = train_dict['X'].shape[1]
    n_events = dl.n_events
    dgps = dl.dgps
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Evaluate models
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = DeepSurv(in_features=n_features, config=config)
                data_train = pd.DataFrame(train_dict['X'])
                data_train['time'] = train_dict['T']
                data_train['event'] = (train_dict['E'] == i+1)*1.0
                data_valid = pd.DataFrame(valid_dict['X'])
                data_valid['time'] = valid_dict['T']
                data_valid['event'] = (valid_dict['E'] == i+1)*1.0
                model = train_deepsurv_model(model, data_train, data_valid, time_bins,
                                             config=config, random_state=0,
                                             reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "deephit":
            config = dotdict(cfg.DEEPHIT_PARAMS)
            max_time = torch.tensor([dl.get_data()[1].max()], device=device, dtype=dtype)
            time_bins_dh = time_bins
            if max_time not in time_bins_dh:
                time_bins_dh = torch.concat([time_bins_dh, max_time], dim=0)
            model = make_deephit_cr(in_features=n_features, out_features=len(time_bins_dh),
                                    num_risks=n_events, duration_index=time_bins_dh, config=config)
            train_data, valid_data, out_features, duration_index = format_data_deephit_cr(train_dict, valid_dict, time_bins_dh)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"synthetic_cr.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                                                            n_time_bins, n_events, censoring_event=False)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = len(time_bins)
            params = cfg.HIERARCH_PARAMS
            n_samples = train_dict['X'].shape[0]            
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mtlrcr":
            train_times = np.digitize(train_dict['T'], bins=time_bins.cpu().numpy()).astype(np.int64)
            train_events = train_dict['E'].type(torch.int64)
            valid_times = np.digitize(valid_dict['T'], bins=time_bins.cpu().numpy()).astype(np.int64)
            valid_events = valid_dict['E'].type(torch.int64)
            y_train = encode_mtlr_format(train_times, train_events, time_bins.cpu().numpy())
            y_valid = encode_mtlr_format(valid_times, valid_events, time_bins.cpu().numpy())
            num_time_bins = len(time_bins) + 1
            config = dotdict(cfg.MTLRCR_PARAMS)
            model = MTLRCR(in_features=n_features, num_time_bins=num_time_bins, num_events=n_events)
            model = train_mtlr_cr(train_dict['X'], y_train, valid_dict['X'], y_valid,
                                  model, time_bins, num_epochs=config['num_epochs'],
                                  lr=config['lr'], batch_size=config['batch_size'],
                                  verbose=True, device=device, C1=config['c1'],
                                  early_stop=config['early_stop'], patience=config['patience'])
        elif model_name == "dsm":
            config = dotdict(cfg.DSM_PARAMS)
            n_iter = config['n_iter']
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
            model = make_dsm_model(config)
            model.fit(train_dict['X'].numpy(), train_dict['T'].numpy(), train_dict['E'].numpy(),
                      val_data=(valid_dict['X'].numpy(), valid_dict['T'].numpy(), valid_dict['T'].numpy()),
                      learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            dropout = config['dropout']
            copula = Nested_Convex_Copula(['cl'], ['cl'], [1, 1], [1], 1e-3,
                                          dtype=dtype, device=device)
            model = MENSA(n_features=n_features, n_events=n_events+1, hidden_layers=layers,
                          dropout=dropout, copula=copula, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs,
                      lr_dict={'network': lr, 'copula': 0.01}, verbose=True)
        elif model_name == "mensa-nocop":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
            n_epochs = config['n_epochs']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            dropout = config['dropout']
            model = MENSA(n_features=n_features, n_events=n_events+1, hidden_layers=layers,
                          dropout=dropout, copula=None, device=device)
            model.fit(train_dict, valid_dict, n_epochs=n_epochs, lr_dict={'network': lr})
        elif model_name == "dgp":
            pass
        else:
            raise NotImplementedError()
        
        # Compute survival function
        if model_name == "deepsurv":
            all_preds = []
            for trained_model in trained_models:
                preds, time_bins_model = make_deepsurv_prediction(trained_model, test_dict['X'].to(device),
                                                                  config=config, dtype=dtype)
                spline = interp1d(time_bins_model.cpu().numpy(), preds.cpu().numpy(),
                                  kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "deephit":
            cif = model.predict_cif(test_dict['X']).cpu().numpy() # TODO: Check output
            all_preds = []
            for i in range(n_events):
                cif_df = pd.DataFrame((1-cif[i]).T, columns=time_bins_dh.cpu().numpy())
                spline = interp1d(time_bins_dh.cpu().numpy(), cif_df, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(torch.tensor(test_data[0]), model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            all_preds = []
            for event_pred in event_preds:
                preds = pd.DataFrame(event_pred, columns=bin_locations)
                spline = interp1d(bin_locations, preds, kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()), columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "mtlrcr":
            pred_prob = model(test_dict['X'].to(device))
            num_points = len(time_bins)
            all_preds = []
            for i in range(n_events):
                start = i * num_time_bins
                end = start + num_time_bins
                preds = mtlr_survival(pred_prob[:, start:end]).detach().cpu().numpy()[:, 1:]
                preds = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "dsm":
            all_preds = []
            for i in range(n_events):
                model_pred = model.predict_survival(test_dict['X'].numpy(), t=list(time_bins.cpu().numpy()), risk=i+1)
                model_pred = pd.DataFrame(model_pred, columns=time_bins.cpu().numpy())
                all_preds.append(model_pred)
        elif model_name in ["mensa", "mensa-nocop"]:
            model_preds = model.predict(test_dict['X'], time_bins)
            all_preds = []
            for model_pred in model_preds:
                model_pred = pd.DataFrame(model_pred.detach().cpu().numpy(), columns=time_bins.cpu().numpy())
                all_preds.append(model_pred)
            all_preds.pop(0) # remove censoring model
        elif model_name == "dgp":
            all_preds = []
            n_samples = test_dict['X'].shape[0]
            for model in dgps:
                preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                for i in range(time_bins.shape[0]):
                    preds[:,i] = model.survival(time_bins[i], test_dict['X'].to(device))
                preds = preds.cpu().numpy()
                preds_df = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
                all_preds.append(preds_df)
            all_preds.pop(0) # remove censoring model
        else:
            raise NotImplementedError()
        
        # Calculate local and global CI
        y_test_time = np.stack([test_dict['T'] for _ in range(n_events)], axis=1)
        y_test_event = np.stack([np.array((test_dict['E'] == i+1)*1.0) for i in range(n_events)], axis=1)
        all_preds_arr = [df.to_numpy() for df in all_preds]
        global_ci = global_C_index(all_preds_arr, y_test_time, y_test_event)
        local_ci = local_C_index(all_preds_arr, y_test_time, y_test_event)
        
        # Make evaluation for each event
        model_results = pd.DataFrame()
        for event_id, surv_preds in enumerate(all_preds):
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict[f'T{event_id+1}']
            y_train_event = np.array([1] * n_train_samples)
            y_test_time = test_dict[f'T{event_id+1}']
            y_test_event = np.array([1] * n_test_samples)
            lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            ci =  lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score(num_points=len(time_bins))
            mae = lifelines_eval.mae(method='Uncensored')
            d_calib = lifelines_eval.d_calibration()[0]
            
            n_samples = test_dict['X'].shape[0]
            truth_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
            for i in range(time_bins.shape[0]):
                truth_preds[:,i] = dgps[event_id+1].survival(time_bins[i], test_dict['X'].to(device))
            model_preds_th = torch.tensor(surv_preds.values, device=device, dtype=dtype)
            survival_l1 = float(compute_l1_difference(truth_preds, model_preds_th,
                                                      n_samples, steps=time_bins))
            
            metrics = [ci, ibs, mae, survival_l1, d_calib, global_ci, local_ci]
            print(f'{model_name} {event_id+1}: ' + f'{metrics}')
            res_sr = pd.Series([model_name, seed, linear, copula_name, k_tau, event_id+1] + metrics,
                                index=["ModelName", "Seed", "Linear", "Copula", "KTau", "EventId",
                                       "CI", "IBS", "MAE", "L1", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/synthetic_cr.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)
                    