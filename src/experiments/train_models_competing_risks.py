"""
run_competing_risks.py
====================================
Models: ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa']
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
from utility.survival import (convert_to_structured, make_time_bins, preprocess_data)
from utility.data import dotdict
from utility.config import load_config
from utility.data import (format_data_deephit_competing, format_hierarchical_data_cr, calculate_layer_size_hierarch)
from utility.evaluation import global_C_index, local_C_index
from data_loader import get_data_loader
from mensa.model import MENSA

# SOTA
from sota_models import (make_coxph_model, make_coxboost_model, make_deephit_cr, make_dsm_model, make_rsf_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)
from utility.mtlr import train_mtlr_cr
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from torchmtlr.utils import encode_mtlr_format
from torchmtlr.model import MTLRCR, mtlr_survival


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ["coxph", "coxboost", "rsf", "deepsurv", "deephit", "hierarch", "mtlrcr", "dsm", "mensa"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='seer_cr')
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    
    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    n_events = dl.n_events
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    n_features = train_dict['X'].shape[1]
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    n_samples = train_dict['X'].shape[0]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    # Evaluate models
    for model_name in MODELS:
        # Reset seeds
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        
        if model_name == "coxph":
            config = load_config(cfg.COXPH_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            trained_models = []
            for i in range(n_events):
                train_times = train_dict['T'].cpu().numpy()
                train_events = (train_dict['E'].cpu().numpy() == i+1)*1.0
                y_train = convert_to_structured(train_times, train_events)
                model = make_coxph_model(config)
                model.fit(train_dict['X'].cpu(), y_train)
                trained_models.append(model)
        elif model_name == "coxboost":
            config = load_config(cfg.COXBOOST_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            trained_models = []
            for i in range(n_events):
                train_times = train_dict['T'].cpu().numpy()
                train_events = (train_dict['E'].cpu().numpy() == i+1)*1.0
                y_train = convert_to_structured(train_times, train_events)
                model = make_coxboost_model(config)
                model.fit(train_dict['X'].cpu(), y_train)
                trained_models.append(model)
        elif model_name == "rsf":
            config = load_config(cfg.RSF_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            trained_models = []
            for i in range(n_events):
                train_times = train_dict['T'].cpu().numpy()
                train_events = (train_dict['E'].cpu().numpy() == i+1)*1.0
                y_train = convert_to_structured(train_times, train_events)
                model = make_rsf_model(config)
                model.fit(train_dict['X'].cpu(), y_train)
                trained_models.append(model)
        elif model_name == "deepsurv":
            config = load_config(cfg.DEEPSURV_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            trained_models = []
            for i in range(n_events):
                data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
                data_train['time'] = train_dict['T'].cpu().numpy()
                data_train['event'] = (train_dict['E'].cpu().numpy() == i+1)*1.0
                data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
                data_valid['time'] = valid_dict['T'].cpu().numpy()
                data_valid['event'] = (valid_dict['E'].cpu().numpy() == i+1)*1.0
                model = DeepSurv(in_features=n_features, config=config)
                model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                             random_state=0, reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "deephit":
            config = load_config(cfg.DEEPHIT_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            max_time = torch.tensor([dl.get_data()[1].max()], dtype=dtype, device=device)
            time_bins_dh = time_bins
            if max_time not in time_bins_dh:
                time_bins_dh = torch.concat([time_bins_dh, max_time], dim=0)
            model = make_deephit_cr(in_features=n_features, out_features=len(time_bins_dh),
                                    num_risks=n_events, duration_index=time_bins_dh, config=config)
            train_data, valid_data, out_features, duration_index = format_data_deephit_competing(train_dict, valid_dict, time_bins_dh)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name}.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                                                            n_time_bins, n_events, censoring_event=False)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = len(time_bins)
            config['n_batches'] = int(n_samples/config['batch_size'])
            layer_size = config['layer_size_fine_bins'][0][0]
            config['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(config)
            verbose = config['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mtlrcr":
            train_events = train_dict['E'].type(torch.int64).cpu().numpy()
            valid_events = valid_dict['E'].type(torch.int64).cpu().numpy()
            y_train = encode_mtlr_format(train_dict['T'], train_events, time_bins.cpu().numpy())
            y_valid = encode_mtlr_format(valid_dict['T'], valid_events, time_bins.cpu().numpy())            
            num_time_bins = len(time_bins.cpu().numpy()) + 1
            config = load_config(cfg.MTLR_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = MTLRCR(in_features=n_features, num_time_bins=num_time_bins, num_events=n_events)
            model = train_mtlr_cr(train_dict['X'], y_train, valid_dict['X'], y_valid,
                                  model, time_bins, num_epochs=config['num_epochs'],
                                  lr=config['lr'], batch_size=config['batch_size'],
                                  verbose=True, device=device, C1=config['c1'],
                                  early_stop=config['early_stop'], patience=config['patience'])
        elif model_name == "dsm":
            config = load_config(cfg.DSM_CONFIGS_DIR, f"{dataset_name}.yaml")
            n_iter = config['n_iter']
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
            model = make_dsm_model(config)
            model.fit(train_dict['X'].cpu().numpy(), train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy(),
                      val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'].cpu().numpy(), valid_dict['T'].cpu().numpy()),
                      learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
        elif model_name == "mensa":
            config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            n_epochs = config['n_epochs']
            n_dists = config['n_dists']
            lr = config['lr']
            batch_size = config['batch_size']
            layers = config['layers']
            weight_decay = config['weight_decay']
            dropout_rate = config['dropout_rate']
            model = MENSA(n_features, layers=layers, dropout_rate=dropout_rate,
                          n_events=n_events, n_dists=n_dists, device=device)
            model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
                      weight_decay=weight_decay, patience=10,
                      batch_size=batch_size, verbose=False)
        else:
            raise NotImplementedError()
        
        # Compute survival function
        if model_name in ['coxph', 'coxboost', 'rsf']:
            all_preds = []
            for trained_model in trained_models:
                model_preds = trained_model.predict_survival_function(test_dict['X'].cpu())
                model_preds = np.row_stack([fn(time_bins.cpu().numpy()) for fn in model_preds])
                spline = interp1d(time_bins.cpu().numpy(), model_preds,
                                  kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()),
                                     columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "deepsurv":
            all_preds = []
            for trained_model in trained_models:
                preds, time_bins_model = make_deepsurv_prediction(trained_model, test_dict['X'].to(device),
                                                                  config=config, dtype=dtype)
                spline = interp1d(time_bins_model.cpu().numpy(),
                                  preds.cpu().numpy(),
                                  kind='linear', fill_value='extrapolate')
                preds = pd.DataFrame(spline(time_bins.cpu().numpy()),
                                     columns=time_bins.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "deephit":
            cif = model.predict_cif(test_dict['X']).cpu().numpy()
            all_preds = []
            for i in range(n_events):
                preds = pd.DataFrame((1-cif[i]).T, columns=time_bins_dh.cpu().numpy())
                all_preds.append(preds)
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(torch.tensor(test_data[0], dtype=dtype), model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            all_preds = []
            for i in range(n_events):
                preds = pd.DataFrame(event_preds[i], columns=bin_locations)
                all_preds.append(preds)
        elif model_name == "mtlrcr":
            pred_prob = model(test_dict['X'])
            num_points = len(time_bins.cpu().numpy())
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
                model_pred = model.predict_survival(test_dict['X'].cpu().numpy(), t=list(time_bins.cpu().numpy()), risk=i+1)
                model_pred = pd.DataFrame(model_pred, columns=time_bins.cpu().numpy())
                all_preds.append(model_pred)
        elif model_name == "mensa":
            all_preds = []
            for i in range(n_events):
                model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=i+1)
                model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
                all_preds.append(model_preds)
        else:
            raise NotImplementedError()
        
        # Calculate local and global CI
        y_test_time = np.stack([test_dict['T'].cpu().numpy() for _ in range(n_events)], axis=1)
        y_test_event = np.stack([np.array((test_dict['E'].cpu().numpy() == i+1)*1.0)
                                for i in range(n_events)], axis=1)
        all_preds_arr = [df.to_numpy() for df in all_preds]
        global_ci = global_C_index(all_preds_arr, y_test_time, y_test_event)
        local_ci = local_C_index(all_preds_arr, y_test_time, y_test_event)

        # Check for NaN or inf and replace with 0.5
        global_ci = 0.5 if np.isnan(global_ci) or np.isinf(global_ci) else global_ci
        local_ci = 0.5 if np.isnan(local_ci) or np.isinf(local_ci) else local_ci
        
        # Make evaluation for each event
        model_results = pd.DataFrame()
        for event_id, surv_preds in enumerate(all_preds):
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict['T']
            y_train_event = (train_dict['E'] == event_id+1)*1.0
            y_test_time = test_dict['T']
            y_test_event = (test_dict['E'] == event_id+1)*1.0
            
            lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            auc = lifelines_eval.auc()
            ibs = lifelines_eval.integrated_brier_score()
            mae_margin = lifelines_eval.mae(method="Margin")
            d_calib = lifelines_eval.d_calibration()[0]
            
            metrics = [global_ci, local_ci, auc, ibs, mae_margin, d_calib]
            print(f'{model_name}: ' + f'{metrics}')
            
            res_sr = pd.Series([model_name, dataset_name, seed, event_id+1] + metrics,
                                index=["ModelName", "DatasetName", "Seed", "EventId",
                                       "GlobalCI", "LocalCI", "AUC", "MAEM", "DCalib"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
            
        # Save results
        filename = f"{cfg.RESULTS_DIR}/competing_risks.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)
        