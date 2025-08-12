"""
run_single_event.py
====================================
Datasets: seer_se, support_se, mimic_se
Models: ["deepsurv", "deephit", "dsm", "mtlr", "mensa"]
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
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured)
from utility.data import dotdict
from utility.config import load_config
from mensa.model import MENSA
from utility.data import format_data_deephit_single
from data_loader import get_data_loader

# SOTA
from sota_models import (make_coxnet_model, make_coxph_model, make_coxboost_model, make_dsm_model, make_rsf_model, make_weibull_aft_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv, make_deephit_single, train_deephit_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# Setup precision
dtype = torch.float32
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ["coxph", "coxnet", "coxboost", "rsf", "weibullaft", "deepsurv", "deephit", "mtlr", "dsm", "mensa"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='seer_se')
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    
    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
        
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features,
                                                num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    train_dict['E'] = torch.tensor(train_dict['E'], device=device, dtype=torch.int32)
    train_dict['T'] = torch.tensor(train_dict['T'], device=device, dtype=torch.float32)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    valid_dict['E'] = torch.tensor(valid_dict['E'], device=device, dtype=torch.int32)
    valid_dict['T'] = torch.tensor(valid_dict['T'], device=device, dtype=torch.float32)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    test_dict['E'] = torch.tensor(test_dict['E'], device=device, dtype=torch.int32)
    test_dict['T'] = torch.tensor(test_dict['T'], device=device, dtype=torch.float32)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    # Format data to work easier with sksurv API
    n_features = train_dict['X'].shape[1]
    X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    X_valid = pd.DataFrame(valid_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    y_train = convert_to_structured(train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy())
    y_valid = convert_to_structured(valid_dict['T'].cpu().numpy(), valid_dict['E'].cpu().numpy())
    y_test = convert_to_structured(test_dict['T'].cpu().numpy(), test_dict['E'].cpu().numpy())
    
    # Evaluate each model
    for model_name in MODELS:
        # Reset seeds
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        
        if model_name == "coxph":
            config = load_config(cfg.COXPH_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = make_coxph_model(config)
            model.fit(X_train, y_train)
        elif model_name == "coxnet":
            config = load_config(cfg.COXBOOST_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = make_coxnet_model(config)
            model.fit(X_train, y_train)
        elif model_name == "coxboost":
            config = load_config(cfg.COXBOOST_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = make_coxboost_model(config)
            model.fit(X_train, y_train)
        elif model_name == "rsf":
            config = load_config(cfg.RSF_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = make_rsf_model(config)
            model.fit(X_train, y_train)
        elif model_name == "weibullaft":
            config = load_config(cfg.WEIBULLAFT_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = make_weibull_aft_model(config)
            model.fit(X_train, y_train)
        elif model_name == "dsm":
            config = load_config(cfg.DSM_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            n_iter = config['n_iter']
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
            model = make_dsm_model(config)
            model.fit(train_dict['X'].cpu().numpy(), train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy(),
                      val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'].cpu().numpy(), valid_dict['E'].cpu().numpy()),
                      learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
        elif model_name == "deepsurv":
            config = load_config(cfg.DEEPSURV_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = DeepSurv(in_features=n_features, config=config)
            data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
            data_train['time'] = train_dict['T'].cpu().numpy()
            data_train['event'] = train_dict['E'].cpu().numpy()
            data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
            data_valid['time'] = valid_dict['T'].cpu().numpy()
            data_valid['event'] = valid_dict['E'].cpu().numpy()
            model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                         random_state=0, reset_model=True, device=device, dtype=dtype)
        elif model_name == "deephit":
            config = load_config(cfg.DEEPHIT_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            model = make_deephit_single(in_features=n_features, out_features=len(time_bins),
                                        time_bins=time_bins.cpu().numpy(), device=device, config=config)
            labtrans = model.label_transform
            train_data, valid_data, out_features, duration_index = format_data_deephit_single(train_dict, valid_dict, labtrans)
            model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                        (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
        elif model_name == "mtlr":
            data_train = X_train.copy()
            data_train["time"] = pd.Series(y_train['time'])
            data_train["event"] = pd.Series(y_train['event']).astype(int)
            data_valid = X_valid.copy()
            data_valid["time"] = pd.Series(y_valid['time'])
            data_valid["event"] = pd.Series(y_valid['event']).astype(int)
            config = load_config(cfg.MTLR_CONFIGS_DIR, f"{dataset_name.partition('_')[0]}.yaml")
            num_time_bins = len(time_bins)
            model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
            model = train_mtlr_model(model, data_train, data_valid, time_bins.cpu().numpy(),
                                     config, random_state=0, dtype=dtype,
                                     reset_model=True, device=device)
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
                          n_events=1, n_dists=n_dists, device=device)
            model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
                      weight_decay=weight_decay, patience=20,
                      batch_size=batch_size, verbose=False)
        else:
            raise NotImplementedError()
        
        # Compute survival function
        if model_name in ["coxph", "coxnet", "coxboost", "rsf"]:
            model_preds = model.predict_survival_function(X_test)
            model_preds = np.row_stack([fn(model.unique_times_) for fn in model_preds])
            spline = interp1d(model.unique_times_, model_preds,
                              kind='linear', fill_value='extrapolate')
            extra_preds = spline(time_bins.cpu().numpy())
            extra_preds = np.minimum(extra_preds, 1)
            model_preds = pd.DataFrame(extra_preds, columns=time_bins.cpu().numpy())
        elif model_name == "weibullaft":
            times_numpy = time_bins.cpu().numpy()
            X_test_df = pd.DataFrame(X_test, columns=model.feature_names_)
            surv_df = model.model.predict_survival_function(X_test_df, times=times_numpy)
            preds_array = np.minimum(np.asarray(surv_df.T), 1.0)
            model_preds = pd.DataFrame(preds_array, columns=times_numpy)
        elif model_name == 'dsm':
            model.torch_model.float()
            X_np  = test_dict['X'].detach().cpu().numpy().astype(np.float32, copy=False)
            t_list = time_bins.detach().cpu().numpy().astype(np.float32, copy=False).tolist()  
            model_preds = model.predict_survival(X_np, t=t_list)
        elif model_name == "deepsurv":
            model_preds, time_bins_deepsurv = make_deepsurv_prediction(model, test_dict['X'].to(device),
                                                                       config=config, dtype=dtype)
            spline = interp1d(time_bins_deepsurv.cpu().numpy(),
                              model_preds.cpu().numpy(),
                              kind='linear', fill_value='extrapolate')
            model_preds = spline(time_bins.cpu().numpy())
        elif model_name == "mtlr":
            data_test = X_test.copy()
            data_test["time"] = pd.Series(y_test['time'])
            data_test["event"] = pd.Series(y_test['event']).astype('int')
            mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                          dtype=dtype, device=device)
            survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, time_bins, config)
            model_preds = survival_outputs[:, 1:].cpu().numpy()
        elif model_name == "deephit":
            model_preds = model.predict_surv(test_dict['X']).cpu().numpy()
        elif model_name == "mensa":
            model_preds = model.predict(test_dict['X'], time_bins, risk=1)
        else:
            raise NotImplementedError()
        
        # Make evaluation
        model_results = pd.DataFrame()
        surv_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        
        y_train_time = train_dict['T'].cpu().numpy()
        y_train_event = train_dict['E'].cpu().numpy()
        y_test_time = test_dict['T'].cpu().numpy()
        y_test_event = test_dict['E'].cpu().numpy()
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        
        ci = lifelines_eval.concordance()[0]

        time_points = np.quantile(y_test_time[y_test_event == 1], [0.25, 0.5, 0.75])
        aucs = []
        for t in time_points:
            try:
                auc = lifelines_eval.auc(t)
            except ValueError:
                auc = 0.5
            aucs.append(auc)
        mean_auc = np.mean(aucs)
        
        ibs = lifelines_eval.integrated_brier_score()
        mae_margin = lifelines_eval.mae(method="Margin")
        d_calib = lifelines_eval.d_calibration()[0]
        
        metrics = [ci, mean_auc, ibs, mae_margin, d_calib]
        print(f'{model_name}: ' + f'{metrics}')
        res_sr = pd.Series([model_name, dataset_name, seed] + metrics,
                            index=["ModelName", "DatasetName", "Seed",
                                   "CI", "AUC", "IBS", "MAEM", "DCalib"])
        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/single_event.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)