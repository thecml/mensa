import numpy as np
import random
import pandas as pd
from pathlib import Path
import joblib
from time import time
from utility.config import load_config
from pycox.evaluation import EvalSurv
import torch
import math
from utility.survival import coverage
from scipy.stats import chisquare
from utility.survival import convert_to_structured
from utility.data import dotdict
from data_loader import get_data_loader
from utility.survival import make_time_bins, preprocess_data
from sota_builder import *
import config as cfg
from utility.survival import compute_survival_curve, calculate_event_times
from Evaluations.util import make_monotonic, check_monotonicity
from utility.evaluator import LifelinesEvaluator
import torchtuples as tt
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.survival import make_stratified_split_multi

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
random.seed(0)

DATASETS = ["rotterdam"] #"mimic", "seer", "rotterdam"
MODELS = ["cox"] #"coxboost", "rsf", "mtlr"

results = pd.DataFrame()

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # For each dataset
    for dataset_name in DATASETS:
        
        # Load data and split it
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        data_pkg = dl.split_data(train_size=0.7, valid_size=0.5)
        n_events = dl.n_events
    
        time_bins = list()
        for event_id in range(n_events):
            result = torch.tensor(make_time_bins(data_pkg[0][1][:,event_id],
                                                 event=data_pkg[0][2][:,event_id]), dtype=torch.float)
            time_bins.append(result)
        time_bins = torch.cat(time_bins, dim=0).sort()[0]
        
        for event_id in range(n_events):
            train_data = [data_pkg[0][0], data_pkg[0][1][:,event_id], data_pkg[0][2][:,event_id]]
            valid_data = [data_pkg[1][0], data_pkg[1][1][:,event_id], data_pkg[1][2][:,event_id]]
            test_data = [data_pkg[2][0], data_pkg[2][1][:,event_id], data_pkg[2][2][:,event_id]]
            
            # Define X matrices
            X_train = train_data[0]
            X_valid = valid_data[0]
            X_test = test_data[0]
            
            # Define y vectors
            y_train = convert_to_structured(train_data[1], train_data[2])
            y_valid = convert_to_structured(valid_data[1], valid_data[2])
            y_test = convert_to_structured(test_data[1], test_data[2])
            
            # Make event times
            time_bins = make_time_bins(train_data[1], event=train_data[2])
        
            # Scale data
            X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features, num_features)
            
            # Convert to array
            X_train_arr = np.array(X_train, dtype=np.float32)
            X_valid_arr = np.array(X_valid, dtype=np.float32)
            X_test_arr = np.array(X_test, dtype=np.float32)
        
            # Train models
            for model_name in MODELS:
                train_start_time = time()
                print(f"Training {model_name} for event {event_id}")
                if model_name == "cox":
                    config = load_config(cfg.COX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    model = make_cox_model(config)
                    model.fit(X_train_arr, y_train)
                elif model_name == "coxboost":
                    config = load_config(cfg.COXBOOST_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    model = make_coxboost_model(config)
                    model.fit(X_train_arr, y_train)
                elif model_name == "rsf":
                    config = load_config(cfg.RSF_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    model = make_rsf_model(config)
                    model.fit(X_train_arr, y_train)
                elif model_name == "mtlr":
                    data_train = X_train.copy()
                    data_train["time"] = pd.Series(y_train['time'])
                    data_train["event"] = pd.Series(y_train['event']).astype(int)
                    data_valid = X_valid.copy()
                    data_valid["time"] = pd.Series(y_valid['time'])
                    data_valid["event"] = pd.Series(y_valid['event']).astype(int)
                    config = dotdict(load_config(cfg.MTLR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
                    n_features = X_train_arr.shape[1]
                    num_time_bins = len(time_bins)
                    model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
                    model = train_mtlr_model(model, data_train, data_valid, time_bins,
                                             config, random_state=0, reset_model=True, device=device)
                train_time = time() - train_start_time
                    
                # Compute survival function
                test_start_time = time()
                if model_name in ['cox', 'coxboost', 'rsf']:
                    surv_preds = model.predict_survival_function(X_test_arr)
                    surv_preds = np.row_stack([fn(time_bins) for fn in surv_preds])
                    surv_preds = pd.DataFrame(surv_preds, columns=time_bins.numpy())
                elif model_name == "deephit-single":
                    surv_preds = model.predict_surv_df(X_test_arr)
                    surv_preds = pd.DataFrame(surv_preds.T, columns=time_bins.numpy())
                elif model_name == "mtlr":
                    data_test = X_test.copy()
                    data_test["time"] = pd.Series(y_test['time'])
                    data_test["event"] = pd.Series(y_test['event']).astype(int)
                    mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                  dtype=torch.float32, device=device)
                    survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, time_bins, config)
                    surv_preds = survival_outputs.numpy()
                    time_bins_th = torch.cat([torch.tensor([0]).to(time_bins.device), time_bins], 0)
                    surv_preds = pd.DataFrame(surv_preds, columns=time_bins_th.numpy())
                else:
                    raise NotImplementedError()
                test_time = time() - test_start_time

                # Compute metrics
                lifelines_eval = LifelinesEvaluator(surv_preds.T, test_data[1].flatten(), test_data[2].flatten(),
                                                    train_data[1].flatten(), train_data[2].flatten())
                mae_hinge = lifelines_eval.mae(method="Hinge")
                mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                ibs = lifelines_eval.integrated_brier_score()
                d_calib = lifelines_eval.d_calibration()[0]
                ci = lifelines_eval.concordance()[0]
                print(ci)
                
                # Save to df
                metrics = [ci, ibs, mae_hinge, mae_pseudo, d_calib, train_time, test_time]
                res_df = pd.DataFrame(np.column_stack(metrics), columns=["CI", "IBS", "MAEHinge", "MAEPseudo",
                                                                         "DCalib", "TrainTime", "TestTime"])
                res_df['ModelName'] = model_name
                res_df['DatasetName'] = dataset_name
                res_df['EventId'] = event_id
                results = pd.concat([results, res_df], axis=0)
                
                # Save results
                results.to_csv(Path.joinpath(cfg.RESULTS_DIR, f"sota_single_results.csv"), index=False)
    