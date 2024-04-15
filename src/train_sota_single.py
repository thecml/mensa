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
from utility.survival import make_time_bins, impute_and_scale
from sota_builder import *
import config as cfg
from utility.survival import compute_survival_curve, calculate_event_times
from Evaluations.util import make_monotonic, check_monotonicity
from utility.evaluator import LifelinesEvaluator
import torchtuples as tt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
random.seed(0)

DATASETS = ["als", "mimic", "seer", "rotterdam"]
MODELS = ["cox", "coxboost", "rsf", "deephit-single"]

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
        df = dl.split_data()
        
        n_events = df[0][0].shape[1]
        for event_id in range(n_events):
            train_data = [df[0][0], df[0][1][:,event_id], df[0][2][:,event_id]]
            valid_data = [df[1][0], df[1][1][:,event_id], df[1][2][:,event_id]]
            test_data = [df[2][0], df[2][1][:,event_id], df[2][2][:,event_id]]
            
            # Define X matrices
            X_train = train_data[0]
            X_valid = valid_data[0]
            X_test = test_data[0]
            
            # Define y vectors
            y_train = convert_to_structured(train_data[1], train_data[2])
            y_valid = convert_to_structured(valid_data[1], valid_data[2])
            y_test = convert_to_structured(test_data[1], test_data[2])
            
            # Make event times
            #time_bins = make_time_bins(train_data[1], event=train_data[2])
            time_bins = calculate_event_times(y_train['time'].copy(), y_train['event'])
        
            # Scale data
            X_train, X_valid, X_test = impute_and_scale(X_train, X_valid, X_test, cat_features, num_features)
            
            # Convert to array
            X_train_arr = np.array(X_train)
            X_valid_arr = np.array(X_valid)
            X_test_arr = np.array(X_test)
        
            # Train models
            for model_name in MODELS:
                print(f"Training {model_name}")
                if model_name == "cox":
                    config = load_config(cfg.COX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    model = make_cox_model(config)
                    train_start_time = time()
                    model.fit(X_train_arr, y_train)
                    train_time = time() - train_start_time
                elif model_name == "coxboost":
                    config = load_config(cfg.COXBOOST_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    model = make_coxboost_model(config)
                    train_start_time = time()
                    model.fit(X_train_arr, y_train)
                    train_time = time() - train_start_time
                elif model_name == "rsf":
                    config = load_config(cfg.RSF_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    model = make_rsf_model(config)
                    train_start_time = time()
                    model.fit(X_train_arr, y_train)
                    train_time = time() - train_start_time
                elif model_name == "deephit-single":
                    config = load_config(cfg.DEEPHIT_SINGLE_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                    in_features = X_train_arr.shape[1]
                    
                    
                    model = make_deephit_single_model(config, in_features, out_features, duration_index)
                    train_start_time = time()
                    model.fit(X_train_arr, y_train)
                    train_time = time() - train_start_time
            
                # Compute survival function
                test_start_time = time()
                if model_name in ['cox', 'coxboost', 'rsf']:
                    surv_preds = model.predict_survival_function(X_test_arr)
                    surv_preds = np.row_stack([fn(time_bins) for fn in surv_preds])
                elif model_name == "deephit-single":
                    print(0)
                else:
                    raise NotADirectoryError()
                test_time = time() - test_start_time

                # Make dataframe
                surv_preds = pd.DataFrame(surv_preds, columns=time_bins)
                
                # Compute metrics
                lifelines_eval = LifelinesEvaluator(surv_preds.T, test_data[1].flatten(), test_data[2].flatten(),
                                                    train_data[1].flatten(), train_data[2].flatten())
                mae_hinge = lifelines_eval.mae(method="Hinge")
                mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                ibs = lifelines_eval.integrated_brier_score()
                d_calib = lifelines_eval.d_calibration()[0]
                ev = EvalSurv(surv_preds.T, y_test["time"], y_test["event"], censor_surv="km")
                ci = ev.concordance_td() # TODO: Decide on CI or CTD
                
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
    