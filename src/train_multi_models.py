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
from utility.survival import make_time_bins_hierarchical, digitize_and_convert
from Evaluations.util import make_monotonic, check_monotonicity
from utility.evaluator import LifelinesEvaluator
import torchtuples as tt
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.survival import make_stratified_split_multi
from utility.data import dotdict
from hierarchical import util
from utility.hierarch import format_hyperparams
from multi_evaluator import MultiEventEvaluator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
random.seed(0)

DATASETS = ["rotterdam"]
MODELS = ["direct-full", "hierarch-full"] # "direct-full", "hierarch-full", "mensa"

results = pd.DataFrame()

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    for dataset_name in DATASETS:
        # Load data
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        
        # Calculate time bins
        data = dl.get_data()
        if dataset_name == "seer":
            time_bins = make_time_bins(data[1], event=data[2][:,0])
        else:
            time_bins = make_time_bins(data[1][:,0], event=data[2][:,0])
        
        # Split data
        train_data, valid_data, test_data = dl.split_data(train_size=0.7, valid_size=0.5)
        n_events = dl.n_events
        
        # Impute and scale data
        train_data[0], valid_data[0], test_data[0] = preprocess_data(train_data[0], valid_data[0], test_data[0],
                                                                     cat_features, num_features, as_array=True)
        
        # Train model
        for model_name in MODELS:
            train_start_time = time()
            print(f"Training {model_name}")
            if model_name in ["direct-full", "hierarch-full"]:
                data_settings = load_config(cfg.DATASET_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                num_bins = data_settings['num_bins']
                train_event_bins = make_time_bins_hierarchical(train_data[1], num_bins=num_bins)
                valid_event_bins = make_time_bins_hierarchical(valid_data[1], num_bins=num_bins)
                test_event_bins = make_time_bins_hierarchical(test_data[1], num_bins=num_bins)
                train_data_hierarch = [train_data[0], train_event_bins, train_data[2]]
                valid_data_hierarch = [valid_data[0], valid_event_bins, valid_data[2]]
                test_data_hierarch = [test_data[0], test_event_bins, test_data[2]]
            if model_name == "direct-full":
                model_settings = load_config(cfg.DIRECT_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                hyperparams = format_hyperparams(model_settings)
                verbose = model_settings['verbose']
                model = util.get_model_and_output(model_name, train_data_hierarch, test_data_hierarch,
                                                  valid_data_hierarch, data_settings, hyperparams, verbose)
            elif model_name == "hierarch-full":
                model_settings = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                hyperparams = format_hyperparams(model_settings)
                verbose = model_settings['verbose']
                model = util.get_model_and_output(model_name, train_data_hierarch, test_data_hierarch,
                                                  valid_data_hierarch, data_settings, hyperparams, verbose)
            train_time = time() - train_start_time
            
            # Compute survival function
            test_start_time = time()
            if model_name in ["direct-full", "hierarch-full"]:
                surv_preds = util.get_surv_curves(torch.Tensor(test_data_hierarch[0]), model) # list of surv preds
            else:
                raise NotImplementedError()
            test_time = time() - test_start_time
            
            # Compute metrics (TODO: per event for now)
            for event_id in range(n_events):
                y_train_time = train_data[1][:,event_id]
                y_train_event = train_data[2][:,event_id]
                y_test_time = test_data[1][:,event_id]
                y_test_event = test_data[2][:,event_id]
                surv_pred_event = pd.DataFrame(surv_preds[event_id])
                lifelines_eval = LifelinesEvaluator(surv_pred_event.T, y_test_time, y_test_event,
                                                    y_train_time, y_train_event)
                ci = lifelines_eval.concordance()[0]
                ibs = lifelines_eval.integrated_brier_score()
                d_calib = lifelines_eval.d_calibration()[0]
                ci = lifelines_eval.concordance()[0]
                mae_hinge = lifelines_eval.mae(method="Hinge")
                mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
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
                