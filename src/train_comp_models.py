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
from utility.survival import convert_to_structured, convert_to_competing_risk
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
from utility.survival import make_stratified_split_single
from utility.data import dotdict
from hierarchical import util
from utility.hierarch import format_hyperparams
from multi_evaluator import MultiEventEvaluator
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from utility.survival import make_time_bins_hierarchical

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
random.seed(0)

DATASETS = ["rotterdam"] # "mimic", "seer", "rotterdam"
MODELS = ["deephit-comp", "direct-full", "hierarch-full"]

results = pd.DataFrame()

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

# Don't save models
torch.save = None

class LabTransform(LabTransDiscreteTime): # for DeepHit CR
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

if __name__ == "__main__":
    # For each dataset
    for dataset_name in DATASETS:

        # Load data
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        data = dl.get_data()
        
        # Calculate time bins
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
            if model_name == "deephit-comp":
                config = load_config(cfg.DEEPHIT_CR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                labtrans = LabTransform(len(time_bins))
                get_target = lambda df: (df['time'].values, df['event'].values)
                df_train = pd.DataFrame(train_data[0])
                df_train['time'] = train_data[1][:,0]
                df_train['event'] = convert_to_competing_risk(train_data[2])
                df_train = df_train.astype(np.float32)
                df_valid = pd.DataFrame(valid_data[0])
                df_valid['time'] = valid_data[1][:,0]
                df_valid['event'] = convert_to_competing_risk(valid_data[2])
                df_valid = df_valid.astype(np.float32)
                df_test = pd.DataFrame(test_data[0])
                df_test['time'] = test_data[1][:,0]
                df_test['event'] = convert_to_competing_risk(test_data[2])
                df_test = df_test.astype(np.float32)
                y_train = labtrans.fit_transform(*get_target(df_train))
                y_val = labtrans.transform(*get_target(df_valid))
                durations_train, events_train = get_target(df_train)
                durations_test, events_test = get_target(df_test)
                val = (valid_data[0].astype('float32'), y_val)
                in_features = train_data[0].shape[1]
                duration_index = labtrans.cuts
                out_features = len(labtrans.cuts)
                num_risks = y_train[1].max()
                model = make_deephit_cr_model(config, in_features, out_features, num_risks, duration_index)
                epochs = config['epochs']
                batch_size = config['batch_size']
                verbose = config['verbose']
                if config['early_stop']:
                    callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
                else:
                    callbacks = []
                model.fit(train_data[0].astype('float32'), y_train,
                          batch_size, epochs, callbacks, verbose, val_data=val)
            elif model_name in ["direct-full", "hierarch-full"]:
                data_settings = load_config(cfg.DATASET_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                if model_name == "direct-full":
                    model_settings = load_config(cfg.DIRECT_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                else:
                    model_settings = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                num_bins = data_settings['num_bins']
                train_event_bins = make_time_bins_hierarchical(train_data[1], num_bins=num_bins)
                valid_event_bins = make_time_bins_hierarchical(valid_data[1], num_bins=num_bins)
                test_event_bins = make_time_bins_hierarchical(test_data[1], num_bins=num_bins)
                train_data_hierarch = [train_data[0], train_event_bins, train_data[2]]
                valid_data_hierarch = [valid_data[0], valid_event_bins, valid_data[2]]
                test_data_hierarch = [test_data[0], test_event_bins, test_data[2]]
                hyperparams = format_hyperparams(model_settings)
                verbose = model_settings['verbose']
                model = util.get_model_and_output(model_name, train_data_hierarch, test_data_hierarch,
                                                  valid_data_hierarch, data_settings, hyperparams, verbose)
            else:
                raise NotImplementedError()
            train_time = time() - train_start_time   
                
            # Predict survival function
            test_start_time = time()
            if model_name == "deephit-comp":
                # The survival function obtained with predict_surv_df is the probability of surviving any of the events,
                # and does, therefore, not distinguish between the event types. This means that we evaluate this "single-event case" as before.
                surv = model.predict_surv_df(test_data[0].astype('float32'))
                survival_outputs = pd.DataFrame(surv.T)
            elif model_name in ["direct-full", "hierarch-full"]:
                surv_preds = util.get_surv_curves(torch.Tensor(test_data_hierarch[0]), model)
            else:
                raise NotImplementedError()
            test_time = time() - test_start_time
            
            # Evaluate
            for event_id in range(n_events):
                if model_name == "deephit-comp":
                    lifelines_eval = LifelinesEvaluator(survival_outputs.T, durations_test, events_test,
                                                        durations_train, events_train)
                elif model_name in ["direct-full", "hierarch-full"]:
                    y_train_time = train_event_bins[:,event_id]
                    y_train_event = train_data[2][:,event_id]
                    y_test_time = test_event_bins[:,event_id]
                    y_test_event = test_data[2][:,event_id]
                    surv_pred_event = pd.DataFrame(surv_preds[event_id])
                    lifelines_eval = LifelinesEvaluator(surv_pred_event.T, y_test_time, y_test_event,
                                                        y_train_time, y_train_event)
                else:
                    raise NotImplementedError()
            
                ci = lifelines_eval.concordance()[0]
                ibs = lifelines_eval.integrated_brier_score()
                d_calib = lifelines_eval.d_calibration()[0]
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
                results.to_csv(Path.joinpath(cfg.RESULTS_DIR, f"sota_comp_results.csv"), index=False)
