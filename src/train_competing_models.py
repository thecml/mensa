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
from data_loader import get_data_loader, get_hiearch_data_settings, get_hiearch_model_settings
from utility.survival import make_time_bins, impute_and_scale
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
random.seed(0)

DATASETS = ["seer"] #"mimic", "seer", "rotterdam"
MODELS = ["deephit-cr"] # "direct-full", "hierarch-full"

results = pd.DataFrame()

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

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
        
        # Make dataframe with X, y
        df = pd.DataFrame(data[0])
        df['time'] = data[1]
        df['event'] = [next((i+1 for i, val in enumerate(subarr)
                             if val == 1), 0) for subarr in data[2]]
        
        # Split data
        df_train = df
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)

        get_x = lambda df: (df
                            .drop(columns=['time', 'event'])
                            .values.astype('float32'))
        
        x_train = get_x(df_train)
        x_val = get_x(df_val)
        x_test = get_x(df_test)

        # Make event times. TODO: uses first even
        time_bins = make_time_bins(df_train['time'], event=df_train['event'])
        
        # Train model
        for model_name in MODELS:
            train_start_time = time()
            print(f"Training {model_name}")
            if model_name == "deephit-cr":
                config = load_config(cfg.DEEPHIT_CR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                labtrans = LabTransform(len(time_bins))
                get_target = lambda df: (df['time'].values, df['event'].values) 
                y_train = labtrans.fit_transform(*get_target(df_train))
                y_val = labtrans.transform(*get_target(df_val))
                durations_train, events_train = get_target(df_train)
                durations_test, events_test = get_target(df_test)
                val = (x_val, y_val)
                in_features = x_train.shape[1]
                duration_index = labtrans.cuts
                out_features = len(labtrans.cuts)
                num_risks = y_train[1].max()
                model = make_deephit_cr_model(config, in_features, out_features, num_risks, duration_index)
                epochs = 5
                batch_size = config['batch_size']
                verbose = config['verbose']
                if config['early_stop']:
                    callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
                else:
                    callbacks = []
                model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val)
            train_time = time() - train_start_time
            
            # Compute survival function
            test_start_time = time()
            if model_name == "deephit-cr":
                # The survival function obtained with predict_surv_df is the probability of surviving any of the events,
                # and does, therefore, not distinguish between the event types. This means that we evaluate this "single-event case" as before.
                surv = model.predict_surv_df(x_test)
                survival_outputs = pd.DataFrame(surv.T)
            test_time = time() - test_start_time
            
            # Evaluate
            lifelines_eval = LifelinesEvaluator(survival_outputs.T, durations_test, events_test != 0,
                                                durations_train, events_train != 0)
            ci = lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score()
            d_calib = lifelines_eval.d_calibration()[0]
            ci = lifelines_eval.concordance()[0]
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                
            # Save to df
            metrics = [ci, ibs, mae_hinge, mae_pseudo, d_calib, train_time, test_time]
            res_df = pd.DataFrame(np.column_stack(metrics), columns=["CI", "IBS", "MAEHinge", "MAEPseudo",
                                                                     "DCalib", "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            results = pd.concat([results, res_df], axis=0)
            
            # Save results
            results.to_csv(Path.joinpath(cfg.RESULTS_DIR, f"sota_cr_results.csv"), index=False)
