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
from utility.survival import preprocess_data
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
from utility.survival import make_time_bins_hierarchical, digitize_and_convert
from utility.data import calculate_vocab_size, format_data_for_survtrace
from survtrace.model import SurvTraceMulti
from survtrace.train_utils import Trainer
from torchmtlr import MTLRCR, mtlr_neg_log_likelihood, mtlr_risk, mtlr_survival
from torchmtlr.utils import encode_survival, reset_parameters
from utility.mtlr import train_mtlr_cr
from torchmtlr.utils import make_time_bins
from models import Mensa
from trainer import train_mensa
from inference import make_ensemble_mtlr_prediction, make_ensemble_mtlr_prediction_multi

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
random.seed(0)

DATASETS = ["rotterdam"] # "mimic", "seer", "rotterdam"
MODELS = ["mensa"] #"deephit", "direct", "hierarch"

results = pd.DataFrame()

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # For each dataset
    for dataset_name in DATASETS:
        # Load data
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        data = dl.get_data()
        
        # Calculate time bins
        # TODO: Implement time bins for competing/multi event
        time_bins = make_time_bins(data[1], event=data[2][:,0])
        
        # Split data
        train_data, valid_data, test_data = dl.split_data(train_size=0.7, valid_size=0.5)
        train_data = [train_data[0], train_data[1], train_data[2]]
        valid_data = [valid_data[0], valid_data[1], valid_data[2]]
        test_data = [test_data[0], test_data[1], test_data[2]]
        n_events = dl.n_events
        
        # Impute and scale data
        train_data[0], valid_data[0], test_data[0] = preprocess_data(train_data[0], valid_data[0], test_data[0],
                                                                     cat_features, num_features,
                                                                     as_array=True)
        # Train model
        df_train = digitize_and_convert(train_data, time_bins)
        df_valid = digitize_and_convert(valid_data, time_bins)
        df_test = digitize_and_convert(test_data, time_bins)
        time_bins = make_time_bins(df_train["time"], event=df_train["event"])
        num_time_bins = len(time_bins) + 1
        X_train = torch.tensor(df_train.drop(['time', 'event'], axis=1).values, dtype=torch.float)
        y_train = encode_survival(df_train['time'], df_train['event'], time_bins)
        y_train = y_train.reshape(y_train.shape[0], num_time_bins, n_events)
        X_valid = torch.tensor(df_valid.drop(['time', 'event'], axis=1).values, dtype=torch.float)
        y_valid = encode_survival(df_valid['time'], df_valid['event'], time_bins)
        y_valid = y_valid.reshape(y_valid.shape[0], num_time_bins, n_events)
        in_features = X_train.shape[1]
        config = dotdict(load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
        model = Mensa(in_features=in_features,
                      num_time_bins=num_time_bins,
                      num_events=2, config=config)
        model = train_mensa(X_train, y_train, X_valid, y_valid, model, time_bins,
                            config=config, random_state=0, reset_model=True, device=device)

        time_bins = torch.from_numpy(time_bins)
        for event_id in range(n_events):
            train_obs = df_train.loc[(df_train['event'] == event_id+1) | (df_train['event'] == 0)]
            test_obs = df_test.loc[(df_train['event'] == event_id+1) | (df_test['event'] == 0)]
            X_test = torch.tensor(test_obs.drop(['time', 'event'], axis=1).values.astype('float32'),
                                  dtype=torch.float)
            y_train_time, y_train_event = train_obs['time'], train_obs['event']
            y_test_time, y_test_event = test_obs['time'], test_obs['event']
            survival_outputs, _, ensemble_outputs = make_ensemble_mtlr_prediction_multi(model, X_test,
                                                                                        time_bins, config,
                                                                                        event_id)
            surv_preds = survival_outputs.numpy()
            surv_preds = pd.DataFrame(surv_preds)
            lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            ci = lifelines_eval.concordance()[0]
            print(ci)
        
        exit(0)

        for event_id in range(n_events):
            # Predict survival function
            if model_name == "mtlrcr":
                train_obs = df_train.loc[(df_train['event'] == event_id+1) | (df_train['event'] == 0)]
                test_obs = df_test.loc[(df_train['event'] == event_id+1) | (df_test['event'] == 0)]
                X_test = torch.tensor(test_obs.drop(['time', 'event'], axis=1).values.astype('float32'),
                                        dtype=torch.float)
                y_train_time, y_train_event = train_obs['time'], train_obs['event']
                y_test_time, y_test_event = test_obs['time'], test_obs['event']
                test_start_time = time()
                pred_prob = model(X_test)
                test_time = time() - test_start_time
                if event_id == 0:
                    survival = mtlr_survival(pred_prob[:,:num_time_bins]).detach().numpy()
                else:
                    survival = mtlr_survival(pred_prob[:,num_time_bins:]).detach().numpy()
                survival_outputs = pd.DataFrame(survival)
                lifelines_eval = LifelinesEvaluator(survival_outputs.T, y_test_time, y_test_event,
                                                    y_train_time, y_train_event)
            elif model_name == "survtrace":
                test_start_time = time()
                surv_pred = model.predict_surv(df_test.drop(['duration', 'proportion'], axis=1),
                                                batch_size=config['batch_size'], event=event_id)
                test_time = time() - test_start_time
                surv_pred = pd.DataFrame(surv_pred)
                y_train_time = np.array(y_train_st[f'event_{event_id}'])
                y_train_event = train_data[2][:,event_id]
                y_test_time = np.array(y_test_st[f'event_{event_id}'])
                y_test_event = test_data[2][:,event_id]
                lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test_time, y_test_event,
                                                    y_train_time, y_train_event)
            elif model_name == "deephit":
                train_obs = df_train.loc[(df_train['event'] == event_id+1) | (df_train['event'] == 0)]
                test_obs = df_test.loc[(df_train['event'] == event_id+1) | (df_test['event'] == 0)]
                x_test = test_obs.drop(['time', 'event'], axis=1).values.astype('float32')
                y_train_time, y_train_event = train_obs['time'], train_obs['event']
                y_test_time, y_test_event = test_obs['time'], test_obs['event']
                test_start_time = time()
                surv = model.predict_surv_df(x_test)
                test_time = time() - test_start_time
                survival_outputs = pd.DataFrame(surv.T)
                lifelines_eval = LifelinesEvaluator(survival_outputs.T, y_test_time, y_test_event,
                                                    y_train_time, y_train_event)
            elif model_name in ["direct", "hierarch"]:
                test_start_time = time()
                surv_preds = util.get_surv_curves(torch.Tensor(test_data_hierarch[0]), model)
                test_time = time() - test_start_time
                y_train_time = train_event_bins[:,event_id]
                y_train_event = train_data[2][:,event_id]
                y_test_time = test_event_bins[:,event_id]
                y_test_event = test_data[2][:,event_id]
                surv_pred_event = pd.DataFrame(surv_preds[event_id])
                lifelines_eval = LifelinesEvaluator(surv_pred_event.T, y_test_time, y_test_event,
                                                    y_train_time, y_train_event)
            else:
                raise NotImplementedError()
            
            # Compute metrics
            ci = lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score()
            d_calib = lifelines_eval.d_calibration()[0]
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            metrics = [ci, ibs, mae_hinge, mae_pseudo, d_calib, train_time, test_time]
            res_df = pd.DataFrame(np.column_stack(metrics), columns=["CI", "IBS", "MAEHinge", "MAEPseudo",
                                                                        "DCalib", "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            res_df['EventId'] = event_id
            results = pd.concat([results, res_df], axis=0)
            
            # Save results
            results.to_csv(Path.joinpath(cfg.RESULTS_DIR, f"sota_comp_results.csv"), index=False)
                 