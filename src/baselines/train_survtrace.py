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
from sota_models import *
import config as cfg
from utility.survival import compute_survival_curve, calculate_event_times
from Evaluations.util import make_monotonic, check_monotonicity
from utility.evaluation import LifelinesEvaluator
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

from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceMulti
from survtrace.train_utils import Trainer
from survtrace.config import STConfig
from utility.data import calculate_vocab_size

from utility.survival import LabTransform

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    dl = get_data_loader("seer").load_data()
    num_features, cat_features = dl.get_features()
    data = dl.get_data()
    
    config = load_config(cfg.SURVTRACE_CONFIGS_DIR, f"seer.yaml")
    time_bins = make_time_bins(data[1], event=data[2][:,0])
    
    # Split data
    train_data, valid_data, test_data = dl.split_data(train_size=0.7, valid_size=0.5)
    train_data = [train_data[0][:1000], train_data[1][:1000], train_data[2][:1000]]
    valid_data = [valid_data[0][:1000], valid_data[1][:1000], valid_data[2][:1000]]
    test_data = [test_data[0][:1000], test_data[1][:1000], test_data[2][:1000]]
    n_events = dl.n_events
    
    config['vocab_size'] = calculate_vocab_size(data[0], cat_features)
    
    # Impute and scale data
    train_data[0], valid_data[0], test_data[0] = preprocess_data(train_data[0], valid_data[0], test_data[0],
                                                                 cat_features, num_features, as_array=True)

    df_train = pd.DataFrame(train_data[0]).astype(np.float32)
    df_train['duration'] = np.digitize(train_data[1][:,0], bins=time_bins).astype(int)
    df_train['proportion'] = convert_to_competing_risk(train_data[2]).astype(int)
    df_valid = pd.DataFrame(valid_data[0]).astype(np.float32)
    df_valid['duration'] = np.digitize(valid_data[1][:,0], bins=time_bins).astype(int)
    df_valid['proportion'] = convert_to_competing_risk(valid_data[2]).astype(int)
    df_test = pd.DataFrame(test_data[0]).astype(np.float32)
    df_test['duration'] = np.digitize(test_data[1][:,0], bins=time_bins).astype(int)
    df_test['proportion'] = convert_to_competing_risk(test_data[2]).astype(int)
    
    y_train, y_valid, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(n_events):
        event_name = "event_{}".format(i)
        y_train[event_name] = df_train['duration']
        y_valid[event_name] = df_valid['duration']
        y_test[event_name] = df_test['duration']
    
    y_train['duration'] = df_train['duration']
    y_train['proportion'] = df_train['proportion']
    y_valid['duration'] = df_valid['duration']
    y_valid['proportion'] = df_valid['proportion']
    y_test['duration'] = df_test['duration']
    y_test['proportion'] = df_test['proportion']
    
    duration_index = np.concatenate([[0], time_bins.numpy()])
    out_features = len(duration_index)
    config['duration_index'] = duration_index
    config['out_feature'] = out_features
    config['num_numerical_feature'] = int(len(num_features))
    config['num_categorical_feature'] = int(len(cat_features))
    config['num_feature'] = int(len(num_features)+len(cat_features))
    config['in_features'] = int(len(num_features)+len(cat_features))
    
    model = SurvTraceMulti(dotdict(config))
    
    trainer = Trainer(model)
    train_loss_list, val_loss_list = trainer.fit((df_train.drop(['duration', 'proportion'], axis=1), y_train),
                                                 (df_valid.drop(['duration', 'proportion'], axis=1), y_valid),
                                                 batch_size=config['batch_size'],
                                                 epochs=config['epochs'],
                                                 learning_rate=config['learning_rate'],
                                                 weight_decay=config['weight_decay'],
                                                 val_batch_size=32)
        
    for event_id in range(n_events):
        surv_pred = model.predict_surv(df_test.drop(['duration', 'proportion'], axis=1),
                                       batch_size=32, event=event_id)
        surv_pred = pd.DataFrame(surv_pred)
        y_train_time = np.array(y_train[f'event_{event_id}'])
        y_train_event = train_data[2][:,event_id]
        y_test_time = np.array(y_test[f'event_{event_id}'])
        y_test_event = test_data[2][:,event_id]
        lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
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
        