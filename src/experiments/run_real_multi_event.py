"""
run_synthetic_competing_risks.py
====================================
Experiment 2.1

Datasets: ALS
Models: ["deepsurv", 'hierarch', 'mensa']
"""

# 3rd party
import pandas as pd
import numpy as np
import config as cfg
import torch
import torch.optim as optim
import torch.nn as nn
import random
import warnings
import copy
import tqdm
import math
import argparse
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import (dotdict, format_data)
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from utility.data import (format_data_deephit_cr, format_hierarchical_data, calculate_layer_size_hierarch,
                          format_survtrace_data, format_data_as_dict_multi)
from utility.evaluation import global_C_index, local_C_index
from data_loader import ALSDataLoader

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                          make_deephit_cr, make_dsm_model, make_rsf_model, train_deepsurv_model,
                          make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction, train_mtlr_cr
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function
from hierarchical.data_settings import synthetic_cr_settings
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from torchmtlr.utils import encode_mtlr_format, reset_parameters, encode_mtlr_format_no_censoring
from torchmtlr.model import MTLRCR, mtlr_neg_log_likelihood, mtlr_risk, mtlr_survival
from utility.data import calculate_vocab_size
from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceMulti
from survtrace.train_utils import Trainer
from survtrace.config import STConfig
from utility.data import calculate_vocab_size
from pycox.models import DeepHit
from utility.data import (format_data_deephit_cr, format_hierarch_data_multi_event,
                          calculate_layer_size_hierarch)

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
MODELS = ['hierarch']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    seed = args.seed
    
    # Load and split data
    dl = ALSDataLoader().load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=seed)
    n_events = dl.n_events
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    event_cols = ['e1', 'e2', 'e3', 'e4']
    time_cols = ['t1', 't2', 't3', 't4']
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    train_dict['E'] = torch.tensor(train_dict['E'], device=device, dtype=torch.int64)
    train_dict['T'] = torch.tensor(train_dict['T'], device=device, dtype=torch.int64)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    valid_dict['E'] = torch.tensor(valid_dict['E'], device=device, dtype=torch.int64)
    valid_dict['T'] = torch.tensor(valid_dict['T'], device=device, dtype=torch.int64)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    test_dict['E'] = torch.tensor(test_dict['E'], device=device, dtype=torch.int64)
    test_dict['T'] = torch.tensor(test_dict['T'], device=device, dtype=torch.int64)
    
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)
    
    # Evaluate models
    model_results = pd.DataFrame()
    for model_name in MODELS:
        if model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            trained_models = []
            for i in range(n_events):
                model = DeepSurv(in_features=n_features, config=config)
                data_train = pd.DataFrame(train_dict['X'])
                data_train['time'] = train_dict['T'][:,i]
                data_train['event'] = train_dict['E'][:,i]
                data_valid = pd.DataFrame(valid_dict['X'])
                data_valid['time'] = valid_dict['T'][:,i]
                data_valid['event'] = valid_dict['E'][:,i]
                model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                             random_state=0, reset_model=True, device=device, dtype=dtype)
                trained_models.append(model)
        elif model_name == "hierarch":
            config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"als.yaml")
            n_time_bins = len(time_bins)
            train_data, valid_data, test_data = format_hierarch_data_multi_event(train_dict, valid_dict,
                                                                                 test_dict, n_time_bins)
            config['min_time'] = int(train_data[1].min())
            config['max_time'] = int(train_data[1].max())
            config['num_bins'] = n_time_bins
            params = cfg.HIERARCH_PARAMS
            params['n_batches'] = int(n_samples/params['batch_size'])
            layer_size = params['layer_size_fine_bins'][0][0]
            params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
            hyperparams = format_hierarchical_hyperparams(params)
            verbose = params['verbose']
            model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                              valid_data, config, hyperparams, verbose)
        elif model_name == "mensa":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        # Compute survival function
        n_samples = test_dict['X'].shape[0]                    
        if model_name == "deepsurv":
            all_preds = []
            for trained_model in trained_models:
                preds, time_bins_model, _ = make_deepsurv_prediction(trained_model, test_dict['X'],
                                                                     config=config, dtype=dtype)
                preds = pd.DataFrame(preds, columns=time_bins_model)
                all_preds.append(preds)
        elif model_name == "hierarch":
            event_preds = util.get_surv_curves(test_data[0], model)
            bin_locations = np.linspace(0, config['max_time'], event_preds[0].shape[1])
            all_preds = []
            for i in range(len(event_preds)):
                preds = pd.DataFrame(event_preds[i], columns=bin_locations)
                all_preds.append(preds)
        elif model_name == "mensa":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        # Test local and global CI
        all_preds_arr = [df.to_numpy() for df in all_preds] # convert dataframe to numpy
        global_ci = global_C_index(all_preds, test_dict['T'].numpy(), test_dict['E'].numpy())
        local_ci = local_C_index(all_preds, test_dict['T'].numpy(), test_dict['E'].numpy()) #TODO Check this
    
        # Make evaluation for each event
        for event_id, surv_preds in enumerate(all_preds):
            surv_preds_df = pd.DataFrame(surv_preds, columns=time_bins.numpy())
            n_train_samples = len(train_dict['X'])
            n_test_samples= len(test_dict['X'])
            y_train_time = train_dict['T'][:,event_id]
            y_train_event = train_dict['E'][:,event_id]
            y_test_time = test_dict['T'][:,event_id]
            y_test_event = test_dict['E'][:,event_id]
            
            lifelines_eval = LifelinesEvaluator(surv_preds_df.T, y_test_time, y_test_event,
                                                y_train_time, y_train_event)
            
            ci = lifelines_eval.concordance()[0]
            ibs = lifelines_eval.integrated_brier_score()
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_margin = lifelines_eval.mae(method="Margin")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            d_calib = lifelines_eval.d_calibration()[0]
            
            metrics = [ci, ibs, mae_hinge, mae_margin, mae_pseudo, d_calib, global_ci, local_ci]
            print(metrics)
            res_sr = pd.Series([model_name] + metrics,
                                index=["ModelName", "CI", "IBS", "MAEH", "MAEM",
                                       "MAEPO", "DCalib", "GlobalCI", "LocalCI"])
            model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
            model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")
            