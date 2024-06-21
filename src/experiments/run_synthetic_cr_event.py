"""
run_synthetic_cr_event.py
====================================
Experiment 2.1
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
from scipy.interpolate import interp1d

# Local
from data_loader import CompetingRiskSyntheticDataLoader
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function)
from utility.data import (dotdict, format_data, format_data_as_dict_single)
from utility.evaluation import LifelinesEvaluator
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from Evaluations.util import predict_median_survival_time
# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                          make_deephit_model, make_dsm_model, make_rsf_model, train_deepsurv_model,
                          make_deepsurv_prediction, DeepSurv)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.25]
MODELS = ["deepsurv"]
N_SAMPLES = 1000
N_FEATURES = 10

if __name__ == "__main__":
    model_results = pd.DataFrame()
    for dataset_version in DATASET_VERSIONS:
        for copula_name in COPULA_NAMES:
            for k_tau in KENDALL_TAUS:
                # Load and split data
                data_config = load_config(cfg.DATA_CONFIGS_DIR, f"synthetic.yaml")
                dl = CompetingRiskSyntheticDataLoader().load_data(data_config, k_tau=k_tau,
                                                                  n_samples=N_SAMPLES, n_features=N_FEATURES)
                dgps = dl.dgps
                
                num_features, cat_features = dl.get_features()
                train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)

                # Make time bins
                time_bins = make_time_bins(train_dict['T'], event=train_dict['E']) # Use first event for time bins
                
                # Evaluate models
                for model_name in MODELS:
                    if model_name == "cox":
                        config = load_config(cfg.COX_CONFIGS_DIR, f"synthetic.yaml")
                        model1 = make_cox_model(config)
                        model2 = make_cox_model(config)
                        y_train_e1 = convert_to_structured(train_dict['T'], (train_dict['E'] == 0)*1.0)
                        y_train_e2 = convert_to_structured(train_dict['T'], (train_dict['E'] == 1)*1.0)
                        model1.fit(train_dict['X'], y_train_e1)
                        model2.fit(train_dict['X'], y_train_e2)
                    elif model_name == "deepsurv":
                        config = dotdict(cfg.DEEPSURV_PARAMS)
                        model1 = DeepSurv(in_features=N_FEATURES, config=config)
                        model2 = DeepSurv(in_features=N_FEATURES, config=config)
                        data_train1 = pd.DataFrame(train_dict['X'])
                        data_train1['time'] = train_dict['T']
                        data_train1['event'] = (train_dict['E'] == 0)*1.0
                        data_train2 = pd.DataFrame(train_dict['X'])
                        data_train2['time'] = train_dict['T']
                        data_train2['event'] = (train_dict['E'] == 1)*1.0
                        model1 = train_deepsurv_model(model1, data_train1, time_bins, config=config, random_state=0,
                                                      reset_model=True, device=device, dtype=dtype)
                        model2 = train_deepsurv_model(model2, data_train2, time_bins, config=config, random_state=0,
                                                      reset_model=True, device=device, dtype=dtype)
                    elif model_name == "mensa":
                        config = load_config(cfg.MENSA_CONFIGS_DIR, f"synthetic.yaml")
                        model1, model2, model3, copula = make_mensa_model_3_events(N_FEATURES, start_theta=2.0, eps=1e-4,
                                                                                   device=device, dtype=dtype)
                        model1, model2, model3, copula = train_mensa_model_3_events(train_dict, valid_dict, model1, model2, model3,
                                                                                    copula, n_epochs=5000, lr=0.001)
                        print(f"NLL all events: {triple_loss(model1, model2, model3, valid_dict, copula)}")
                        print(f"DGP loss: {triple_loss(dgps[0], dgps[1], dgps[2], valid_dict, copula)}")
                    else:
                        raise NotImplementedError()
                    
                    # Compute survival function
                    if model_name == "cox":
                        preds_e1 = model1.predict_survival_function(test_dict['X'])
                        preds_e1 = np.row_stack([fn(time_bins) for fn in preds_e1])
                        preds_e2 = model2.predict_survival_function(test_dict['X'])
                        preds_e2 = np.row_stack([fn(time_bins) for fn in preds_e2])
                        preds_e1 = pd.DataFrame(preds_e1, columns=time_bins)
                        preds_e2 = pd.DataFrame(preds_e2, columns=time_bins)
                        all_preds = [preds_e1, preds_e2]
                    elif model_name == "deepsurv":
                        preds_e1, time_bins_model1, _ = make_deepsurv_prediction(model1, test_dict['X'],
                                                                                     config=config, dtype=dtype)
                        preds_e2, time_bins_model2, _ = make_deepsurv_prediction(model2, test_dict['X'],
                                                                                     config=config, dtype=dtype)
                        spline1 = interp1d(time_bins_model1, preds_e1, kind='linear', fill_value='extrapolate')
                        spline2 = interp1d(time_bins_model2, preds_e2, kind='linear', fill_value='extrapolate')
                        all_preds = [spline1(time_bins), spline2(time_bins)]
                    elif model_name == "mensa":
                        preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy()
                        preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
                        preds_e1 = pd.DataFrame(preds_e1, columns=time_bins)
                        preds_e2 = pd.DataFrame(preds_e2, columns=time_bins)
                        all_preds = [preds_e1, preds_e2]
                    else:
                        raise NotImplementedError()
                
                    # Make evaluation for each event
                    for event_id, surv_preds in enumerate(all_preds):
                        surv_preds_df = pd.DataFrame(surv_preds, columns=time_bins.numpy())
                        n_train_samples = len(train_dict['X'])
                        n_test_samples= len(test_dict['X'])
                        y_train_time = train_dict[f'T{event_id+1}']
                        y_train_event = np.array([1] * n_train_samples)
                        y_test_time = test_dict[f'T{event_id+1}']
                        y_test_event = np.array([1] * n_test_samples)
                        lifelines_eval = LifelinesEvaluator(surv_preds_df.T, y_test_time, y_test_event,
                                                            y_train_time, y_train_event)
                        ci = lifelines_eval.concordance()[0]
                        ibs = lifelines_eval.integrated_brier_score(num_points=len(time_bins))
                        mae = lifelines_eval.mae(method='Uncensored')
                        d_calib = lifelines_eval.d_calibration()[0]
                        
                        n_samples = test_dict['X'].shape[0]
                        truth_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
                        for i in range(time_bins.shape[0]):
                            truth_preds[:,i] = dgps[event_id].survival(time_bins[i], test_dict['X'])
                        survival_l1 = float(compute_l1_difference(truth_preds, surv_preds, n_samples, steps=time_bins))
                        
                        metrics = [ci, ibs, mae, survival_l1, d_calib]
                        print(metrics)
                        res_sr = pd.Series([model_name, dataset_version, copula_name, k_tau] + metrics,
                                           index=["ModelName", "DatasetVersion", "Copula", "KTau", "CI", "IBS", "MAE", "L1", "DCalib"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                        model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")