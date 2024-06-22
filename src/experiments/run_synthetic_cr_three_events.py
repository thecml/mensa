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
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from data_loader import CompetingRiskSyntheticDataLoader
from copula import Clayton2D, Frank2D
from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear
from utility.survival import (make_time_bins, preprocess_data, convert_to_structured,
                              risk_fn, compute_l1_difference, predict_survival_function,
                              make_times_hierarchical)
from utility.data import (dotdict, format_data, format_data_as_dict_single)
from utility.config import load_config
from utility.loss import triple_loss
from mensa.model import train_mensa_model_3_events, make_mensa_model_3_events
from utility.data import format_deephit_data, format_hierarchical_data

# SOTA
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from dcsurvival.model import train_dcsurvival_model
from sota_models import (make_cox_model, make_coxnet_model, make_coxboost_model, make_dcph_model,
                          make_deephit_multi, make_dsm_model, make_rsf_model, train_deepsurv_model,
                          make_deepsurv_prediction, DeepSurv, make_deephit_multi, train_deephit_model)
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction, train_mtlr_cr
from trainer import independent_train_loop_linear, dependent_train_loop_linear, loss_function
from hierarchical.data_settings import synthetic_settings
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from torchmtlr.utils import encode_survival, reset_parameters, encode_survival_no_censoring
from torchmtlr.model import MTLRCR, mtlr_neg_log_likelihood, mtlr_risk, mtlr_survival

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
MODELS = ["mtlrcr"]
N_SAMPLES = 1000
N_FEATURES = 10
N_EVENTS = 3

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
                        model3 = make_cox_model(config)
                        y_train_e1 = convert_to_structured(train_dict['T'], (train_dict['E'] == 0)*1.0)
                        y_train_e2 = convert_to_structured(train_dict['T'], (train_dict['E'] == 1)*1.0)
                        y_train_e3 = convert_to_structured(train_dict['T'], (train_dict['E'] == 2)*1.0)
                        model1.fit(train_dict['X'], y_train_e1)
                        model2.fit(train_dict['X'], y_train_e2)
                        model3.fit(train_dict['X'], y_train_e3)
                    elif model_name == "deepsurv":
                        config = dotdict(cfg.DEEPSURV_PARAMS)
                        model1 = DeepSurv(in_features=N_FEATURES, config=config)
                        model2 = DeepSurv(in_features=N_FEATURES, config=config)
                        model3 = DeepSurv(in_features=N_FEATURES, config=config)
                        data_train1 = pd.DataFrame(train_dict['X'])
                        data_train1['time'] = train_dict['T']
                        data_train1['event'] = (train_dict['E'] == 0)*1.0
                        data_train2 = pd.DataFrame(train_dict['X'])
                        data_train2['time'] = train_dict['T']
                        data_train2['event'] = (train_dict['E'] == 1)*1.0
                        data_train3 = pd.DataFrame(train_dict['X'])
                        data_train3['time'] = train_dict['T']
                        data_train3['event'] = (train_dict['E'] == 2)*1.0
                        model1 = train_deepsurv_model(model1, data_train1, time_bins, config=config, random_state=0,
                                                      reset_model=True, device=device, dtype=dtype)
                        model2 = train_deepsurv_model(model2, data_train2, time_bins, config=config, random_state=0,
                                                      reset_model=True, device=device, dtype=dtype)
                        model3 = train_deepsurv_model(model3, data_train3, time_bins, config=config, random_state=0,
                                                      reset_model=True, device=device, dtype=dtype)
                    elif model_name == "deephit":
                        config = dotdict(cfg.DEEPHIT_PARAMS)
                        model = make_deephit_multi(in_features=N_FEATURES, out_features=len(time_bins),
                                                   num_risks=3, duration_index=time_bins, config=config)
                        train_data, valid_data, out_features, duration_index = format_deephit_data(train_dict, valid_dict, len(time_bins))
                        model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                                    (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
                    elif model_name == "hierarch":
                        data_settings = synthetic_settings
                        train_data, valid_data, test_data = format_hierarchical_data(train_dict, valid_dict,
                                                                                     test_dict, len(time_bins))
                        data_settings['min_time'] = int(train_dict['T'].min())
                        data_settings['max_time'] = int(train_dict['T'].max())
                        params = cfg.HIERARCH_FULL_PARAMS
                        hyperparams = format_hierarchical_hyperparams(params)
                        verbose = params['verbose']
                        model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                                          valid_data, data_settings, hyperparams, verbose)
                    elif model_name == "mtlrcr":
                        train_times = np.digitize(train_dict['T'], bins=time_bins).astype(np.int64)
                        train_events = train_dict['E'].type(torch.int64)
                        y_train = encode_survival_no_censoring(train_times, train_events + 1, time_bins)
                        num_time_bins = len(time_bins) + 1
                        config = dotdict(cfg.MTLR_PARAMS)
                        model = MTLRCR(in_features=N_FEATURES, num_time_bins=num_time_bins, num_events=N_EVENTS)
                        model = train_mtlr_cr(train_dict['X'], y_train, model, time_bins, num_epochs=10,
                                              lr=1e-3, batch_size=32, verbose=True, device=device, C1=1.)
                    elif model_name == "dcsa":
                        raise NotImplementedError()
                    elif model_name == "dsm":
                        config = load_config(cfg.DSM_CONFIGS_DIR, f"synthetic.yaml")
                        model = make_dsm_model(config)
                        X_train = pd.DataFrame(train_dict['X'], columns=[f'X{i}' for i in range(N_FEATURES)])
                        X_valid = pd.DataFrame(valid_dict['X'], columns=[f'X{i}' for i in range(N_FEATURES)])
                        y_train = pd.DataFrame({'event': train_dict['E'], 'time': train_dict['T']})
                        y_valid = pd.DataFrame({'event': valid_dict['E'], 'time': valid_dict['T']})
                        model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
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
                        preds_e3 = model3.predict_survival_function(test_dict['X'])
                        preds_e3 = np.row_stack([fn(time_bins) for fn in preds_e3])
                        all_preds = [preds_e1, preds_e2, preds_e3]
                    elif model_name == "deepsurv":
                        preds_e1, time_bins_model1, _ = make_deepsurv_prediction(model1, test_dict['X'],
                                                                                 config=config, dtype=dtype)
                        preds_e2, time_bins_model2, _ = make_deepsurv_prediction(model2, test_dict['X'],
                                                                                 config=config, dtype=dtype)
                        preds_e3, time_bins_model3, _ = make_deepsurv_prediction(model3, test_dict['X'],
                                                                                 config=config, dtype=dtype)
                        spline1 = interp1d(time_bins_model1, preds_e1, kind='linear', fill_value='extrapolate')
                        spline2 = interp1d(time_bins_model2, preds_e2, kind='linear', fill_value='extrapolate')
                        spline3 = interp1d(time_bins_model3, preds_e3, kind='linear', fill_value='extrapolate')
                        all_preds = [spline1(time_bins), spline2(time_bins), spline3(time_bins)]
                    elif model_name == "deephit":
                        cif = model.predict_cif(test_dict['X'])
                        model_time_bins = list(model.duration_index.numpy())
                        cif1 = pd.DataFrame(1-cif[0], model_time_bins).T
                        cif2 = pd.DataFrame(1-cif[1], model_time_bins).T
                        cif3 = pd.DataFrame(1-cif[2], model_time_bins).T
                        spline1 = interp1d(model_time_bins, cif1, kind='linear', fill_value='extrapolate')
                        spline2 = interp1d(model_time_bins, cif2, kind='linear', fill_value='extrapolate')
                        spline3 = interp1d(model_time_bins, cif3, kind='linear', fill_value='extrapolate')
                        all_preds = [spline1(time_bins), spline2(time_bins), spline3(time_bins)]
                    elif model_name == "hierarch":
                        event_preds = util.get_surv_curves(test_data[0], model)
                        bin_locations = np.linspace(0, data_settings['max_time'], event_preds[0].shape[1])
                        preds_e1 = pd.DataFrame(event_preds[0], columns=bin_locations)
                        preds_e2 = pd.DataFrame(event_preds[1], columns=bin_locations)
                        preds_e3 = pd.DataFrame(event_preds[2], columns=bin_locations)
                        spline1 = interp1d(bin_locations, preds_e1, kind='linear', fill_value='extrapolate')
                        spline2 = interp1d(bin_locations, preds_e2, kind='linear', fill_value='extrapolate')
                        spline3 = interp1d(bin_locations, preds_e3, kind='linear', fill_value='extrapolate')
                        all_preds = [spline1(time_bins), spline2(time_bins), spline3(time_bins)]
                    elif model_name == "mtlrcr":
                        pred_prob = model(test_dict['X'])
                        num_points = len(time_bins)
                        preds_e1 = mtlr_survival(pred_prob[:,:num_time_bins]).detach().numpy()[:, 1:] # drop extra bin
                        preds_e2 = mtlr_survival(pred_prob[:,num_time_bins:num_time_bins*2]).detach().numpy()[:, 1:]
                        preds_e3 = mtlr_survival(pred_prob[:,num_time_bins*2:]).detach().numpy()[:, 1:]
                        all_preds = [preds_e1, preds_e2, preds_e3]
                    elif model_name == "dcsa":
                        raise NotImplementedError()
                    elif model_name == "dsm":
                        X_test = pd.DataFrame(test_dict['X'], columns=[f'X{i}' for i in range(N_FEATURES)])
                        model_preds = model.predict_survival(X_test, times=list(time_bins.numpy()))
                        all_preds = [model_preds.copy(), model_preds.copy(), model_preds.copy()]
                    elif model_name == "mensa":
                        preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy()
                        preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
                        preds_e3 = predict_survival_function(model3, test_dict['X'], time_bins).detach().numpy()
                        all_preds = [preds_e1, preds_e2, preds_e3]
                    else:
                        raise NotImplementedError()
                
                    # Make evaluation for each event
                    pred_time_bins = torch.cat([torch.tensor([0], device=device, dtype=dtype), time_bins])
                    for event_id, surv_preds in enumerate(all_preds):
                        surv_preds = np.hstack((np.ones((surv_preds.shape[0], 1)), surv_preds))
                        surv_preds_df = pd.DataFrame(surv_preds, columns=pred_time_bins.numpy())
                        n_train_samples = len(train_dict['X'])
                        n_test_samples= len(test_dict['X'])
                        y_train_time = train_dict[f'T{event_id+1}']
                        y_train_event = np.array([1] * n_train_samples)
                        y_test_time = test_dict[f'T{event_id+1}']
                        y_test_event = np.array([1] * n_test_samples)
                        lifelines_eval = LifelinesEvaluator(surv_preds_df.T, y_test_time, y_test_event,
                                                            y_train_time, y_train_event)
                        ci = lifelines_eval.concordance()[0]
                        ibs = lifelines_eval.integrated_brier_score(num_points=len(pred_time_bins))
                        mae = lifelines_eval.mae(method='Uncensored')
                        d_calib = lifelines_eval.d_calibration()[0]
                        
                        n_samples = test_dict['X'].shape[0]
                        truth_preds = torch.zeros((n_samples, pred_time_bins.shape[0]), device=device)
                        for i in range(pred_time_bins.shape[0]):
                            truth_preds[:,i] = dgps[event_id].survival(pred_time_bins[i], test_dict['X'])
                        survival_l1 = float(compute_l1_difference(truth_preds, surv_preds, n_samples, steps=pred_time_bins))
                        
                        metrics = [ci, ibs, mae, survival_l1, d_calib]
                        print(metrics)
                        res_sr = pd.Series([model_name, dataset_version, copula_name, k_tau] + metrics,
                                           index=["ModelName", "DatasetVersion", "Copula", "KTau", "CI", "IBS", "MAE", "L1", "DCalib"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                        model_results.to_csv(f"{cfg.RESULTS_DIR}/model_results.csv")