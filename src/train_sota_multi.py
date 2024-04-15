import numpy as np
import tensorflow as tf

import random
import pandas as pd

from utility.survival import make_time_bins, calculate_event_times, calculate_percentiles, compute_survival_curve
from utility.training import get_data_loader, scale_data, split_time_event
from tools.sota_builder import make_cox_model, make_coxnet_model, make_coxboost_model
from tools.sota_builder import make_rsf_model, make_dsm_model, make_dcph_model, make_dcm_model
from tools.sota_builder import make_baycox_model, make_baymtlr_model
from tools.bnn_isd_trainer import train_bnn_model
from utility.bnn_isd_models import make_ensemble_cox_prediction, make_ensemble_mtlr_prediction
from pathlib import Path
import paths as pt
import joblib
from time import time
from utility.config import load_config
from pycox.evaluation import EvalSurv
import torch
from utility.survival import survival_probability_calibration
from tools.evaluator import LifelinesEvaluator
import math
from utility.survival import coverage
from scipy.stats import chisquare
from utility.risk import InputFunction
from utility.training import make_stratified_split
from utility.survival import convert_to_structured
from tools.Evaluations.util import make_monotonic, check_monotonicity

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
MODELS = ["cox", "coxnet", "coxboost", "rsf", "dsm", "dcm", "baycox", "baymtlr"]

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
        df = dl.get_data()
        
        # Split data
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                            frac_valid=0.1, frac_test=0.2, random_state=0)
        X_train = df_train[cat_features+num_features]
        X_valid = df_valid[cat_features+num_features]
        X_test = df_test[cat_features+num_features]
        y_train = convert_to_structured(df_train["time"], df_train["event"])
        y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
        y_test = convert_to_structured(df_test["time"], df_test["event"])
    
        # Scale data
        X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
        
        # Convert to array
        X_train_arr = np.array(X_train)
        X_valid_arr = np.array(X_valid)
        X_test_arr = np.array(X_test)

        # Make time/event split
        t_train, e_train = split_time_event(y_train)
        t_valid, e_valid = split_time_event(y_valid)
        t_test, e_test = split_time_event(y_test)
        
        # Make event times
        event_times = calculate_event_times(t_train, e_train)
        mtlr_times = make_time_bins(t_train, event=e_train)
        
        # Calculate quantiles
        event_times_pct = calculate_percentiles(event_times)
        mtlr_times_pct = calculate_percentiles(mtlr_times)
        
        for model_name in MODELS:
            print(f"Training {model_name}")
            # Get batch size for MLP to use for loss calculation
            mlp_config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
            batch_size = mlp_config['batch_size']
            
            if model_name in ["baycox", "baymtlr"]:
                # Make data for BayCox/BayMTLR models
                data_train = X_train.copy()
                data_train["time"] = pd.Series(y_train['time'])
                data_train["event"] = pd.Series(y_train['event']).astype(int)
                data_valid = X_valid.copy()
                data_valid["time"] = pd.Series(y_valid['time'])
                data_valid["event"] = pd.Series(y_valid['event']).astype(int)
                data_test = X_test.copy()
                data_test["time"] = pd.Series(y_test['time'])
                data_test["event"] = pd.Series(y_test['event']).astype(int)
                num_features = X_train.shape[1]
                        
            # Make model and train
            if model_name == "cox":
                config = load_config(pt.COX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                model = make_cox_model(config)
                train_start_time = time()
                model.fit(X_train_arr, y_train)
                train_time = time() - train_start_time  
            elif model_name == "coxnet":
                config = load_config(pt.COXNET_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                model = make_coxnet_model(config)
                train_start_time = time()
                model.fit(X_train_arr, y_train)
                train_time = time() - train_start_time
            elif model_name == "dsm":
                config = load_config(pt.DSM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                train_start_time = time()
                model = make_dsm_model(config)
                model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
                train_time = time() - train_start_time
            elif model_name == "dcph":
                config = load_config(pt.DCPH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                model = make_dcph_model(config)
                train_start_time = time()
                model.fit(X_train, t_train, e_train, batch_size=config['batch_size'],
                          iters=config['iters'], val_data=(X_valid, t_valid, e_valid),
                          learning_rate=config['learning_rate'], optimizer=config['optimizer'])
                train_time = time() - train_start_time
            elif model_name == "dcm":
                config = load_config(pt.DCM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                model = make_dcm_model(config)
                train_start_time = time()
                model.fit(X_train, pd.DataFrame(y_train), val_data=(X_valid, pd.DataFrame(y_valid)))
                train_time = time() - train_start_time
            elif model_name == "rsf":
                config = load_config(pt.RSF_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                model = make_rsf_model(config)
                train_start_time = time()
                model.fit(X_train_arr, y_train)
                train_time = time() - train_start_time
            elif model_name == "coxboost":
                config = load_config(pt.COXBOOST_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                model = make_coxboost_model(config)
                train_start_time = time()
                model.fit(X_train_arr, y_train)
                train_time = time() - train_start_time
            elif model_name == "baycox":
                config = dotdict(load_config(pt.BAYCOX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
                model = make_baycox_model(num_features, config)
                train_start_time = time()
                model = train_bnn_model(model, data_train, data_valid, mtlr_times,
                                        config=config, random_state=0, reset_model=True, device=device)
                train_time = time() - train_start_time
            elif model_name == "baymtlr":
                config = dotdict(load_config(pt.BAYMTLR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
                model = make_baymtlr_model(num_features, mtlr_times, config)
                train_start_time = time()
                model = train_bnn_model(model, data_train, data_valid,
                                        mtlr_times, config=config,
                                        random_state=0, reset_model=True, device=device)
                train_time = time() - train_start_time
            
            # Compute survival function
            test_start_time = time()
            if model_name == "dsm":
                surv_preds = model.predict_survival(X_test, times=list(event_times))
            elif model_name == "dcph":
                surv_preds = model.predict_survival(X_test, t=list(event_times))
            elif model_name == "dcm":
                surv_preds = model.predict_survival(X_test, times=list(event_times))
            elif model_name == "rsf": # uses KM estimator instead
                test_surv_fn = model.predict_survival_function(X_test_arr)
                surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
            elif model_name == "coxboost":
                test_surv_fn = model.predict_survival_function(X_test_arr)
                surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
            elif model_name == "baycox":
                baycox_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                dtype=torch.float, device=device)
                survival_outputs, _, ensemble_outputs = make_ensemble_cox_prediction(model, baycox_test_data, config)
                surv_preds = survival_outputs.numpy()
            elif model_name == "baymtlr":
                baycox_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                dtype=torch.float, device=device)
                survival_outputs, _, ensemble_outputs = make_ensemble_mtlr_prediction(model, baycox_test_data, mtlr_times, config)
                surv_preds = survival_outputs.numpy()
            else:
                surv_preds = compute_survival_curve(model, X_train_arr, X_test_arr,
                                                                  e_train, t_train, event_times, model_name)
            test_time = time() - test_start_time
            
            # Check monotonicity
            if not check_monotonicity(surv_preds):
                surv_preds = make_monotonic(surv_preds, event_times, method='ceil')
                
            # Make dataframe
            if model_name == "baymtlr":
                mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
                surv_preds = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
            else:
                surv_preds = pd.DataFrame(surv_preds, columns=event_times)
                
            # Sanitize
            surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
            bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
            sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
            sanitized_y_test = np.delete(y_test, bad_idx, axis=0)
            sanitized_x_test = np.delete(X_test_arr, bad_idx, axis=0)
            sanitized_t_test = np.delete(t_test, bad_idx, axis=0)
            sanitized_e_test = np.delete(e_test, bad_idx, axis=0)

            # Compute metrics
            lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_y_test["time"],
                                                sanitized_y_test["event"], t_train, e_train)
            ibs = lifelines_eval.integrated_brier_score()
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            d_calib = lifelines_eval.d_calibration()[0] # 1 if lifelines_eval.d_calibration()[0] > 0.05 else 0
            km_mse = lifelines_eval.km_calibration()
            ev = EvalSurv(sanitized_surv_preds.T, sanitized_y_test["time"],
                          sanitized_y_test["event"], censor_surv="km")
            inbll = ev.integrated_nbll(event_times)
            ci = ev.concordance_td()
            
            # Calculate C-cal for BNN models
            if model_name in ['baycox', 'baymtlr']:
                baycox_test_data = torch.tensor(sanitized_x_test, dtype=torch.float, device=device)
                if model_name == "baycox":
                    _, _, ensemble_outputs = make_ensemble_cox_prediction(model, baycox_test_data, config)
                else:
                    _, _, ensemble_outputs = make_ensemble_mtlr_prediction(model, baycox_test_data, mtlr_times, config)
                n_samples_test = config['n_samples_test']
                credible_region_sizes = np.arange(0.1, 1, 0.1)
                coverage_stats = {}
                for percentage in credible_region_sizes:
                    drop_num = math.floor(0.5 * n_samples_test * (1 - percentage))
                    lower_outputs = torch.kthvalue(ensemble_outputs, k=1 + drop_num, dim=0)[0]
                    upper_outputs = torch.kthvalue(ensemble_outputs, k=n_samples_test - drop_num, dim=0)[0]
                    if model_name == 'baycox':
                        coverage_stats[percentage] = coverage(event_times, upper_outputs, lower_outputs,
                                                              sanitized_t_test, sanitized_e_test)
                    else:
                        coverage_stats[percentage] = coverage(mtlr_times, upper_outputs, lower_outputs,
                                                              sanitized_t_test, sanitized_e_test)
                data = [list(coverage_stats.keys()), list(coverage_stats.values())]
                _, pvalue = chisquare(data)
                c_calib = pvalue[0]
            else:
                c_calib = 0
        
            # Compute calibration curves
            deltas = dict()
            if model_name != "baymtlr": # use event times for non-mtlr model
                for t0 in event_times_pct.values():
                    _, _, _, deltas_t0 = survival_probability_calibration(sanitized_surv_preds,
                                                                          sanitized_y_test["time"],
                                                                          sanitized_y_test["event"],
                                                                          t0)
                    deltas[t0] = deltas_t0
            else:
                for t0 in mtlr_times_pct.values():
                    _, _, _, deltas_t0 = survival_probability_calibration(sanitized_surv_preds,
                                                                          sanitized_y_test["time"],
                                                                          sanitized_y_test["event"],
                                                                          t0)
                    deltas[t0] = deltas_t0
            ici = deltas[t0].mean()
            
            # Save to df
            metrics = [ci, ibs, mae_hinge, mae_pseudo, d_calib, km_mse, inbll, c_calib, ici, train_time, test_time]
            res_df = pd.DataFrame(np.column_stack(metrics), columns=["CI", "IBS", "MAEHinge", "MAEPseudo", "DCalib", "KM",
                                                                     "INBLL", "CCalib", "ICI", "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            results = pd.concat([results, res_df], axis=0)

            # Save model
            if model_name in ["baycox", "baymtlr"]:
                path = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}.pt")
                torch.save(model.state_dict(), path)
            else:
                path = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}.joblib")
                joblib.dump(model, path)

            # Save results
            results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv"), index=False)
    