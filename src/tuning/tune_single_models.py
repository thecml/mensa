import numpy as np
import os
import argparse
import pandas as pd
import config as cfg
from pycox.evaluation import EvalSurv
import torch
from utility.tuning import *
from sota_models import *
from utility.data import dotdict
import data_loader
from utility.survival import make_time_bins, preprocess_data, convert_to_structured
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.evaluation import LifelinesEvaluator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 1
PROJECT_NAME = "mensa_single"

#DATASETS = ["seer"] #"mimic", "als", "rotterdam"
#MODELS = ["cox", "coxboost", "rsf", "mtlr"]

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

def main():
    global model_name
    global dataset_name
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    parser.add_argument('--model', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()
    
    if args.dataset:
        dataset_name = args.dataset
    if args.model:
        model_name = args.model
    """
    
    model_name = "mtlr"
    dataset_name = "rotterdam"
        
    if model_name == "cox":
        sweep_config = get_cox_sweep_config()
    elif model_name == "coxboost":
        sweep_config = get_coxboost_sweep_config()
    elif model_name == "rsf":
        sweep_config = get_rsf_sweep_config()
    elif model_name == "mtlr":
        sweep_config = get_mtlr_sweep_config()
    else:
        raise ValueError("Model not found")
    
    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}_{model_name}')
    wandb.agent(sweep_id, train_deepsurv_model, count=N_RUNS)

def train_deepsurv_model():
# Make and train mdoel
    if model_name == "cox":
        config_defaults = cfg.DEEPSURV_PARAMS
    elif model_name == "coxboost":
        config_defaults = cfg.COXBOOST_PARAMS
    elif model_name == "rsf":
        config_defaults = cfg.RSF_PARAMS
    elif model_name == "mtlr":
        config_defaults = cfg.MTLR_PARAMS
    else:
        raise ValueError("Model not found")
    
    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=dataset_name)
    config = wandb.config
    
    # Load data
    if dataset_name == "seer":
        dl = data_loader.SeerDataLoader().load_data()
    elif dataset_name == "mimic":
        dl = data_loader.MimicDataLoader().load_data()
    elif dataset_name == "als":
        dl = data_loader.ALSDataLoader().load_data()
    elif dataset_name == "rotterdam":
        dl = data_loader.RotterdamDataLoader().load_data()
    else:
        raise ValueError("Dataset not found")
    
    num_features, cat_features = dl.get_features()
    data = dl.get_data()

    n_events = dl.n_events
    ci_results = list()
    for event_id in range(n_events):
        # Split data
        data_pkg = dl.split_data(train_size=0.7, valid_size=0.5)
        train_data = [data_pkg[0][0], data_pkg[0][1][:,event_id], data_pkg[0][2][:,event_id]]
        valid_data = [data_pkg[1][0], data_pkg[1][1][:,event_id], data_pkg[1][2][:,event_id]]
        test_data = [data_pkg[2][0], data_pkg[2][1][:,event_id], data_pkg[2][2][:,event_id]]
    
        # Impute and scale data
        train_data[0], valid_data[0], _ = preprocess_data(train_data[0], valid_data[0], test_data[0],
                                                          cat_features, num_features, as_array=True)

        # Format data
        X_train = train_data[0]
        X_valid = valid_data[0]
        y_train = convert_to_structured(train_data[1], train_data[2])
        y_valid = convert_to_structured(valid_data[1], valid_data[2])
        
        # Calculate time bins
        time_bins = make_time_bins(train_data[1], event=train_data[2])
        
        # Train model
        if model_name == "cox":
            model = make_cox_model(config)
            model.fit(X_train, y_train)
        elif model_name == "coxboost":
            model = make_coxboost_model(config)
            model.fit(X_train, y_train)
        elif model_name == "rsf":
            model = make_rsf_model(config)
            model.fit(X_train, y_train)
        elif model_name == "mtlr":
            data_train = X_train.copy()
            data_train["time"] = y_train['time']
            data_train["event"] = y_train['event'].astype(int)
            data_valid = X_valid.copy()
            data_valid["time"] = y_valid['time']
            data_valid["event"] = y_valid['event'].astype(int)
            n_features = X_train.shape[1]
            num_time_bins = len(time_bins)
            model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
            model = train_mtlr_model(model, data_train, data_valid, time_bins,
                                     config, random_state=0, reset_model=True, device=device)
        else:
            raise ValueError("Model not found")
        
        # Compute survival function
        if model_name == "cox":
            test_surv_fn = model.predict_survival_function(X_valid)
            surv_preds = np.row_stack([fn(time_bins) for fn in test_surv_fn])
            surv_preds = pd.DataFrame(surv_preds, dtype=np.float64, columns=time_bins.numpy())
        elif model_name == "coxboost":
            test_surv_fn = model.predict_survival_function(X_valid)
            surv_preds = np.row_stack([fn(time_bins) for fn in test_surv_fn])
            surv_preds = pd.DataFrame(surv_preds, dtype=np.float64, columns=time_bins.numpy())
        elif model_name == "rsf":
            test_surv_fn = model.predict_survival_function(X_valid)
            surv_preds = np.row_stack([fn(time_bins) for fn in test_surv_fn])
            surv_preds = pd.DataFrame(surv_preds, dtype=np.float64, columns=time_bins.numpy())
        elif model_name == "mtlr":
            mtlr_valid_data = torch.tensor(data_valid.drop(["time", "event"], axis=1).values,
                                           dtype=torch.float, device=device)
            survival_outputs, _, ensemble_outputs = make_mtlr_prediction(model, mtlr_valid_data, time_bins, config)
            surv_preds = survival_outputs.numpy()
            time_bins_th = torch.cat([torch.tensor([0]).to(time_bins.device), time_bins], 0)
            surv_preds = pd.DataFrame(surv_preds, columns=time_bins_th.numpy())
        else:
            raise ValueError("Model not found")
        
        # Compute CI
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_valid['time'], y_valid['event'],
                                            y_train['time'], y_train['event'])
        ci_results.append(lifelines_eval.concordance()[0])
    
    # Log to wandb
    wandb.log({"val_ci": np.mean(ci_results)})
    
if __name__ == "__main__":
    main()