from SurvivalEVAL import LifelinesEvaluator
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import argparse
import numpy as np
import os
import config as cfg
import torch
from hierarchical import util
from hierarchical.helper import format_hierarchical_hyperparams
from sota_models import DeepSurv, make_coxph_model, make_coxboost_model, make_deephit_single, make_deepsurv_prediction, train_deephit_model, train_deepsurv_model
from utility.data import calculate_layer_size_hierarch, format_data_deephit_single, format_hierarchical_data_cr, format_hierarchical_data_me
from utility.tuning import get_coxboost_sweep_cfg, get_deephit_sweep_cfg, get_deepsurv_sweep_cfg, get_hierarch_sweep_cfg
from utility.config import load_config
from utility.survival import convert_to_structured, make_time_bins, preprocess_data
from data_loader import get_data_loader
from mensa.model import MENSA
import warnings
import random
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = cfg.N_RUNS
PROJECT_NAME = "mensa"

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = "cpu"

def main():
    global dataset_name
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default="mimic_me")
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    
    sweep_config = get_hierarch_sweep_cfg()
    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}')
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    # Initialize a new wandb run
    config_defaults = cfg.HIERARCH_PARAMS
    wandb.init(config=config_defaults, group=dataset_name, tags=["hierarch"])
    config = wandb.config
    
    # Load and split data
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    n_events = dl.n_events
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=0)

    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
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

    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Train model
    n_samples = train_dict['X'].shape[0]
    dataset_config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name}.yaml")
    n_time_bins = len(time_bins)
    if dataset_name == "seer_cr":
        train_data, valid_data, test_data = format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                                                        n_time_bins, n_events, censoring_event=False)
    else:
        train_data, valid_data, test_data = format_hierarchical_data_me(train_dict, valid_dict, test_dict, n_time_bins)
    dataset_config['min_time'] = int(train_data[1].min())
    dataset_config['max_time'] = int(train_data[1].max())
    dataset_config['num_bins'] = n_time_bins
    params = config
    params['n_batches'] = int(n_samples/params['batch_size'])
    layer_size = params['layer_size_fine_bins'][0][0]
    params['layer_size_fine_bins'] = calculate_layer_size_hierarch(layer_size, n_time_bins)
    hyperparams = format_hierarchical_hyperparams(params)
    verbose = params['verbose']
    model = util.get_model_and_output("hierarch_full", train_data, test_data,
                                      valid_data, dataset_config, hyperparams, verbose)
    
    # Make predictions
    if dataset_name == "seer_cr":
        event_preds = util.get_surv_curves(torch.tensor(valid_data[0], dtype=dtype), model)
        bin_locations = np.linspace(0, dataset_config['max_time'], event_preds[0].shape[1])
        all_preds = []
        for i in range(n_events):
            preds = pd.DataFrame(event_preds[i], columns=bin_locations)
            all_preds.append(preds)
    else:
        event_preds = util.get_surv_curves(torch.tensor(valid_data[0], dtype=dtype), model)
        bin_locations = np.linspace(0, dataset_config['max_time'], event_preds[0].shape[1])
        all_preds = []
        for i in range(n_events):
            preds = pd.DataFrame(event_preds[i], columns=bin_locations)
            all_preds.append(preds)
    
    if dataset_name == "seer_cr":
        surv_preds = pd.DataFrame(all_preds[1], columns=bin_locations)
        y_train_time = train_dict['T']
        y_train_event = train_dict['E']
        y_valid_time = valid_dict['T']
        y_valid_event = valid_dict['E']
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_valid_time, y_valid_event,
                                            y_train_time, y_train_event)
        ci = lifelines_eval.concordance()[0]
    else:
        surv_preds = pd.DataFrame(all_preds[1], columns=bin_locations)
        y_train_time = train_dict['T'][:,0]
        y_train_event = train_dict['E'][:,0]
        y_valid_time = valid_dict['T'][:,0]
        y_valid_event = valid_dict['E'][:,0]
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_valid_time, y_valid_event,
                                            y_train_time, y_train_event)
        ci = lifelines_eval.concordance()[0]  
    
    # Log to wandb
    wandb.log({"c_harrell": ci})

if __name__ == "__main__":
    main()
