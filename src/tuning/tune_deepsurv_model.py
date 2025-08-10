from SurvivalEVAL import LifelinesEvaluator
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import argparse
import numpy as np
import os
import config as cfg
import torch
from sota_models import DeepSurv, make_coxph_model, make_coxboost_model, make_deepsurv_prediction, train_deepsurv_model
from utility.data import get_first_event
from utility.tuning import get_coxboost_sweep_cfg, get_deepsurv_sweep_cfg
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
dtype = torch.float32
torch.set_default_dtype(dtype)

# Setup device
device = "cpu"

def main():
    global dataset_name
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default="seer_se")
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    
    sweep_config = get_deepsurv_sweep_cfg()
    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}')
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    # Initialize a new wandb run
    config_defaults = cfg.DEEPSURV_PARAMS
    wandb.init(config=config_defaults, group=dataset_name, tags=["deepsurv"])
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
    train_dict['E'] = get_first_event(torch.tensor(train_dict['E'], device=device, dtype=torch.int32))
    train_dict['T'] = get_first_event(torch.tensor(train_dict['T'], device=device, dtype=torch.float32))
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    valid_dict['E'] = get_first_event(torch.tensor(valid_dict['E'], device=device, dtype=torch.int32))
    valid_dict['T'] = get_first_event(torch.tensor(valid_dict['T'], device=device, dtype=torch.float32))
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    test_dict['E'] = get_first_event(torch.tensor(test_dict['E'], device=device, dtype=torch.int32))
    test_dict['T'] = get_first_event(torch.tensor(test_dict['T'], device=device, dtype=torch.float32))

    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Train model
    n_features = train_dict['X'].shape[1]
    model = DeepSurv(in_features=n_features, config=config)
    data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
    data_train['time'] = train_dict['T'].cpu().numpy()
    data_train['event'] = train_dict['E'].cpu().numpy()
    data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
    data_valid['time'] = valid_dict['T'].cpu().numpy()
    data_valid['event'] = valid_dict['E'].cpu().numpy()
    model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                 random_state=0, reset_model=True, device=device, dtype=dtype)
    
    # Make predictions
    model_preds, time_bins_deepsurv = make_deepsurv_prediction(model, valid_dict['X'].to(device),
                                                               config=config, dtype=dtype)
    spline = interp1d(time_bins_deepsurv.cpu().numpy(),
                        model_preds.cpu().numpy(),
                        kind='linear', fill_value='extrapolate')
    model_preds = spline(time_bins.cpu().numpy())
    surv_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
    y_train_time = train_dict['T']
    y_train_event = (train_dict['E'])*1.0
    y_valid_time = valid_dict['T']
    y_valid_event = (valid_dict['E'])*1.0
    lifelines_eval = LifelinesEvaluator(surv_preds.T, y_valid_time, y_valid_event,
                                        y_train_time, y_train_event)
    ci = lifelines_eval.concordance()[0]
    
    # Log to wandb
    wandb.log({"c_harrell": ci})

if __name__ == "__main__":
    main()
