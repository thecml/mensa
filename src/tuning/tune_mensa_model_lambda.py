import argparse
from SurvivalEVAL import LifelinesEvaluator
import numpy as np
import os
import config as cfg
import torch
from utility.evaluation import global_C_index, local_C_index  # <- added local_C_index
from utility.tuning import get_mensa_sweep_cfg
from utility.config import load_config
from utility.survival import make_time_bins, preprocess_data
from data_loader import get_data_loader
from mensa.model import MENSA
import warnings
import random
import pandas as pd

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 10
PROJECT_NAME = "mensa"

# Setup precision
dtype = torch.float32
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global dataset_name
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="rotterdam_me")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    
    sweep_config = get_mensa_sweep_cfg()
    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}')
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    # Initialize a new wandb run
    config_defaults = cfg.MENSA_PARAMS
    wandb.init(config=config_defaults, group=dataset_name, tags=["mensa"])
    config = wandb.config
    
    # Load and split data (fixed seed=0)
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    n_events = dl.n_events
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=0)
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    train_dict['E'] = torch.tensor(train_dict['E'], device=device, dtype=torch.int32)
    train_dict['T'] = torch.tensor(train_dict['T'], device=device, dtype=torch.float32)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    valid_dict['E'] = torch.tensor(valid_dict['E'], device=device, dtype=torch.int32)
    valid_dict['T'] = torch.tensor(valid_dict['T'], device=device, dtype=torch.float32)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    test_dict['E'] = torch.tensor(test_dict['E'], device=device, dtype=torch.int32)
    test_dict['T'] = torch.tensor(test_dict['T'], device=device, dtype=torch.float32)
    n_features = train_dict['X'].shape[1]

    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    # Train model
    n_epochs = config['n_epochs']
    n_dists = config['n_dists']
    lr = config['lr']
    batch_size = config['batch_size']
    layers = config['layers']
    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']
    traj_lambda = config['traj_lambda']
    model = MENSA(n_features, layers=layers, dropout_rate=dropout_rate,
                  n_events=n_events, n_dists=n_dists, device=device)
    model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
              weight_decay=weight_decay, patience=20, batch_size=batch_size,
              traj_lambda=traj_lambda, verbose=False)
    
    # Make predictions for all events on validation set
    all_preds = []
    event_metrics = []
    for ev in range(n_events):
        preds = model.predict(valid_dict['X'].to(device), time_bins, risk=ev+1)
        df_pred = pd.DataFrame(preds, columns=time_bins.cpu().numpy())
        all_preds.append(df_pred)
        y_train_time = train_dict['T'][:, ev]
        y_train_event = train_dict['E'][:, ev]
        y_valid_time = valid_dict['T'][:, ev]
        y_valid_event = valid_dict['E'][:, ev]
        lifelines_eval = LifelinesEvaluator(df_pred.T, y_valid_time, y_valid_event,
                                            y_train_time, y_train_event)
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        mae = lifelines_eval.mae(method="Margin")
        d_calib = lifelines_eval.d_calibration()[0]
        event_metrics.append((ci, ibs, mae, d_calib))
    
    # Average metrics across events
    ci_avg = float(np.mean([m[0] for m in event_metrics]))
    ibs_avg = float(np.mean([m[1] for m in event_metrics]))
    mae_avg = float(np.mean([m[2] for m in event_metrics]))
    dcal_avg = float(np.mean([m[3] for m in event_metrics]))
    
    # Global / Local CI across events
    all_preds_arr = [df.to_numpy() for df in all_preds]
    global_ci = float(global_C_index(all_preds_arr, valid_dict['T'].cpu().numpy(),
                                     valid_dict['E'].cpu().numpy()))
    local_ci = float(local_C_index(all_preds_arr, valid_dict['T'].cpu().numpy(),
                                   valid_dict['E'].cpu().numpy()))
    
    # Log to wandb
    wandb.log({
        "ci": ci_avg,
        "ibs": ibs_avg,
        "mae": mae_avg,
        "d_calib": dcal_avg,
        "global_ci": global_ci,
        "local_ci": local_ci
    })
    
if __name__ == "__main__":
    main()
