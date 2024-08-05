import pandas as pd
import numpy as np
import config as cfg
import torch
import random
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from torch.utils.data import DataLoader, TensorDataset

# Local
from data_loader import SingleEventSyntheticDataLoader
from utility.survival import (make_time_bins, convert_to_structured,
                              compute_l1_difference, predict_survival_function)
from utility.config import load_config

from dsm.my_dsm import DeepSurvivalMachinesTorch
from dsm.utilities import conditional_weibull_loss
from mensa.utility import weibull_log_survival
from utility.data import dotdict
from sota_models import DeepSurv, train_deepsurv_model, make_deepsurv_prediction
from scipy.interpolate import interp1d
from data_loader import get_data_loader
from utility.survival import preprocess_data

from mensa.model import MENSA

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Load and split data
    dataset_name = "seer_se"
    dl = get_data_loader(dataset_name)
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=0)
    
    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    n_events = dl.n_events
    n_features = train_dict['X'].shape[1]
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    n_samples = train_dict['X'].shape[0]
    
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    # Train
    model = MENSA(n_features, n_events=2, copula=None, device=device)
    model.fit(train_dict, valid_dict, verbose=True)
    model_preds = model.predict(test_dict, time_bins)
    
    # Evaluate
    surv_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
    y_train_time = train_dict['T']
    y_train_event = (train_dict['E'])*1.0
    y_test_time = test_dict['T']
    y_test_event = (test_dict['E'])*1.0
    lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                        y_train_time, y_train_event)
    
    ci = lifelines_eval.concordance()[0]
    ibs = lifelines_eval.integrated_brier_score()
    mae_hinge = lifelines_eval.mae(method="Hinge")
    mae_margin = lifelines_eval.mae(method="Margin")
    mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
    d_calib = lifelines_eval.d_calibration()[0]
    
    metrics = [ci, ibs, mae_hinge, mae_margin, mae_pseudo, d_calib]
    print(metrics)