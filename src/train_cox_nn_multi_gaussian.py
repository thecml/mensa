import pandas as pd
import numpy as np
import config as cfg
from utility.survival import convert_to_structured, make_time_bins
from utility.survival import make_event_times, calculate_baseline_hazard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from utility.training import scale_data
from utility.evaluator import LifelinesEvaluator
from trainer import train_multi_model_gaussian
from preprocessor import Preprocessor
from utility.training import split_and_scale_data
from torch.utils.data import DataLoader, TensorDataset
from pycox.evaluation import EvalSurv
import torch
import random
import warnings
from get_data import make_synthetic
from preprocess import split_data, scale_data
from scipy.stats import entropy
from models import MultiEventCoxPHGaussian
from inference import make_cox_prediction_multi
from multi_evaluator import MultiEventEvaluator
import math

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    params = cfg.SYNTHETIC_SETTINGS
    n_events = params['num_events']
    raw_data, event_times, labels = make_synthetic(n_events)
    
    if params['discrete'] == False:
        min_time = np.min(event_times[event_times != -1]) 
        max_time = np.max(event_times[event_times != -1]) 
        time_range = max_time - min_time
        bin_size = time_range / params['num_bins']
        
        binned_event_time = np.floor((event_times - min_time) / bin_size)
        binned_event_time[binned_event_time == params['num_bins']] = params['num_bins'] - 1
    dataset = [raw_data, binned_event_time, labels, min_time, max_time]
    
    data_packages = split_data(dataset[0], dataset[1], dataset[2])
    
    train_data = data_packages[0]
    test_data = data_packages[1]
    valid_data = data_packages[2]

    # Make event times
    time_bins = make_time_bins(train_data[1][:,0], event=train_data[2][:,0])

    # Scale data
    train_data[0] = scale_data(train_data[0], norm_mode='standard')
    valid_data[0] = scale_data(valid_data[0], norm_mode='standard')
    test_data[0] = scale_data(test_data[0], norm_mode='standard')
    
    # Train model
    config = dotdict(cfg.PARAMS_COX_MULTI_GAUSSIAN)
    n_features = train_data[0].shape[1]
    model = MultiEventCoxPHGaussian(in_features=n_features, config=config)
    data_train = pd.DataFrame(train_data[0])
    data_train["y1_time"] = pd.Series(train_data[1][:,0])
    data_train["y2_time"] = pd.Series(train_data[1][:,1])
    data_train["y1_event"] = pd.Series(train_data[2][:,0])
    data_train["y2_event"] = pd.Series(train_data[2][:,1])
    data_valid = pd.DataFrame(valid_data[0])
    data_valid["y1_time"] = pd.Series(valid_data[1][:,0])
    data_valid["y2_time"] = pd.Series(valid_data[1][:,1])
    data_valid["y1_event"] = pd.Series(valid_data[2][:,0])
    data_valid["y2_event"] = pd.Series(valid_data[2][:,1])
    data_test = pd.DataFrame(test_data[0])
    data_test["y1_time"] = pd.Series(test_data[1][:,0])
    data_test["y2_time"] = pd.Series(test_data[1][:,1])
    data_test["y1_event"] = pd.Series(test_data[2][:,0])
    data_test["y2_event"] = pd.Series(test_data[2][:,1])
    
    model = train_multi_model_gaussian(model, data_train, data_valid, time_bins, config=config,
                                       random_state=0, reset_model=True, device=device)

    #print(log_vars)
    #print([math.exp(log_var) ** 0.5 for log_var in log_vars])
    
    #std_1 = torch.exp(model.log_vars[0])**0.5
    #std_2 = torch.exp(model.log_vars[1])**0.5
    #rint([std_1.item(), std_2.item()])
    
    # Evaluate event prediction
    evaluator = MultiEventEvaluator(data_test, data_train, model, config, device)
    surv_preds = evaluator.predict_survival_curves_gaussian()
    for event_id in range(n_events):
        ci = evaluator.calculate_ci(surv_preds[event_id], event_id)
        mae = evaluator.calculate_mae(surv_preds[event_id], event_id, method="Hinge")
        print(f"Event {event_id} - CI={round(ci, 2)} - MAE={round(mae, 2)}")