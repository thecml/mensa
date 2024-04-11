import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_stratified_split_multi, convert_to_structured, make_time_bins, make_event_times
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from utility.evaluator import LifelinesEvaluator
from inference import make_cox_prediction
from trainer import train_model
from models import CoxPH
from preprocessor import Preprocessor
from utility.training import split_and_scale_data
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import warnings
from get_data import make_synthetic
from preprocess import split_data, scale_data
from scipy.stats import entropy

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
    raw_data, event_times, labs = make_synthetic(params['num_events'])
    
    if params['discrete'] == False:
        min_time = np.min(event_times[event_times != -1]) 
        max_time = np.max(event_times[event_times != -1]) 
        time_range = max_time - min_time
        bin_size = time_range / params['num_bins']
        
        binned_event_time = np.floor((event_times - min_time) / bin_size)
        binned_event_time[binned_event_time == params['num_bins']] = params['num_bins'] - 1
    
    n_events = params['num_events']
    for event_id in range(n_events):
        dataset = [raw_data, binned_event_time[:,event_id].reshape(-1,1), labs[:,event_id].reshape(-1,1), min_time, max_time] # first event only

        data_packages = split_data(dataset[0], dataset[1], dataset[2])

        train_data = data_packages[0]
        test_data = data_packages[1]
        val_data = data_packages[2]

        # Make event times
        time_bins = make_time_bins(train_data[1], event=train_data[2])

        # Scale data
        train_data[0] = scale_data(train_data[0], norm_mode='standard')
        test_data[0] = scale_data(test_data[0], norm_mode='standard')
        val_data[0] = scale_data(val_data[0], norm_mode='standard')

        # Format data
        data_train = pd.DataFrame(train_data[0])
        data_train["time"] = pd.Series(train_data[1].flatten())
        data_train["event"] = pd.Series(train_data[2].flatten()).astype(int)
        data_valid = pd.DataFrame(val_data[0])
        data_valid["time"] = pd.Series(val_data[1].flatten())
        data_valid["event"] = pd.Series(val_data[2].flatten()).astype(int)
        data_test = pd.DataFrame(test_data[0])
        data_test["time"] = pd.Series(test_data[1].flatten())
        data_test["event"] = pd.Series(test_data[2].flatten()).astype(int)

        # Train model
        config = dotdict(cfg.PARAMS_COX)
        n_features = data_train.shape[1] - 2
        model = CoxPH(in_features=n_features, config=config)
        model = train_model(model, data_train, time_bins, config=config,
                            random_state=0, reset_model=True, device=device)

        # Evaluate
        x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
        survival_outputs, time_bins, ensemble_outputs = make_cox_prediction(model, x_test, config=config)
        survival_outputs = pd.DataFrame(survival_outputs, columns=np.array(time_bins))
        surv_entropy = entropy(survival_outputs, axis=1)
        lifelines_eval = LifelinesEvaluator(survival_outputs.T, test_data[1].flatten(), test_data[2].flatten(),
                                            train_data[1].flatten(), train_data[2].flatten())
        mae_hinge = lifelines_eval.mae(method="Hinge")
        ci = lifelines_eval.concordance()[0]
        d_calib = lifelines_eval.d_calibration()[0]

        print(f"Done training event {event_id}. CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")
