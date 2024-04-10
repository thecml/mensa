import pandas as pd
import numpy as np
import config as cfg
from tools.event_loader import EventDataLoader
from utility.survival import make_stratified_split, convert_to_structured, make_time_bins, make_event_times
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from utility.training import scale_data
from tools.evaluator import LifelinesEvaluator
from tools.models import MultiEventCoxPH, train_multi_model, make_cox_prediction_multi
from tools.preprocessor import Preprocessor
from utility.training import split_and_scale_data
from torch.utils.data import DataLoader, TensorDataset
from pycox.evaluation import EvalSurv
import torch
import random
import warnings

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
    # Load data
    dl_speech = EventDataLoader().load_data(event='Speech')
    dl_walking = EventDataLoader().load_data(event='Walking')
    
    num_features, cat_features = dl_speech.get_features()
    df_speech = dl_speech.get_data()
    df_walking = dl_walking.get_data()
    
    # Split and scale data
    data_train_s, data_valid_s, data_test_s, time_bins_s = split_and_scale_data(df_speech, num_features, cat_features)
    data_train_w, data_valid_w, data_test_w, time_bins_w = split_and_scale_data(df_walking, num_features, cat_features)
    
    # Train model
    config = dotdict(cfg.PARAMS_COX)
    n_features = data_train_s.shape[1] - 2
    model = MultiEventCoxPH(in_features=n_features, config=config)
    model = train_multi_model(model, data_train_s, data_train_w, time_bins_s, config=config,
                              random_state=0, reset_model=True, device=device)
    
    # Evaluate speech event
    x_test = torch.tensor(data_test_s.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
    survival_outputs, time_bins, ensemble_outputs = make_cox_prediction_multi(model, x_test, config=config, event_id=0)
    survival_outputs = pd.DataFrame(survival_outputs, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(survival_outputs.T, data_test_s["time"], data_test_s["event"],
                                        data_train_s['time'], data_train_s['event'])
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ci = lifelines_eval.concordance()[0]
    d_calib = lifelines_eval.d_calibration()[0]
    print(f"Evaluated speech: CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")
    
    # Evaluate walking event
    x_test = torch.tensor(data_test_w.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
    survival_outputs, time_bins, ensemble_outputs = make_cox_prediction_multi(model, x_test, config=config, event_id=1)
    survival_outputs = pd.DataFrame(survival_outputs, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(survival_outputs.T, data_test_w["time"], data_test_w["event"],
                                        data_train_w['time'], data_train_w['event'])
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ci = lifelines_eval.concordance()[0]
    d_calib = lifelines_eval.d_calibration()[0]
    print(f"Evaluated walking: CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")
