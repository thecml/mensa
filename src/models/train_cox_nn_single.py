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
from utility.preprocessor import Preprocessor
from utility.training import split_and_scale_data
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import warnings
from scipy.stats import entropy
from data_loader import SyntheticDataLoader, ALSDataLoader
from utility.survival import preprocess_data
from utility.data import dotdict
from data_loader import get_data_loader

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

dataset_name = "rotterdam"

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    data_packages = dl.split_data()
    
    n_events = 2
    for event_id in range(n_events):
        train_data = [data_packages[0][0], data_packages[0][1][:,event_id], data_packages[0][2][:,event_id]]
        test_data = [data_packages[1][0], data_packages[1][1][:,event_id], data_packages[1][2][:,event_id]]
        val_data = [data_packages[2][0], data_packages[2][1][:,event_id], data_packages[2][2][:,event_id]]

        # Make event times
        time_bins = make_time_bins(train_data[1], event=train_data[2])

        # Scale data
        train_data[0] = preprocess_data(train_data[0].values, norm_mode='standard')
        test_data[0] = preprocess_data(test_data[0].values, norm_mode='standard')
        val_data[0] = preprocess_data(val_data[0].values, norm_mode='standard')

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
        lifelines_eval = LifelinesEvaluator(survival_outputs.T, test_data[1].flatten(), test_data[2].flatten(),
                                            train_data[1].flatten(), train_data[2].flatten())
        mae_hinge = lifelines_eval.mae(method="Hinge")
        ci = lifelines_eval.concordance()[0]
        d_calib = lifelines_eval.d_calibration()[0]

        print(f"Done training event {event_id}. CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")
