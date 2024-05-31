import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from utility.evaluator import LifelinesEvaluator
import torch
import random
import warnings
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility.survival import preprocess_data
from utility.data import dotdict
from data_loader import *

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # Load data
    dl = SyntheticDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    data_packages = dl.split_data(train_size=0.7, valid_size=0.5)
    
    n_events = 2
    for event_id in range(n_events):
        train_data = [data_packages[0][0], data_packages[0][1][:,event_id], data_packages[0][2][:,event_id]]
        test_data = [data_packages[1][0], data_packages[1][1][:,event_id], data_packages[1][2][:,event_id]]
        val_data = [data_packages[2][0], data_packages[2][1][:,event_id], data_packages[2][2][:,event_id]]

        # Make event times
        time_bins = make_time_bins(train_data[1], event=train_data[2])

        # Scale data
        train_data[0], val_data[0], test_data[0] = preprocess_data(train_data[0],
                                                                   val_data[0],
                                                                   test_data[0],
                                                                   cat_features,
                                                                   num_features,
                                                                   as_array=True)

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
        config = dotdict(cfg.MTLR_PARAMS)
        n_features = data_train.shape[1] - 2
        num_time_bins = len(time_bins)
        model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
        model = train_mtlr_model(model, data_train, data_valid, time_bins,
                                 config, random_state=0, reset_model=True, device=device)
        
        # Evaluate
        x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
        survival_outputs, _, _ = make_mtlr_prediction(model, x_test, time_bins, config)
        time_bins = torch.cat([torch.tensor([0]).to(time_bins.device), time_bins], 0)
        survival_outputs = pd.DataFrame(survival_outputs, columns=np.array(time_bins))
        lifelines_eval = LifelinesEvaluator(survival_outputs.T, test_data[1].flatten(), test_data[2].flatten(),
                                            train_data[1].flatten(), train_data[2].flatten())
        mae_hinge = lifelines_eval.mae(method="Hinge")
        ci = lifelines_eval.concordance()[0]
        d_calib = lifelines_eval.d_calibration()[0]

        print(f"Done training event {event_id}. CI={round(ci,2)}, MAE={round(mae_hinge,2)}, D-Calib={round(d_calib,2)}")
