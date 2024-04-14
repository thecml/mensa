import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_multi_model
import torch
import random
import warnings
from models import MultiEventCoxPH
from multi_evaluator import MultiEventEvaluator
from data_loader import SyntheticDataLoader
from utility.survival import scale_data
from utility.data import dotdict

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
    data_packages = dl.split_data()
    n_events = 2
    
    train_data = [data_packages[0][0], data_packages[0][1], data_packages[0][2]]
    test_data = [data_packages[1][0], data_packages[1][1], data_packages[1][2]]
    valid_data = [data_packages[2][0], data_packages[2][1], data_packages[2][2]]

    # Make event times
    time_bins = make_time_bins(train_data[1], event=train_data[2])

    # Scale data
    train_data[0] = scale_data(train_data[0].values, norm_mode='standard')
    test_data[0] = scale_data(test_data[0].values, norm_mode='standard')
    valid_data[0] = scale_data(valid_data[0].values, norm_mode='standard')
    
    # Train model
    config = dotdict(cfg.PARAMS_COX_MULTI)
    n_features = train_data[0].shape[1]
    model = MultiEventCoxPH(in_features=n_features)
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
    
    model, log_vars = train_multi_model(model, data_train, data_valid, time_bins, config=config,
                                        random_state=0, reset_model=True, device=device)

    # Evaluate event prediction
    evaluator = MultiEventEvaluator(data_test, data_train, model, config, device)
    surv_preds = evaluator.predict_survival_curves()
    for event_id in range(n_events):
        ci = evaluator.calculate_ci(surv_preds[event_id], event_id)
        mae = evaluator.calculate_mae(surv_preds[event_id], event_id, method="Hinge")
        print(f"Event {event_id} - CI={round(ci, 2)} - MAE={round(mae, 2)}")
