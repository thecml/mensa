import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
from trainer import train_mensa_model
import torch
import random
import warnings
from models import Mensa
from multi_evaluator import MultiEventEvaluator
from data_loader import *
from utility.survival import preprocess_data
from utility.data import dotdict
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loader
from src.copula import Clayton

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
    n_events = 3
    n_output = 1
    n_hidden = 100
    
    train_data = [data_packages[0][0], data_packages[0][1], data_packages[0][2]]
    test_data = [data_packages[1][0], data_packages[1][1], data_packages[1][2]]
    valid_data = [data_packages[2][0], data_packages[2][1], data_packages[2][2]]

    # Make event times
    time_bins = make_time_bins(train_data[1], event=train_data[2])

    # Scale data
    train_data[0], valid_data[0], test_data[0] = preprocess_data(train_data[0], valid_data[0], test_data[0],
                                                                 cat_features, num_features,
                                                                 as_array=True)
    
    # Format data
    config = dotdict(cfg.COX_MULTI_PARAMS)
    n_features = train_data[0].shape[1]
    model = Mensa(in_features=n_features, n_hidden=n_hidden, n_output=n_output)
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
    
    # Create censoring event
    data_train['y3_time'] = data_train.apply(lambda x: x['y1_time'] if (x['y1_event'] == 0)
                                             & (x['y2_event'] == 0) else 0, axis=1)
    data_train['y3_event'] = data_train.apply(lambda x: 1 if x['y3_time'] != 0 else 0, axis=1)
    data_valid['y3_time'] = data_valid.apply(lambda x: x['y1_time'] if (x['y1_event'] == 0)
                                             & (x['y2_event'] == 0) else 0, axis=1)
    data_valid['y3_event'] = data_valid.apply(lambda x: 1 if x['y3_time'] != 0 else 0, axis=1)
    data_test['y3_time'] = data_test.apply(lambda x: x['y1_time'] if (x['y1_event'] == 0)
                                             & (x['y2_event'] == 0) else 0, axis=1)
    data_test['y3_event'] = data_test.apply(lambda x: 1 if x['y3_time'] != 0 else 0, axis=1)                                         
    
    # Train neural model to learn non-linear network weights.
    model = train_mensa_model(model, data_train, data_valid, time_bins, config=config,
                              random_state=0, reset_model=True, device=device)

    # Evaluate event prediction before copula
    evaluator = MultiEventEvaluator(data_test, data_train, model, config, device)
    surv_preds = evaluator.predict_survival_curves()
    for event_id in range(n_events):
        if event_id == 2:
            break
        ci = evaluator.calculate_ci(surv_preds[event_id], event_id)
        mae = evaluator.calculate_mae(surv_preds[event_id], event_id, method="Hinge")
        print(f"Event {event_id} - CI={round(ci, 2)} - MAE={round(mae, 2)}")
    
    # Freeze the weights and replace the K output layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc1 = nn.Linear(n_hidden, n_output)
    model.fc2 = nn.Linear(n_hidden, n_output)
    
    # Retrain model with copula
    copula = Clayton(torch.tensor([0.1], device=device).type(torch.float32), device=device)
    model = train_mensa_model(model, data_train, data_valid, time_bins, config=config,
                              random_state=0, reset_model=True, device=device, copula=copula)
    
    # Evaluate event prediction after copula
    evaluator = MultiEventEvaluator(data_test, data_train, model, config, device)
    surv_preds = evaluator.predict_survival_curves()
    for event_id in range(n_events):
        ci = evaluator.calculate_ci(surv_preds[event_id], event_id)
        mae = evaluator.calculate_mae(surv_preds[event_id], event_id, method="Hinge")
        print(f"Event {event_id} - CI={round(ci, 2)} - MAE={round(mae, 2)}")