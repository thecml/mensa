import torch
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm

from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split

sample_size=30000
torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stop_epochs = 10

def train_dcsurvival_model(model, X_train, X_valid, times_train, events_train,
                           times_valid, events_valid, num_epochs, batch_size,
                           learning_rate, device):
    # Format data
    times_tensor_train = torch.tensor(times_train).to(device)
    event_indicator_tensor_train = torch.tensor(events_train).to(device)
    covariate_tensor_train = torch.tensor(X_train).to(device)

    times_tensor_val = torch.tensor(times_valid).to(device)
    event_indicator_tensor_val = torch.tensor(events_valid).to(device)
    covariate_tensor_val = torch.tensor(X_valid).to(device)

    # Make the model
    optimizer = optim.Adam([{"params": model.sumo_e.parameters(), "lr": learning_rate},
                            {"params": model.sumo_c.parameters(), "lr": learning_rate},
                            {"params": model.phi.parameters(), "lr": learning_rate}])
    
    # Train the model
    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    for epoch in tqdm(range(num_epochs)):
    # for epoch in range(num_epochs):
        optimizer.zero_grad()
        logloss = model(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train, max_iter = 10000)
        (-logloss).backward() 
        optimizer.step()

        if epoch % 10 == 0:
            val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter = 1000)
            if val_loglikelihood > (best_val_loglikelihood + 1):
                best_val_loglikelihood = val_loglikelihood
                epochs_no_improve = 0
            else:
                if val_loglikelihood > best_val_loglikelihood:
                    best_val_loglikelihood = val_loglikelihood
                epochs_no_improve = epochs_no_improve + 10
        # Early stopping condition
        if epochs_no_improve == early_stop_epochs:
            # print('Early stopping triggered at epoch: %s' % epoch)
            break
    return model
