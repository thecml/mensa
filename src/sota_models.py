from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import DeepHitSingle
import torchtuples as tt
from pycox.models import DeepHit
from auton_survival.models.dsm import DeepSurvivalMachines
import torch
import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.loss import cox_nll
from utility.survival import cox_survival, calculate_baseline_hazard

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out
    
class DeepSurv(nn.Module):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        
        n_hidden = self.config['hidden_size']
        dropout = self.config['dropout']
        
        # Shared parameters
        self.hidden = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc1 = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        hidden = self.hidden(x)
        return self.fc1(hidden)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_azhard, self.baseline_survival = calculate_baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()

def make_coxph_model(config):
    alpha = config['alpha']
    n_iter = config['n_iter']
    tol = config['tol']
    model = CoxPHSurvivalAnalysis(n_iter=n_iter, tol=tol, alpha=alpha)
    return model

def make_coxboost_model(config):
    n_estimators = config['n_estimators']
    learning_rate = config['learning_rate']
    max_depth = config['max_depth']
    loss = config['loss']
    min_samples_split = config['min_samples_split']
    min_samples_leaf = config['min_samples_leaf']
    max_features = config['max_features']
    dropout_rate = config['dropout_rate']
    subsample = config['subsample']
    model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            loss=loss,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            dropout_rate=dropout_rate,
                                            subsample=subsample,
                                            random_state=0)
    return model
    
def make_dsm_model(config):
    k = config['k']
    layers = config['network_layers']
    return DeepSurvivalMachines(k=k, layers=layers)
    
def make_rsf_model(config):
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    min_samples_split = config['min_samples_split']
    min_samples_leaf =  config['min_samples_leaf']
    max_features = config['max_features']
    model = RandomSurvivalForest(random_state=0,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)
    return model

def make_deephit_cr(config, in_features, out_features, num_risks, duration_index):
    num_nodes_shared = config['num_nodes_shared']
    num_nodes_indiv = config['num_nodes_indiv']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                           out_features, batch_norm, dropout)
    optimizer = tt.optim.AdamWR(lr=config['lr'],
                                decoupled_weight_decay=config['weight_decay'],
                                cycle_eta_multiplier=config['eta_multiplier'])
    model = DeepHit(net, optimizer, alpha=config['alpha'], sigma=config['sigma'],
                    duration_index=duration_index)
    return model

def train_deepsurv_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_valid: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float64
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    train_size = data_train.shape[0]
    val_size = data_valid.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=dtype),
                                 torch.tensor(data_train["time"].values, dtype=dtype),
                                 torch.tensor(data_train["event"].values, dtype=dtype))
    x_val, t_val, e_val = (torch.tensor(data_valid.drop(["time", "event"], axis=1).values, dtype=dtype).to(device),
                           torch.tensor(data_valid["time"].values, dtype=dtype).to(device),
                           torch.tensor(data_valid["event"].values, dtype=dtype).to(device))

    train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
    model.config.batch_size = train_size

    for i in pbar:
        nll_loss = 0
        for xi, ti, ei in train_loader:
            if ei.sum() == 0:
                continue
            xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            nll_loss = cox_nll(y_pred, 1, 0, ti, ei, model, C1=config.c1)

            nll_loss.backward()
            optimizer.step()
            # here should have only one iteration
        logits_outputs = model.forward(x_val)
        eval_nll = cox_nll(logits_outputs, 1, 0, t_val, e_val, model, C1=0)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss.item():.4f}; "
                                f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model

def make_deepsurv_prediction(
        model: DeepSurv,
        x: torch.Tensor,
        config: argparse.Namespace,
        dtype: torch.dtype
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        if config.verbose:
            print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = cox_survival(model.baseline_survival, pred, dtype)
        survival_curves = survival_curves.squeeze()

    time_bins = model.time_bins
    return survival_curves, time_bins
    
def make_deephit_single(in_features, out_features, time_bins, device, config):
    num_nodes = config['num_nodes_shared']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    labtrans = DeepHitSingle.label_transform(time_bins)
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes,
                                  out_features=labtrans.out_features, batch_norm=batch_norm,
                                  dropout=dropout)
    model = DeepHitSingle(net, tt.optim.Adam, device=device, alpha=0.2, sigma=0.1,
                          duration_index=labtrans.cuts)
    model.label_transform = labtrans
    return model
    
def make_deephit_multi(config, in_features, out_features, num_risks, duration_index):
    num_nodes_shared = config['num_nodes_shared']
    num_nodes_indiv = config['num_nodes_indiv']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                           out_features, batch_norm, dropout)
    optimizer = tt.optim.AdamWR(lr=config['lr'],
                                decoupled_weight_decay=config['weight_decay'],
                                cycle_eta_multiplier=config['eta_multiplier'])
    model = DeepHit(net, optimizer, alpha=config['alpha'], sigma=config['sigma'],
                    duration_index=duration_index)

def train_deephit_model(model, x_train, y_train, valid_data, config):
    epochs = config['epochs']
    batch_size = config['batch_size']
    verbose = config['verbose']
    if config['early_stop']:
        callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
    else:
        callbacks = []
    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=valid_data)
    return model