import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import reformat_survival
from utility.loss import mtlr_nll, cox_nll
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split_multi, make_stratified_split_single
from utility.data import MultiEventDataset
from models import CoxPH
from utility.data import dotdict

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def criterion(outputs, targets, log_vars, model, config):
  loss = 0
  for i in range(len(outputs)):
    precision = torch.exp(-log_vars[i])
    cox_loss = cox_nll(outputs[i], 1, 0, targets[i][:,0], targets[i][:,1], model, C1=config.c1)
    loss += precision * cox_loss + log_vars[i]
  return torch.mean(loss)

"""
def criterion_gaussian(outputs, targets, log_vars, model, config):
    loss = 0
    for i in range(len(outputs)):
        precision = torch.exp(-log_vars[i])
        cox_loss = cox_nll(outputs[i], targets[i][:,0], targets[i][:,1], model, C1=config.c1)
        loss += cox_loss
    return torch.mean(loss)
"""

def criterion_gaussian(outputs, targets, log_vars, model, config):
    precision1 = torch.exp(-log_vars[0])
    cox_loss1 = cox_nll(outputs[0], precision1, log_vars[0], targets[0][:,0], targets[0][:,1], model, C1=config.c1)
    
    precision2 = torch.exp(-log_vars[1])
    cox_loss2 = cox_nll(outputs[1], precision2, log_vars[1], targets[1][:,0], targets[1][:,1], model, C1=config.c1)
    
    loss = cox_loss1 + cox_loss2
    
    return loss

def train_multi_model_gaussian(
        model: nn.Module,
        df_train: pd.DataFrame, # Dataframe with shape [x, Y1_T, Y2_T, Y1_E, Y2_E]
        df_valid: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
        
    train_size, n_features = df_train.shape[0], df_train.shape[1]
    val_size = df_valid.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)
    
    x_train_features = df_train.drop(["y1_time", "y2_time", "y1_event", "y2_event"], axis=1).to_numpy()
    x_train_times = df_train[["y1_time", "y2_time"]].to_numpy()
    x_train_events = df_train[["y1_event", "y2_event"]].to_numpy()
    x_val_features = df_valid.drop(["y1_time", "y2_time", "y1_event", "y2_event"], axis=1).to_numpy()
    x_val_times = df_valid[["y1_time", "y2_time"]].to_numpy()
    x_val_events = df_valid[["y1_event", "y2_event"]].to_numpy()
    
    train_dataset = MultiEventDataset(n_features, x_train_features, x_train_times, x_train_events)
    val_dataset = MultiEventDataset(n_features, x_train_features, x_train_times, x_train_events)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=config.batch_size)
    
    start_time = datetime.now()
    loss_list = []
    
    for i in pbar:
        cumulative_loss = 0
        
        # Training
        for X, Y1, Y2 in train_data_loader:
            optimizer.zero_grad()
            
            logits_dists = model(X)
            
            n_samples = config.n_samples_train
            logits_cpd1 = torch.stack([torch.reshape(logits_dists[0].sample(), (X.shape[0], 1)) for _ in range(n_samples)])
            logits_cpd2 = torch.stack([torch.reshape(logits_dists[1].sample(), (X.shape[0], 1)) for _ in range(n_samples)])
            log_var1 = torch.log(torch.mean(torch.var(logits_cpd1, axis=0, keepdims=True)))
            log_var2 = torch.log(torch.mean(torch.var(logits_cpd2, axis=0, keepdims=True)))
            logits_mean1 = torch.mean(logits_cpd1, axis=0)
            logits_mean2 = torch.mean(logits_cpd2, axis=0)
            logits_mean = [logits_dists[0].mean, logits_dists[1].mean]
            
            loss = criterion_gaussian(logits_mean, [Y1, Y2], [log_var1, log_var2], model, config)
            cumulative_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        epoch_loss = cumulative_loss/len(train_data_loader)
        loss_list.append(epoch_loss)
        
        # Validation
        valid_loss = 0
        for X, Y1, Y2 in val_data_loader:
            
            logits_dists = model(X)
            
            n_samples = config.n_samples_test
            logits_cpd1 = torch.stack([torch.reshape(logits_dists[0].sample(), (X.shape[0], 1)) for _ in range(n_samples)])
            logits_cpd2 = torch.stack([torch.reshape(logits_dists[1].sample(), (X.shape[0], 1)) for _ in range(n_samples)])
            log_var1 = torch.log(torch.mean(torch.var(logits_cpd1, axis=0, keepdims=True)))
            log_var2 = torch.log(torch.mean(torch.var(logits_cpd2, axis=0, keepdims=True)))
            logits_mean1 = torch.mean(logits_cpd1, axis=0)
            logits_mean2 = torch.mean(logits_cpd2, axis=0)
            logits = [logits_dists[0].mean, logits_dists[1].mean]
            
            loss = criterion_gaussian(logits, [Y1, Y2], [log_var1, log_var2], model, config)
            valid_loss += loss.item()
        
        total_val_loss = (valid_loss/len(val_data_loader))
        print(f"Train loss: {epoch_loss} - Valid loss: {total_val_loss}")
        
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {loss_list[-1]:.4f}; "
                             f"Validation nll = {total_val_loss:.4f};")
        
        if config.early_stop:
            if best_val_nll > total_val_loss:
                best_val_nll = total_val_loss
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    model.eval()
    model.calculate_baseline_survival(torch.tensor(x_train_features, dtype=torch.float32).to(device),
                                      torch.tensor(x_train_times, dtype=torch.float32).to(device),
                                      torch.tensor(x_train_events, dtype=torch.float32).to(device))
    return model

def train_multi_model(
        model: nn.Module,
        df_train: pd.DataFrame, # Dataframe with shape [x, Y1_T, Y2_T, Y1_E, Y2_E]
        df_valid: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
        
    train_size, n_features = df_train.shape[0], df_train.shape[1]
    val_size = df_valid.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)
    
    x_train_features = df_train.drop(["y1_time", "y2_time", "y1_event", "y2_event"], axis=1).to_numpy()
    x_train_times = df_train[["y1_time", "y2_time"]].to_numpy()
    x_train_events = df_train[["y1_event", "y2_event"]].to_numpy()
    x_val_features = df_valid.drop(["y1_time", "y2_time", "y1_event", "y2_event"], axis=1).to_numpy()
    x_val_times = df_valid[["y1_time", "y2_time"]].to_numpy()
    x_val_events = df_valid[["y1_event", "y2_event"]].to_numpy()
    
    train_dataset = MultiEventDataset(n_features, x_train_features, x_train_times, x_train_events)
    val_dataset = MultiEventDataset(n_features, x_train_features, x_train_times, x_train_events)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=config.batch_size)
    
    start_time = datetime.now()
    loss_list = []
    
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    
    for i in pbar:
        cumulative_loss = 0
        
        # Train
        for X, Y1, Y2 in train_data_loader:
            optimizer.zero_grad()
            
            logits = model(X)
            loss = criterion(logits, [Y1, Y2], [log_var_a, log_var_b], model, config)
            cumulative_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        loss_list.append(cumulative_loss/len(train_data_loader))
        
        # Validate
        valid_loss = 0
        for X, Y1, Y2 in val_data_loader:
            
            logits = model(X)
            loss = criterion(logits, [Y1, Y2], [log_var_a, log_var_b], model, config)
            valid_loss += loss.item()
        
        total_val_loss = (valid_loss/len(val_data_loader))
        print(total_val_loss)
        
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {loss_list[-1]:.4f}; "
                             f"Validation nll = {total_val_loss:.4f};")
        
        if config.early_stop:
            if best_val_nll > total_val_loss:
                best_val_nll = total_val_loss
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    model.eval()
    model.calculate_baseline_survival(torch.tensor(x_train_features, dtype=torch.float32).to(device),
                                      torch.tensor(x_train_times, dtype=torch.float32).to(device),
                                      torch.tensor(x_train_events, dtype=torch.float32).to(device))
    return model, [log_var_a, log_var_b]

def train_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    data_train, _, data_val = make_stratified_split_single(data_train, stratify_colname='both',
                                                           frac_train=0.9, frac_test=0.1,
                                                           random_state=random_state)

    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                 torch.tensor(data_train["time"].values, dtype=torch.float),
                                 torch.tensor(data_train["event"].values, dtype=torch.float))
    x_val, t_val, e_val = (torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                           torch.tensor(data_val["time"].values, dtype=torch.float).to(device),
                           torch.tensor(data_val["event"].values, dtype=torch.float).to(device))

    train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
    model.config.batch_size = train_size

    for i in pbar:
        nll_loss = 0
        for xi, ti, ei in train_loader:
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
    if isinstance(model, CoxPH):
        model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model

def train_mensa(X_train, y_train,
                X_valid, y_valid,
                model: nn.Module,
                time_bins: NumericArrayLike,
                config: dotdict,
                random_state: int,
                reset_model: bool = True,
                device: torch.device = torch.device("cuda")):
    train_size = X_train.shape[0]
    val_size = X_valid.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    
    X_valid, y_valid = X_valid.to(device), y_valid.to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    
    for i in pbar:
        total_loss = 0
        total_log_likelihood = 0
        total_kl_divergence = 0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, yi, train_size)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() / train_size
            total_log_likelihood += log_likelihood.item() / train_size
            total_kl_divergence += (log_variational_posterior.item() -
                                    log_prior.item()) * config.batch_size / train_size**2

        val_loss, _, _, val_log_likelihood = model.sample_elbo(X_valid, y_valid, dataset_size=val_size)
        val_loss /= val_size
        val_log_likelihood /= val_size
        print(val_log_likelihood)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"Train: Total = {total_loss:.4f}, "
                            f"KL = {total_kl_divergence:.4f}, "
                            f"nll = {total_log_likelihood:.4f}; "
                            f"Val: Total = {val_loss.item():.4f}, "
                            f"nll = {val_log_likelihood.item():.4f}; ")
        if config.early_stop:
            if best_val_nll > val_loss:
                best_val_nll = val_loss
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break
            
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    return model
    
    