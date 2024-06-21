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
from utility.data import dotdict
from utility.survival import cox_survival, calculate_baseline_hazard
from sksurv.linear_model.coxph import BreslowEstimator

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def multi_cox_loss_function(outputs, targets, model, config, copula=None):
  loss = 0
  for i in range(len(outputs)):
    cox_loss = cox_nll(outputs[i], 1, 0, targets[i][:,0], targets[i][:,1], model, C1=config.c1)
    loss += cox_loss
  return torch.mean(loss)

def loss_function(model1, model2, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    return -torch.mean(p1 * data['E'] + (1-data['E'])*p2)

def dependent_train_loop_linear(model1, model2, train_data, val_data,
                               n_itr, optimizer1='Adam', lr1=5e-3, verbose=False, copula=None):
    model1.enable_grad()
    model2.enable_grad()
    copula.enable_grad()
    
    min_val_loss = 1000
    
    optimizer = torch.optim.Adam([{"params": model1.parameters(), "lr": 5e-3},
                                {"params": model2.parameters(), "lr": 5e-3},
                                {"params": copula.parameters(), "lr": 5e-3}])
    
    for itr in range(n_itr):
        optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, copula)
        loss.backward()
        for p in copula.parameters():
            p.grad = p.grad * 100
            p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
        
        optimizer.step()
        
        for p in copula.parameters():
            if p <= 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)
        
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, copula)
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr =0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:
                    break
                
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    
    return model1, model2, copula

def independent_train_loop_linear(model1, model2, train_data, val_data,
                                  n_itr, optimizer1='Adam', optimizer2='Adam',
                                  lr1=5e-3, lr2=5e-3, sub_itr=5, verbose=False):
    train_loss_log = []
    val_loss_log = []
    copula_log = torch.zeros((n_itr,))
    model1.enable_grad()
    model2.enable_grad()
    
    copula_grad_log = []
    mu_grad_log = [[], []]
    sigma_grad_log = [[], []]
    coeff_grad_log = [[], []]
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    stop_itr = 0
    if optimizer1 == 'Adam':
        model_optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr1, weight_decay=0.0)
    
    for itr in range(n_itr):
        model_optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, None)
        loss.backward()
        model_optimizer.step() 
        train_loss_log.append(loss.detach().clone())
    ##########################
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, None)
            val_loss_log.append(val_loss.detach().clone())
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr =0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:  
                    break
                
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    return model1, model2


    
def copula_loss_function(event_survival, event_pdf, targets, copula):
    s1 = event_survival[0]
    s2 = event_survival[1]
    f1 = event_pdf[0]
    f2 = event_pdf[1]
    e1 = targets[0][:,1]
    e2 = targets[1][:,1]
    
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
        
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    
    return -torch.mean(p1*e1 + p2*e2 + (1-e1)*p1 + (1-e1)*p2)
    #return -torch.mean(p1 * data['E'] + (1-data['E'])*p2)
"""
def train_mensa_model(
        model: nn.Module,
        df_train: pd.DataFrame, # Dataframe with shape [x, Y1_T, Y2_T, Y1_E, Y2_E]
        df_valid: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda"),
        copula = None
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
        
    train_size, n_features = df_train.shape[0], df_train.shape[1]
    val_size = df_valid.shape[0]
    n_events = 2

    bh = list()
    for i in range(n_events):
        mean_survival_time = df_train.loc[df_train[f'y{i+1}_event'] == 1][f'y{i+1}_time'].mean()
        baseline_hazard = 1. / mean_survival_time
        bh.append(baseline_hazard)
    
    if copula:
        copula.enable_grad()
    
    if copula:
        optimizer = optim.Adam(list(model.parameters()) + [copula.theta], lr=0.001, weight_decay=0.0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.005)

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
    val_dataset = MultiEventDataset(n_features, x_val_features, x_val_times, x_val_events)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=config.batch_size)
    
    start_time = datetime.now()
    loss_list = []
    
    for i in pbar:
        cumulative_loss = 0
        
        # Train
        for X, Y1, Y2 in train_data_loader:
            optimizer.zero_grad()
            targets = [Y1, Y2]
            logits = model(X)
            
            if copula:
                event_survival, event_pdf = list(), list()
                for i in range(n_events):
                    hazard = bh[i] * torch.exp(logits[i].flatten())
                    cum_hazard = hazard * targets[i][:,0]
                    survival = torch.exp(-cum_hazard)
                    event_survival.append(survival)
                    event_pdf.append(survival*hazard)
                
                loss = copula_loss_function(event_survival, event_pdf, targets, copula)
                cumulative_loss += loss.item()
                loss.backward()
                copula.theta.grad = copula.theta.grad * 100
                copula.theta.grad = copula.theta.grad.clamp(-1,1)
                
                if torch.isnan(copula.theta.grad):
                    print(copula.theta)
                    assert 0
                    
                optimizer.step()
                
                if copula.theta <= 0:
                    with torch.no_grad():
                        copula.theta[:] = torch.clamp(copula.theta, 0.001, 30)
            else:
                event_survival, event_pdf = list(), list()
                for i in range(n_events):
                    hazard = bh[i] * torch.exp(logits[i].flatten())
                    cum_hazard = hazard * targets[i][:,0]
                    survival = torch.exp(-cum_hazard)
                    event_survival.append(survival)
                    event_pdf.append(survival*hazard)
                
                loss = copula_loss_function(event_survival, event_pdf, targets, None)
                cumulative_loss += loss.item()
                loss.backward()
                optimizer.step()
            
        loss_list.append(cumulative_loss/len(train_data_loader))
        
        # Validate
        valid_loss = 0
        for X, Y1, Y2 in val_data_loader:
            
            targets = [Y1, Y2]
            logits = model(X)
            
            if copula:
                event_survival, event_pdf = list(), list()
                for i in range(n_events):
                    hazard = bh[i] * torch.exp(logits[i].flatten())
                    cum_hazard = hazard * targets[i][:,0]
                    survival = torch.exp(-cum_hazard)
                    event_survival.append(survival)
                    event_pdf.append(survival*hazard)
                loss = copula_loss_function(event_survival, event_pdf, targets, copula)
                valid_loss += loss.item()
            else:
                event_survival, event_pdf = list(), list()
                for i in range(n_events):
                    hazard = bh[i] * torch.exp(logits[i].flatten())
                    cum_hazard = hazard * targets[i][:,0]
                    survival = torch.exp(-cum_hazard)
                    event_survival.append(survival)
                    event_pdf.append(survival*hazard)
                loss = copula_loss_function(event_survival, event_pdf, targets, None)
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
    return model
"""
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
            loss = multi_cox_loss_function(logits, [Y1, Y2], model, config)
            cumulative_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        loss_list.append(cumulative_loss/len(train_data_loader))
        
        # Validate
        valid_loss = 0
        for X, Y1, Y2 in val_data_loader:
            
            logits = model(X)
            loss = multi_cox_loss_function(logits, [Y1, Y2], model, config)
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
    return model

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