import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function

import wandb

import numpy as np
import config as cfg

from utility.loss import triple_loss
from distributions import (Weibull_linear, Weibull_nonlinear, Weibull_log_linear, Exp_linear,
                           EXP_nonlinear, LogNormal_linear, LogNormal_nonlinear, LogNormalCox_linear)
from copula import NestedClayton, NestedFrank, ConvexCopula
from copula import Clayton2D, Frank2D, Clayton

from utility.model_helper import get_model_best_params, set_model_best_params
from mensa.loss import double_loss, triple_loss
from mensa.utility import *

from torch.utils.data import TensorDataset, DataLoader

MAX_PATIENCE = 100

"""
From: https://github.com/aligharari96/mensa_mine/blob/main/new_model.py
"""
class Net(torch.nn.Module):
    def __init__(self, nf, n_events, shared_layers, event_layers,
                 dropout=0.5, activation_fn=nn.ReLU):
        super(Net, self).__init__()
        
        d_in = nf
        self.layers_ = []
        for l in shared_layers:
            d_out = l
            self.layers_.append(torch.nn.Linear(d_in, d_out))
            self.layers_.append(torch.nn.Dropout(dropout))
            self.layers_.append(activation_fn())
            d_in = d_out
            
        self.layers_.append(nn.Linear(d_in, d_in)) # Skip connection
        self.network = torch.nn.Sequential(*self.layers_)
        
        # Creating an event and an output layer for each event
        self.event_layers = nn.ModuleList()
        for _ in range(n_events):
            event_specific_layers = []
            d_in_event = d_in + nf  # Initial input dimension for event-specific layers
            for size in event_layers:
                event_specific_layers.append(nn.Linear(d_in_event, size))
                event_specific_layers.append(activation_fn())
                event_specific_layers.append(nn.Dropout(dropout))
                d_in_event = size
            self.event_layers.append(nn.Sequential(*event_specific_layers))
        
        self.out_layers = nn.ModuleList([nn.Linear(event_layers[-1], 2) for _ in range(n_events)])
        
    def forward(self, x):
        tmp = self.network(x)
        
        tmp = torch.cat([tmp, x], dim=1)
        
        event_params = []
        for inter_layer, out_layer in zip(self.event_layers, self.out_layers):
            intermediate_output = inter_layer(tmp)
            intermediate_output = torch.relu(intermediate_output) # use relu
            params = out_layer(intermediate_output)
            exp_params = torch.exp(params)
            event_params.append(exp_params)
        
        flat_params = [param for event in event_params for param in event.permute(1, 0)]
        
        return tuple(flat_params)

"""
class Net3(torch.nn.Module):
    def __init__(self, nf, layers, dropout):
        super().__init__()
        
        d_in = nf
        self.layers_ = []
        for l in layers:
            d_out = l
            self.layers_.append(torch.nn.Linear(d_in, d_out))
            self.layers_.append(torch.nn.Dropout(dropout))
            self.layers_.append(torch.nn.ReLU())
            d_in = d_out
        self.out = torch.nn.Linear(d_in + nf, 6)
        self.layers_.append(torch.nn.Dropout(dropout))
        
        self.network = torch.nn.Sequential(*self.layers_)
    
    def forward(self, x):

        tmp = self.network(x)
        tmp = torch.cat([tmp, x], dim=1)
        params = self.out(tmp)
        k1 = torch.exp(params[:,0])
        k2 = torch.exp(params[:,1])
        k3 = torch.exp(params[:,2])
        
        lam1 = torch.exp(params[:,3])
        lam2 = torch.exp(params[:,4])
        lam3 = torch.exp(params[:,5])
        return k1, k2, k3, lam1, lam2, lam3
"""

class MENSA:
    """
    Implements MENSA model
    """
    def __init__(self, n_features, n_events, shared_layers=[64, 64], event_layers=[32],
                 dropout=0.25, activation_fn="relu", copula=None, device="cpu",
                 dtype=torch.float64, config=None):
        self.config = config
        self.n_features = n_features
        self.copula = copula
        self.n_events = n_events
        
        self.device = device
        self.dtype = dtype
        
        self.train_loss, self.valid_loss = list(), list()
        self.thetas = list()
        
        if activation_fn == "relu":
            activation_fn = nn.ReLU
        elif activation_fn == "leakyrelu":
            activation_fn = nn.LeakyReLU
        elif activation_fn == "elu":
            activation_fn = nn.ELU
        else:
            raise NotImplementedError("Not supported activation fn")
            
        self.model = Net(n_features, self.n_events, shared_layers, event_layers, dropout, activation_fn).to(device)
            
    def get_model(self):
        return self.model
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, n_epochs=10000, lr=0.01, batch_size=32, use_wandb=False):
        if self.copula is not None:
            self.copula.enable_grad()

        train_dataset = TensorDataset(train_dict['X'], train_dict['T'], train_dict['E'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
        params = [{"params": self.model.parameters(), "lr": lr, "weight_decay":1e-4}]
        if self.copula is not None:
            params.append({"params": self.copula.parameters(), "lr": lr, "weight_decay":1e-4})
            
        optimizer = torch.optim.Adam(params)
        
        min_val_loss = 1000
        patience = MAX_PATIENCE
        for itr in range(n_epochs):
            self.model.train()
            batch_loss = list()
            for batch in train_dataloader:
                optimizer.zero_grad()
                X = batch[0].to(self.device)
                T = batch[1].to(self.device)
                E = batch[2].to(self.device)
                
                if self.n_events == 2:
                    loss = double_loss(self.model, X, T, E, self.copula, self.device)
                elif self.n_events == 3:
                    loss = triple_loss(self.model, X, T, E, self.copula, self.device)
                else:
                    raise NotImplementedError()
                
                #batch_loss.append(float(loss.cpu().detach().numpy()))
            
                loss.backward()
                optimizer.step()
                
                if self.copula is not None:
                    for p in self.copula.parameters()[:-2]: #TODO: Add [:-2] using nested cop
                        if p < 0.01:
                            with torch.no_grad():
                                p[:] = torch.clamp(p, 0.01, 100)
            
            #self.train_loss.append(np.mean(batch_loss))
            if self.copula is not None:
                self.thetas.append(tuple([float(tensor.cpu().detach().numpy())
                                        for tensor in self.copula.parameters()[:-2]]))
                
            with torch.no_grad():
                self.model.eval()
                
                if self.n_events == 2:
                    val_loss = double_loss(self.model, valid_dict['X'].to(self.device),
                                           valid_dict['T'].to(self.device), valid_dict['E'].to(self.device),
                                           self.copula, self.device).detach().clone().cpu().numpy()
                elif self.n_events == 3:
                    val_loss = triple_loss(self.model, valid_dict['X'].to(self.device),
                                           valid_dict['T'].to(self.device), valid_dict['E'].to(self.device),
                                           self.copula, self.device).detach().clone().cpu().numpy()
                else:
                    raise NotImplementedError()
                
                self.valid_loss.append(val_loss)
                
                if use_wandb:
                    pass
                    #theta = float(self.copula.parameters()[0].cpu().detach().numpy())
                    #wandb.log({"val_loss": val_loss, "theta": theta})
                    wandb.log({"val_loss": val_loss})
                
                if val_loss  < min_val_loss:
                    min_val_loss = val_loss
                    filename = f"{cfg.MODELS_DIR}/mensa.pt"
                    torch.save(self.model.state_dict(), filename)
                    patience = MAX_PATIENCE
                else:
                    patience = patience - 1
                
            if patience == 0:
                print("Early stopping...")
                break
            
            if (itr % 100 == 0):
                if self.copula is not None:
                    if self.n_events == 2:
                        pass
                        #print(f"{min_val_loss} - {self.copula.parameters()}")
                    else:
                        pass
                        #print(f"{min_val_loss} - {self.copula.parameters()}")
                else:
                    pass
                    #print(f"{min_val_loss}")
                    
    def predict(self, x_test, time_bins):
        filename = f"{cfg.MODELS_DIR}/mensa.pt"
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        
        if self.n_events == 2:
            k1, k2, lam1, lam2 = self.model(x_test.to(self.device))
            k1 = k1.to(self.device)
            lam1 = lam1.to(self.device)
            k = [k1, k2]
            lam = [lam1, lam2]
        elif self.n_events == 3:
            k1, k2, k3, lam1, lam2, lam3 = self.model(x_test.to(self.device))
            k1 = k1.to(self.device)
            lam1 = lam1.to(self.device)
            k2 = k2.to(self.device)
            lam2 = lam2.to(self.device)
            k3 = k3.to(self.device)
            lam3 = lam3.to(self.device)
            k = [k1, k2, k3]
            lam = [lam1, lam2, lam3]
        else:
            raise NotImplementedError()
    
        surv_estimates = list()
        for k_i, lam_i in zip(k, lam):
            surv_estimate = torch.zeros((x_test.shape[0], time_bins.shape[0]), device=self.device)
            time_bins = torch.tensor(time_bins, device=self.device)
            for i in range(time_bins.shape[0]):
                surv_estimate[:,i] = torch.exp(weibull_log_survival(time_bins[i], k_i, lam_i))
            surv_estimates.append(surv_estimate)
            
        return surv_estimates
                    