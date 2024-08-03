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
    def __init__(self, nf, n_events, layers,
                 dropout=0.5, activation_fn=nn.ReLU):
        super(Net, self).__init__()
        
        d_in = nf
        self.layers_ = []
        for l in layers:
            d_out = l
            self.layers_.append(torch.nn.Linear(d_in, d_out))
            self.layers_.append(torch.nn.Dropout(dropout))
            self.layers_.append(activation_fn())
            d_in = d_out
            
        self.layers_.append(nn.Linear(d_in, d_in))  # Skip connection
        self.network = nn.Sequential(*self.layers_)
        
        self.out = nn.Linear(d_in + nf, n_events*2)
        
        #Initize weights with a small number
        self._initialize_weights()
        
    def _initialize_weights(self):
        small_value = 0.0001
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -small_value, small_value)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        tmp = self.network(x)
        tmp = torch.cat([tmp, x], dim=1) # Skip connection
        params = self.out(tmp)
        
        exp_params = torch.exp(params)
        
        return tuple(exp_params[:, i] for i in range(exp_params.shape[1]))

class MENSA:
    """
    Implements MENSA model
    """
    def __init__(self, n_features, n_events, layers=[64, 64], dropout=0.25,
                 activation_fn="relu", copula=None, device="cpu",
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
            
        self.model = Net(n_features, self.n_events, layers, dropout, activation_fn).to(device)
            
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
                    