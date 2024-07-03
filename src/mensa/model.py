import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function

import numpy as np

from utility.loss import triple_loss
from dgp import Weibull_linear, Weibull_log_linear, Weibull_nonlinear
from copula import NestedClayton, NestedFrank, ConvexCopula
from copula import Clayton2D, Frank2D, Clayton

from mensa.loss import (calculate_loss_two_models, calculate_loss_three_models, calculate_loss_three_models_me)

class SingleMENSA:
    """
    Implements MENSA model for single event scenario.
    """
    def __init__(self, n_features, distribution='weibull',
                 copula=None, device="cpu", dtype=torch.float64, config=None):
        self.config = config
        self.n_features = n_features
        self.distribution = distribution
        self.copula = copula
        
        self.device = device
        self.dtype = dtype
        
        self.models = []
        for i in range(2): # one model for censoring, one for event
            if self.distribution == "weibull":
                model = Weibull_log_linear(n_features, 2, 1, device, dtype)
                self.models.append(model)
            else:
                raise NotImplementedError()
            
    def get_models(self):
        return self.models
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, n_epochs=1000, lr=0.01):
        for model in self.models:
            model.enable_grad()
            
        if self.copula is not None:
            self.copula.enable_grad()
    
        params = [{"params": model.parameters(), "lr": lr} for model in self.models]
        if self.copula is not None:
            params.append({"params": self.copula.parameters(), "lr": lr})
        optimizer = torch.optim.Adam(params)
        
        min_val_loss = 1000
        for itr in range(n_epochs):
            optimizer.zero_grad()
            loss = calculate_loss_two_models(self.models[0], self.models[1],
                                             train_dict, self.copula)
            loss.backward()
            if self.copula is not None:
                for p in self.copula.parameters():
                    p.grad = p.grad * 100
                    p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
            optimizer.step()
            
            with torch.no_grad():
                val_loss = calculate_loss_two_models(self.models[0], self.models[1],
                                                     valid_dict, self.copula)
                if itr % 100 == 0:
                    if self.copula is not None:
                        print(f"{val_loss} - {self.copula.parameters()}")
                    else:
                        print(f"{val_loss}")
                    
                if not torch.isnan(val_loss) and val_loss < min_val_loss:
                    stop_itr = 0
                    min_val_loss = val_loss.detach().clone()
                else:
                    stop_itr += 1
                    if stop_itr == 2000:
                        break
                    
        return self

class CompetingMENSA:
    """
    Implements MENSA model for competing risks scenario.
    """
    def __init__(self, n_features, n_events, distribution='weibull',
                 copula=None, device="cpu", dtype=torch.float64, config=None):
        self.config = config
        self.n_features = n_features
        self.distribution = distribution
        self.copula = copula
        self.n_events = n_events
        
        self.device = device
        self.dtype = dtype
        
        self.models = []
        for i in range(n_events):
            if self.distribution == "weibull":
                model = Weibull_log_linear(n_features, 2, 1, device, dtype)
                self.models.append(model)
            else:
                raise NotImplementedError()
            
    def get_models(self):
        return self.models
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, n_epochs=1000, lr=0.01):
        for model in self.models:
            model.enable_grad()
            
        if self.copula is not None:
            self.copula.enable_grad()
    
        params = [{"params": model.parameters(), "lr": lr} for model in self.models]
        if self.copula is not None:
            params.append({"params": self.copula.parameters(), "lr": lr})
        optimizer = torch.optim.Adam(params)
        
        min_val_loss = 1000
        for itr in range(n_epochs):
            optimizer.zero_grad()
            
            if self.n_events == 3:
                loss = calculate_loss_three_models(self.models[0], self.models[1],
                                                   self.models[2], train_dict,
                                                   self.copula)
            else:
                raise NotImplementedError()
            
            loss.backward()
            if self.copula is not None:
                for p in self.copula.parameters():
                    p.grad = p.grad * 100
                    p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
            optimizer.step()
            
            with torch.no_grad():
                if self.n_events == 3:
                    val_loss = calculate_loss_three_models(self.models[0], self.models[1],
                                                           self.models[2], valid_dict, self.copula)
                else:
                    raise NotImplementedError()
                
                if itr % 100 == 0:
                    if self.copula is not None:
                        print(f"{val_loss} - {self.copula.parameters()}")
                    else:
                        print(f"{val_loss}")
                    
                if not torch.isnan(val_loss) and val_loss < min_val_loss:
                    stop_itr = 0
                    min_val_loss = val_loss.detach().clone()
                else:
                    stop_itr += 1
                    if stop_itr == 2000:
                        break
                    
        return self

class MultiMENSA:
    """
    Implements MENSA model for multi event scenario.
    """
    def __init__(self, n_features, n_events, distribution='weibull',
                 copula=None, device="cpu", dtype=torch.float64, config=None):
        self.config = config
        self.n_features = n_features
        self.distribution = distribution
        self.copula = copula
        self.n_events = n_events
        
        self.device = device
        self.dtype = dtype
        
        self.models = []
        for i in range(n_events):
            if self.distribution == "weibull":
                model = Weibull_log_linear(n_features, 2, 1, device, dtype)
                self.models.append(model)
            else:
                raise NotImplementedError()
            
    def get_models(self):
        return self.models
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, n_epochs=1000, lr=0.01):
        for model in self.models:
            model.enable_grad()
            
        if self.copula is not None:
            self.copula.enable_grad()
    
        params = [{"params": model.parameters(), "lr": lr} for model in self.models]
        if self.copula is not None:
            params.append({"params": self.copula.parameters(), "lr": lr})
        optimizer = torch.optim.Adam(params)
        
        min_val_loss = 1000
        for itr in range(n_epochs):
            optimizer.zero_grad()
            
            if self.n_events == 3:
                loss = calculate_loss_three_models_me(self.models[0], self.models[1],
                                                      self.models[2], train_dict,
                                                      self.copula)
            else:
                raise NotImplementedError()
            
            loss.backward()
            if self.copula is not None:
                for p in self.copula.parameters():
                    p.grad = p.grad * 100
                    p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
            optimizer.step()
            
            with torch.no_grad():
                if self.n_events == 3:
                    val_loss = calculate_loss_three_models_me(self.models[0], self.models[1],
                                                              self.models[2], valid_dict, self.copula)
                else:
                    raise NotImplementedError()
                
                if itr % 100 == 0:
                    if self.copula is not None:
                        print(f"{val_loss} - {self.copula.parameters()}")
                    else:
                        print(f"{val_loss}")
                    
                if not torch.isnan(val_loss) and val_loss < min_val_loss:
                    stop_itr = 0
                    min_val_loss = val_loss.detach().clone()
                else:
                    stop_itr += 1
                    if stop_itr == 2000:
                        break
                    
        return self