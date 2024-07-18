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
from distributions import (Weibull_linear, Weibull_nonlinear, Weibull_log_linear, Exp_linear,
                           EXP_nonlinear, LogNormal_linear, LogNormal_nonlinear, LogNormalCox_linear)
from copula import NestedClayton, NestedFrank, ConvexCopula
from copula import Clayton2D, Frank2D, Clayton

from mensa.loss import (calculate_loss_two_models, calculate_loss_three_models, calculate_loss_three_models_me)
from utility.model_helper import get_model_best_params, set_model_best_params

def make_mensa_model_2_events(n_features, start_theta, eps, device, dtype):
    model1 = Weibull_log_linear(n_features, 2, 1, device, dtype)
    model2 = Weibull_log_linear(n_features, 2, 1, device, dtype)
    copula = Clayton2D(torch.tensor([start_theta], dtype=dtype), device, dtype)
    return model1, model2, copula

def make_mensa_model_3_events(n_features, start_theta, eps, device, dtype):
    # Create models and copula
    #copula = Clayton.Clayton3(torch.tensor([2.0], dtype=dtype), eps, dtype, device)
    #copula = NestedFrank(torch.tensor([copula_start_point]),
    #                     torch.tensor([copula_start_point]), eps, eps, device, dtype)
    #c1 = NestedFrank(torch.tensor([2.0]), torch.tensor([2.0]), 1e-4, 1e-4, device, dtype)
    #copula = NestedClayton(torch.tensor([2.0]), torch.tensor([2.0]), 1e-4, 1e-4, device, dtype)
    #copula = ConvexCopula(c1, c2, beta=10000, device=device, dtype=dtype)
    model1 = Weibull_log_linear(n_features, 2, 1, device, dtype)
    model2 = Weibull_log_linear(n_features, 2, 1, device, dtype)
    model3 = Weibull_log_linear(n_features, 2, 1, device, dtype)
    copula = NestedClayton(torch.tensor([start_theta]), torch.tensor([start_theta]), eps, eps, device, dtype)
    return model1, model2, model3, copula

def train_mensa_model_2_events(train_dict, valid_dict, model1, model2, copula, n_epochs=1000, lr=5e-3, model_type='Weibull_log_linear', device = 'cpu'):
    model1.enable_grad()
    model2.enable_grad()
    copula.enable_grad()
    
    optimizer = torch.optim.Adam([{"params": model1.parameters(), "lr": lr},
                                  {"params": model2.parameters(), "lr": lr},
                                  {"params": copula.parameters(), "lr": lr}])
    
    min_val_loss = 1000
    for itr in range(n_epochs):
        optimizer.zero_grad()
        loss = calculate_loss_two_models(model1, model2, train_dict, copula)
        loss.backward()
        
        for p in copula.parameters():
            p.grad = p.grad * 100
            p.grad.clamp_(torch.tensor([-0.5], device=device), torch.tensor([0.5], device=device))
        
        optimizer.step()
        
        for p in copula.parameters():
            if p <= 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)
        
        with torch.no_grad():
            val_loss = calculate_loss_two_models(model1, model2, valid_dict, copula)
            if itr % 100 == 0:
                print(f"{val_loss} - {copula.theta}")
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                best_params = get_model_best_params([model1, model2], model_type)
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:
                    break
                
    [model1, model2] = set_model_best_params([model1, model2], best_params, model_type)
    return model1, model2, copula

def train_mensa_model_3_events(train_dict, valid_dict, model1, model2, model3, copula, n_epochs=1000, lr=5e-3):
    # Run training loop
    model1.enable_grad()
    model2.enable_grad()
    model3.enable_grad()
    copula.enable_grad()
    optimizer = torch.optim.Adam([{"params": model1.parameters(), "lr": lr},
                                  {"params": model2.parameters(), "lr": lr},
                                  {"params": model3.parameters(), "lr": lr},
                                  {"params": copula.parameters(), "lr": lr}])
    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = triple_loss(model1, model2, model3, train_dict, copula)
        loss.backward()
        
        for p in copula.parameters():
            p.grad = p.grad * 100
            p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
            
        #copula.theta.grad = copula.theta.grad*1000
        #play with the clip range to see if it makes any differences 
        #copula.theta.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
        
        optimizer.step()
        
        if i % 500 == 0:
                print(loss, copula.parameters())
    
    return model1, model2, model3, copula

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
                model = Weibull_nonlinear(n_features, n_hidden=4, device=device)
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
                    