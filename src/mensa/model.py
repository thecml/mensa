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
from copula import Nested_Convex_Copula

from utility.model_helper import get_model_best_params, set_model_best_params
from mensa.loss import double_loss, triple_loss
from mensa.utility import weibull_log_survival

MAX_PATIENCE = 100

class Net(nn.Module):
    def __init__(self,  n_features=10, n_events=3, hidden_layers=[32,32], 
                 activation_func='relu', dropout=0.5, residual=True, bias=True):
        super().__init__()
        self.n_features = n_features
        self.n_events = n_events
        self.dropout_val = dropout
        self.residual = residual
        self.layers = nn.ModuleList()
        
        if activation_func == 'relu':
            self.activation = nn.ReLU()
        elif activation_func == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_func == 'tanh':
            self.activation = nn.Tanh()
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_func == 'selu':
            self.activation = nn.SELU()
        elif activation_func == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError("Activation fn not available")
        
        d_in = n_features
        for h in hidden_layers:
            l = nn.Linear(d_in, h, bias=bias)
            d_in = h
            l.weight.data.fill_(0.01)
            if bias:
                l.bias.data.fill_(0.01)
            self.layers.append(l)
        self.dropout = nn.Dropout(dropout)
        if residual:
            self.last_layer = nn.Linear(hidden_layers[-1] + n_features, n_events*2)
        else:
            self.last_layer = nn.Linear(hidden_layers[-1], n_events*2)
        self.last_layer.weight.data.fill_(0.01)
        if bias:
            self.last_layer.bias.data.fill_(0.01)
        
    def forward(self, x):
        tmp = x
        for i, l in enumerate(self.layers):
            x = l(x)
            if self.dropout_val > 0:
                if i != (len(self.layers)-1):
                    x = self.dropout(x)
            x = self.activation(x)
        if self.residual:
            x = torch.cat([x, tmp], dim=1)
        x = self.dropout(x)
        p = self.last_layer(x)
        p = torch.exp(p)
        return tuple(p[:, i] for i in range(p.shape[1]))
        
class MENSA:
    def __init__(self, n_features, n_events, dropout = 0.8, residual = True,
                 bias = True, hidden_layers = [32, 128], activation_func='relu',
                 copula = None, device = 'cuda'):
        
        self.n_features = n_features
        self.copula = copula
        
        self.n_events = n_events
        self.device = device
        self.net = Net(n_features=n_features, 
                       n_events=n_events, 
                       hidden_layers=hidden_layers, 
                       activation_func=activation_func, 
                       dropout =dropout, 
                       residual=residual, 
                       bias=bias).to(self.device)
        
    def get_model(self):
        return self.net
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, batch_size=10000, n_epochs=100, 
            copula_grad_multiplier=1.0, copula_grad_clip = 1.0, model_path=f"{cfg.MODELS_DIR}/mensa.pt",
            patience_tresh=100, optimizer='adamw', weight_decay=1e-4, lr_dict={'network':0.004, 'copula':0.01},
            betas=(0.9, 0.999), use_wandb=False, verbose=False):
        
        optim_dict = [{'params': self.net.parameters(), 'lr': lr_dict['network']}]
        if self.copula is not None:
            self.copula.enable_grad()
            optim_dict.append({'params': self.copula.parameters(), 'lr': lr_dict['copula']})
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)
            
        min_val_loss = 10000
        patience = 0
        best_theta = []
        for itr in range(n_epochs):
            epoch_loss = 0
            self.net.train()
            X = train_dict['X']
            T = train_dict['T']
            E = train_dict['E']
            idx = torch.randperm(E.shape[0])
            n_batch = int(np.ceil(E.shape[0]/batch_size))
            
            for i in range(n_batch):
                idx_start = batch_size * i
                idx_end = min(X.shape[0], (i+1)*batch_size)
                x = X[idx[idx_start:idx_end]].to(self.device)
                t = T[idx[idx_start:idx_end]].to(self.device)
                e = E[idx[idx_start:idx_end]].to(self.device)

                optimizer.zero_grad()
                
                if self.n_events == 2:
                    loss = double_loss(self.net, x, t, e, self.copula)
                elif self.n_events == 3:
                    loss = triple_loss(self.net, x, t, e, self.copula)
                else:
                    raise NotImplementedError()
                
                epoch_loss += loss.detach().clone().cpu()*x.shape[0]
                loss.backward()

                if (copula_grad_multiplier) and (self.copula is not None):
                    if isinstance(self.copula, Nested_Convex_Copula):
                        for p in self.copula.parameters()[:-2]:
                            p.grad= (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip,1 *copula_grad_clip)
                    else:
                        for p in self.copula.parameters():
                            p.grad= (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip,1 *copula_grad_clip)

                optimizer.step()
                if self.copula is not None:
                    if isinstance(self.copula, Nested_Convex_Copula):
                        for p in self.copula.parameters()[-2]:
                            if p < 0.01:
                                with torch.no_grad():
                                    p[:] = torch.clamp(p, 0.01, 100)
                    else:
                        for p in self.copula.parameters():
                            if p < 0.01:
                                with torch.no_grad():
                                    p[:] = torch.clamp(p, 0.01, 100)
                
            epoch_loss = epoch_loss / X.shape[0]
            self.net.eval()
            
            with torch.no_grad():
                if self.n_events == 2:
                    val_loss = double_loss(self.net, valid_dict['X'].to(self.device),
                                           valid_dict['T'].to(self.device),
                                           valid_dict['E'].to(self.device), self.copula)
                elif self.n_events == 3:
                    val_loss = triple_loss(self.net, valid_dict['X'].to(self.device),
                                           valid_dict['T'].to(self.device),
                                           valid_dict['E'].to(self.device), self.copula)
                else:
                    raise NotImplementedError()
                
                if use_wandb:
                    wandb.log({"val_loss": val_loss})
                    
                if val_loss < min_val_loss + 1e-6:
                    min_val_loss = val_loss
                    patience = 0
                    torch.save(self.net.state_dict(), model_path)
                    if self.copula is not None:
                        best_theta = [p.detach().clone().cpu() for p in self.copula.parameters()]
                else:
                    patience += 1
                    if patience == patience_tresh:
                        if verbose:
                            print('Early stopping!')
                        break
                
            if itr % 100 == 0:
                if verbose:
                    if self.copula is not None:
                        print(itr, "/", n_epochs, "train_loss: ", round(epoch_loss.item(),4),
                            "val_loss: ", round(val_loss.item(),4), "min_val_loss: ", round(min_val_loss.item(),4), self.copula.parameters())
                    else:
                        print(itr, "/", n_epochs, "train_loss: ", round(epoch_loss.item(),4),
                            "val_loss: ", round(val_loss.item(),4), "min_val_loss: ", round(min_val_loss.item(),4))

        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        if self.copula is not None:
            self.copula.set_params(best_theta)
        
        return self.net.to('cpu'), self.copula
                    
    def predict(self, x_test, time_bins):
        filename = f"{cfg.MODELS_DIR}/mensa.pt"
        self.net.load_state_dict(torch.load(filename))
        self.net.to(self.device)
        self.net.eval()
        
        if self.n_events == 2:
            k1, k2, lam1, lam2 = self.net(x_test.to(self.device))
            k1 = k1.to(self.device)
            lam1 = lam1.to(self.device)
            k = [k1, k2]
            lam = [lam1, lam2]
        elif self.n_events == 3:
            k1, k2, k3, lam1, lam2, lam3 = self.net(x_test.to(self.device))
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
                    