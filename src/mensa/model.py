import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

import wandb

import numpy as np
import config as cfg

from utility.loss import triple_loss
from copula import Nested_Convex_Copula

from mensa.loss import double_loss, triple_loss, conditional_weibull_loss
from mensa.utility import weibull_log_survival

def create_representation(inputdim, layers, activation, bias=False):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=bias))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)

class DeepSurvivalMachinesTorch(torch.nn.Module):
    """"
    k: number of distributions
    temp: 1000 default, temprature for softmax
    discount: not used yet 
    """
    def __init__(self, inputdim, k, layers, temp, risks, discount=1.0):
        super(DeepSurvivalMachinesTorch, self).__init__()

        self.k = k
        self.temp = float(temp)
        self.discount = float(discount)
        
        self.risks = risks

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0: lastdim = inputdim
        else: lastdim = layers[-1]

        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.k *  risks)) #(k * risk)
        self.scale = nn.Parameter(-torch.ones(self.k * risks))

        self.gate = nn.Linear(lastdim, self.k * self.risks, bias=False)
        self.scaleg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.shapeg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.embedding = create_representation(inputdim, layers, 'ReLU6')

    def forward(self, x):
        xrep = self.embedding(x)
        dim = x.shape[0]
        shape = self.act(self.shapeg(xrep))  + self.shape.expand(dim,-1)
        scale = self.act(self.scaleg(xrep)) + self.scale.expand(dim,-1)
        
        gate = self.gate(xrep) / self.temp
        outcomes = []
        for i in range(self.risks):
            outcomes.append((shape[:,i*self.k:(i+1)* self.k], scale[:,i*self.k:(i+1)* self.k], gate[:,i*self.k:(i+1)* self.k]))
            
        return outcomes
        
class MENSA:
    def __init__(self, n_features, n_events, n_dists=5,
                 layers=[32, 32], copula=None, device='cuda'):
        
        self.n_features = n_features
        self.copula = copula
        
        self.n_events = n_events
        self.device = device
        self.model = DeepSurvivalMachinesTorch(n_features, n_dists, layers, 1000,
                                               risks=n_events) # single event
        
    def get_model(self):
        return self.model
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, batch_size=1024, n_epochs=2000, 
            copula_grad_multiplier=1.0, copula_grad_clip=1.0,
            patience=100, optimizer='adam', weight_decay=0.005,
            lr_dict={'network': 5e-4, 'copula': 0.01},
            betas=(0.9, 0.999), use_wandb=False, verbose=False):
        
        optim_dict = [{'params': self.model.parameters(), 'lr': lr_dict['network']}]
        if self.copula is not None:
            self.copula.enable_grad()
            optim_dict.append({'params': self.copula.parameters(), 'lr': lr_dict['copula']})
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)
        
        X = train_dict['X'].to(self.device)
        T = train_dict['T'].to(self.device)
        E = train_dict['E'].to(self.device)
        train_loader = DataLoader(TensorDataset(X, T, E), batch_size=batch_size, shuffle=True)
        
        self.model.to(self.device)
        patience = 100
        min_delta = 0.001
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Training loop with early stopping
        for itr in range(10000):
            self.model.train()
            for xi, ti, ei in train_loader:
                optimizer.zero_grad()
                loss = conditional_weibull_loss(self.model, xi, ti, ei)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_loss = conditional_weibull_loss(self.model, valid_dict['X'].to(self.device),
                                                    valid_dict['T'].to(self.device), valid_dict['E'].to(self.device))
                if use_wandb:
                    wandb.log({"val_loss": val_loss})

            if itr % 100 == 0:
                if verbose:
                    print(f"Iteration {itr}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

            # Check for early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at iteration {itr}")
                break
                    
    def predict(self, x_test, time_bins, risk=0):
        t = list(time_bins.cpu().numpy())
        params = self.model.forward(x_test)
        
        shape, scale, logits = params[risk][0], params[risk][1], params[risk][2]
        k_ = shape
        b_ = scale

        squish = nn.LogSoftmax(dim=1)
        logits = squish(logits)
        
        t_horz = torch.tensor(time_bins).double().to(logits.device)
        t_horz = t_horz.repeat(shape.shape[0], 1)
        
        cdfs = []
        for j in range(len(time_bins)):

            t = t_horz[:, j]
            lcdfs = []

            for g in range(self.model.k):

                k = k_[:, g]
                b = b_[:, g]
                s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
                lcdfs.append(s)

            lcdfs = torch.stack(lcdfs, dim=1)
            lcdfs = lcdfs+logits
            lcdfs = torch.logsumexp(lcdfs, dim=1)
            cdfs.append(lcdfs.detach().cpu().numpy())
        
        return np.exp(np.array(cdfs)).T
                    