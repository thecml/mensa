import numpy as np
import torch
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
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split_multi, calculate_baseline_hazard
from utility.data import MultiEventDataset
import torchtuples as tt
from abc import abstractmethod
from utility.bnn_distributions import ParametrizedGaussian, ScaleMixtureGaussian, InverseGamma
from utility.data import dotdict
from torchmtlr import pack_sequence

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class Exp_linear:
    def __init__(self, bh, nf) -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32)
        self.coeff = torch.rand((nf,))
    
    def hazard(self, t, x):
        return self.bh * torch.exp(torch.matmul(x, self.coeff))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)
    
    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)
    
class EXP_nonlinear:
    def __init__(self, bh, nf, risk_function) -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32)
        self.coeff = torch.rand((nf,))
        self.risk_function = risk_function
    
    def hazard(self, t, x):
        return self.bh * torch.exp(self.risk_function(x, self.coeff ))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)

    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)


class Weibull_linear:
    def __init__(self, n_features, alpha, gamma, beta, device):
        self.n_features = n_features
        self.alpha = torch.tensor([alpha], device=device).type(torch.float32)
        self.gamma = torch.tensor([gamma], device=device).type(torch.float32)
        self.beta = torch.tensor(beta, device=device).type(torch.float32)

    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t,x)
    
    def CDF(self ,t ,x):   
        return 1 - self.survival(t,x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t,x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(torch.matmul(x, self.beta))
        
    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(torch.matmul(x, self.beta))
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(torch.matmul(x, self.beta)))**(1/self.gamma))*self.alpha

class Weibull_nonlinear:
    def __init__(self, n_features, alpha, gamma, beta, risk_function, device):
        self.n_features = n_features
        self.alpha = torch.tensor([alpha],device=device).type(torch.float32)
        self.gamma = torch.tensor([gamma], device=device).type(torch.float32)
        self.beta = torch.tensor(beta, device=device).type(torch.float32)
        self.risk_function = risk_function
        
    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self ,t ,x):    
        return 1 - self.survival(t, x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t, x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(self.risk_function(x, self.beta))

    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(self.risk_function(x, self.beta))
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(self.risk_function(x, self.beta)))**(1/self.gamma))*self.alpha

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

class CoxPH(nn.Module):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        n_hidden=100
        
        # Shared parameters
        self.hidden = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
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

class MultiEventCoxPH(nn.Module):
    def __init__(self, in_features, n_hidden=100, n_output=1, config={}):
        super().__init__()
        
        self.config = config
        
        self.time_bins = list()
        self.cum_baseline_hazards = list()
        self.baseline_survivals = list()
        
        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.fc3 = nn.Linear(n_hidden, n_output)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        shared = self.shared_layer(x)
        
        # Output for event 1 and 2
        out1 = self.fc2(shared)
        out2 = self.fc3(shared)
        
        return [out1, out2] # two events

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        for i in range(len(outputs)):
            time_bins, cum_baseline_hazard, baseline_survival = calculate_baseline_hazard(outputs[i], t[:,i], e[:,i])
            self.time_bins.append(time_bins)
            self.cum_baseline_hazards.append(cum_baseline_hazard)
            self.baseline_survivals.append(baseline_survival)
            
    def reset_parameters(self):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()
    
class Mensa(nn.Module):
    def __init__(self, in_features, n_hidden=100, n_output=1, config={}):
        super().__init__()
        
        self.config = config
        
        self.time_bins = list()
        self.cum_baseline_hazards = list()
        self.baseline_survivals = list()
        
        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(n_hidden, n_output)
        self.fc2 = nn.Linear(n_hidden, n_output)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        shared = self.shared_layer(x)
        
        # Output for event 1 and 2
        out1 = self.fc1(shared)
        out2 = self.fc2(shared)
        
        return [out1, out2] # two events

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        for i in range(len(outputs)):
            time_bins, cum_baseline_hazard, baseline_survival = calculate_baseline_hazard(outputs[i], t[:,i], e[:,i])
            self.time_bins.append(time_bins)
            self.cum_baseline_hazards.append(cum_baseline_hazard)
            self.baseline_survivals.append(baseline_survival)
            
    def reset_parameters(self):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()