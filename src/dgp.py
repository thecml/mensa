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
import torchtuples as tt
from abc import abstractmethod
from utility.data import dotdict
from torchmtlr.model import pack_sequence

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class DGP_LogNormal_linear:
    # Note this is the LogNormal model, not the LogNormal CoxPH model
    def __init__(self, mu: List[float], sigma: List[float], device='cpu', dtype=torch.float64) -> None:
        self.mu_coeff = torch.tensor(mu, device=device).type(dtype)
        self.sigma_coeff = torch.tensor(sigma, device=device).type(dtype)
        self.device = device

    def hazard(self, t, x):
        return self.PDF(t, x) / self.survival(t, x)
    
    def cum_hazard(self, t, x):
        # No closed form solution for the cumulative hazard function of the lognormal distribution
        # we have to approximate it using numerical integration
        n_steps = 1000
        t_grids = torch.linspace(1e-5, t, n_steps).to(self.device)  # avoid zero to prevent division by zero
        delta_u = t / n_steps
        h_values = self.hazard(t_grids, x)
        H_t = torch.sum(h_values * delta_u)
        return H_t

    def survival(self, t, x):
        return 1 - self.CDF(t, x)

    def CDF(self, t, x):
        mu, sigma = self.pred_params(x)
        return 0.5 + 0.5 * torch.erf((LOG(t) - mu) / (sigma * torch.sqrt(torch.tensor(2)).to(self.device)))

    def PDF(self, t, x):
        mu, sigma = self.pred_params(x)
        return (1 / (t * sigma * torch.sqrt(torch.tensor(2 * torch.pi)).to(self.device))) * torch.exp(
            -((LOG(t) - mu) ** 2) / (2 * sigma ** 2))

    def parameters(self):
        return [self.mu_coeff, self.sigma_coeff]

    def pred_params(self, x):
        mu = torch.matmul(x, self.mu_coeff)
        sigma = torch.matmul(x, self.sigma_coeff)
        return mu, sigma

    def rvs(self, x, u):
        mu, sigma = self.pred_params(x)
        return torch.exp(mu + sigma * torch.erfinv(2 * u - 1) * torch.sqrt(torch.tensor(2)).to(self.device))

class DGP_LogNormal_nonlinear(DGP_LogNormal_linear):
    # Note this is the LogNormal nonlinear model
    def __init__(self, n_features, n_hidden, mu: List[float], sigma: List[float], risk_function=torch.nn.ReLU(),
                 device='cpu', dtype=torch.float64) -> None:
        self.beta = torch.rand((n_features, n_hidden), device=device).type(dtype)
        self.mu_coeff = torch.tensor(mu, device=device).type(dtype)
        self.sigma_coeff = torch.tensor(sigma, device=device).type(dtype)
        self.hidden_layer = risk_function
        self.device = device

    def parameters(self):
        return [self.beta, self.mu_coeff, self.sigma_coeff]

    def pred_params(self, x):
        hidden = self.hidden_layer(torch.matmul(x, self.beta))
        mu = torch.matmul(hidden, self.mu_coeff)
        sigma = torch.matmul(hidden, self.sigma_coeff)
        return mu, sigma

class DGP_LogNormalCox_linear:
    def __init__(self, n_features, mu: float, sigma: float, device="cpu", dtype=torch.float64) -> None:
        self.mu = torch.tensor([mu]).type(dtype).to(device)
        self.sigma = torch.tensor([sigma]).type(dtype).to(device)
        self.coeff = torch.rand((n_features,)).to(device)

    def bl_hazard(self, t):
        # The baseline hazard function of the lognormal CoxPH model, here we use the hazard function of the lognormal
        # distribution as the baseline hazard function
        pdf = (1 / (t * self.sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(
            -((LOG(t) - self.mu) ** 2) / (2 * self.sigma ** 2))
        survival = 0.5 - 0.5 * torch.erf((LOG(t) - self.mu) / (self.sigma * torch.sqrt(torch.tensor(2))))
        return pdf / survival

    def hazard(self, t, x):
        return self.bl_hazard(t) * torch.exp(torch.matmul(x, self.coeff))

    def cum_hazard(self, t, x):
        # No closed form solution for the cumulative hazard function of the lognormal distribution
        # we have to approximate it using numerical integration
        n_steps = 1000
        t_grids = torch.linspace(1e-5, t, n_steps)  # avoid zero to prevent division by zero
        delta_u = t / n_steps
        h_values = self.hazard(t_grids, x)
        H_t = torch.sum(h_values * delta_u)
        return H_t

    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))

    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)

    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def parameters(self):
        return [self.mu, self.sigma, self.coeff]

    def rvs(self, x, u):
        # TODO: no closed form solution for the lognormal CoxPH model
        raise NotImplementedError

class DGP_Exp_linear:
    def __init__(self, n_features, baseline_hazard: float, device="cpu", dtype=torch.float64) -> None:
        self.bh = torch.tensor([baseline_hazard]).type(dtype).to(device)
        self.coeff = torch.rand((n_features,)).to(device)
    
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
        
    def parameters(self):
        return [self.bh, self.coeff]
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)

class DGP_EXP_nonlinear(DGP_Exp_linear):
    # This is the exponential CoxPH model with a nonlinear risk function
    def __init__(self, n_features, baseline_hazard: float, n_hidden, risk_function=torch.nn.ReLU(),
                 device='cpu', dtype=torch.float64) -> None:
        self.bh = torch.tensor([baseline_hazard], device=device).type(dtype)
        self.beta = torch.rand((n_features, n_hidden), device=device).type(dtype)
        self.coeff = torch.rand((n_hidden,), device=device).type(dtype)
        self.hidden_layer = risk_function
    
    def hazard(self, t, x):
        risks = torch.matmul(self.hidden_layer(torch.matmul(x, self.beta)), self.coeff)
        return self.bh * torch.exp(risks)

class DGP_Weibull_linear:
    def __init__(self, n_features, alpha: float, gamma: float, device="cpu", dtype=torch.float64):
        self.alpha = torch.tensor([alpha], device=device).type(dtype)
        self.gamma = torch.tensor([gamma], device=device).type(dtype)
        self.coeff = torch.rand((n_features,), device=device).type(dtype)

    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t,x)
    
    def CDF(self ,t ,x):
        return 1 - self.survival(t,x)
    
    def survival(self ,t ,x):
        return torch.exp(-self.cum_hazard(t,x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(torch.matmul(x, self.coeff))
        
    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(torch.matmul(x, self.coeff))

    def parameters(self):
        return [self.alpha, self.gamma, self.coeff]
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(torch.matmul(x, self.coeff)))**(1/self.gamma))*self.alpha

class DGP_Weibull_nonlinear:
    def __init__(self, n_features, n_hidden, alpha: List[float], gamma: List[float],
                 risk_function=torch.nn.ReLU(), device='cpu', dtype=torch.float64):
        self.alpha = torch.tensor(alpha, device=device).type(dtype)
        self.gamma = torch.tensor(gamma, device=device).type(dtype)
        self.beta = torch.rand((n_features, n_hidden), device=device).type(dtype)
        self.hidden_layer = risk_function
        
    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self ,t ,x):    
        return 1 - self.survival(t, x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t, x))
    
    def hazard(self, t, x):
        shape, scale = self.pred_params(x)
        return shape/scale * (t/scale)**(shape-1)

    def cum_hazard(self, t, x):
        shape, scale = self.pred_params(x)
        return (t/scale)**shape

    def parameters(self):
        return [self.alpha, self.gamma, self.beta]

    def pred_params(self, x):
        hidden = self.hidden_layer(torch.matmul(x, self.beta))
        shape = torch.matmul(hidden, self.alpha)
        scale = torch.matmul(hidden, self.gamma)
        return shape, scale
    
    def parameters(self):
        return [self.alpha, self.gamma, self.beta] 
    
    def rvs(self, x, u):
        shape, scale = self.pred_params(x)
        return scale * ((-LOG(u))**(1/shape))