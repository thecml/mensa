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
from utility.loss import mtlr_nll, cox_nll, cox_nll2
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split_multi, calculate_baseline_hazard
from utility.data import MultiEventDataset

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
        self.l1 = nn.Linear(self.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.l1(x)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = calculate_baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        self.l1.reset_parameters()
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
    
class MultiEventCoxPHGaussian(nn.Module):
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
    
        # Mean parameters
        self.mean_layer1 = nn.Sequential(
            nn.Linear(n_hidden, 1),
        )

        self.mean_layer2 = nn.Sequential(
            nn.Linear(n_hidden, 1),
        )

        # Standard deviation parameters
        self.std_layer1 = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Softplus(),
        )
    
        self.std_layer2 = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Softplus(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        shared = self.shared_layer(x)
        
        # Parametrization of the mean
        mu1 = self.mean_layer1(shared)
        mu2 = self.mean_layer2(shared)
        
        # Parametrization of the standard deviation
        sigma1 = self.std_layer1(shared)
        sigma2 = self.std_layer2(shared)
        
        return [torch.distributions.Normal(mu1, sigma1), torch.distributions.Normal(mu2, sigma2)] # two events
        
    def calculate_baseline_survival(self, x, t, e):
        logits_dists = self.forward(x)
        
        n_samples = self.config.n_samples_test
        logits_cpd1 = torch.stack([torch.reshape(logits_dists[0].sample(), (x.shape[0], 1)) for _ in range(n_samples)])
        logits_cpd2 = torch.stack([torch.reshape(logits_dists[1].sample(), (x.shape[0], 1)) for _ in range(n_samples)])
        logits_mean1 = torch.mean(logits_cpd1, axis=0)
        logits_mean2 = torch.mean(logits_cpd2, axis=0)
        outputs = [logits_dists[0].mean, logits_dists[1].mean]
        
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