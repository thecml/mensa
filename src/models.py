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
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split, calculate_baseline_hazard
from utility.data import MultiEventDataset

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.config = config
        self.log_vars = nn.Parameter(torch.zeros((task_num)), requires_grad=True)
        
        #self.std_1 = torch.exp(self.log_vars[0])**0.5
        #self.std_2 = torch.exp(self.log_vars[1])**0.5
        #print([self.std_1.item(), self.std_2.item()])

    def forward(self, input, targets):
        loss = 0
        outputs = self.model(input)
        for i in range(len(outputs)):
            precision = torch.exp(-self.log_vars[i])
            nnl_loss = cox_nll(outputs[i], targets[i][:,0], targets[i][:,1],
                               self.model, C1=self.config.c1)
            loss += precision * nnl_loss + self.log_vars[i]

        loss = torch.mean(loss)
        return loss, self.log_vars.data.tolist()

class CoxPH(nn.Module):
    """Cox proportional hazard model for individualised survival prediction."""

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
    """Cox proportional hazard model for individualised survival prediction."""

    def __init__(self, in_features, n_hidden=100, n_output=1, config={}):
        super().__init__()
        
        self.config = config
        
        self.time_bins = list()
        self.cum_baseline_hazards = list()
        self.baseline_survivals = list()
        
        self.net1 = nn.Sequential(nn.Linear(in_features, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_output))
        self.net2 = nn.Sequential(nn.Linear(in_features, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return [self.net1(x), self.net2(x)] # two events

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