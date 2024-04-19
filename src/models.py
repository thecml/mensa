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
from utility.loss import mtlr_nll, cox_nll
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split_multi, calculate_baseline_hazard
from utility.data import MultiEventDataset
import torchtuples as tt
from abc import abstractmethod
from utility.bnn_distributions import ParametrizedGaussian, ScaleMixtureGaussian, InverseGamma
from utility.data import dotdict
from torchmtlr import pack_sequence

class BayesianBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_parameters(self):
        pass    

    @abstractmethod
    def log_prior(self):
        pass

    @abstractmethod
    def log_variational_posterior(self):
        pass

    def get_name(self):
        return self._get_name()

class BayesianLinear(nn.Module):
    """
    Single linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            config: argparse.Namespace,
            use_mixture: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights
        self.config = config
        if self.config.mu_scale is None:
            self.weight_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        else:
            self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(out_features, in_features)),
                                              -self.config.mu_scale, self.config.mu_scale)

        self.weight_rho = nn.Parameter(torch.ones([out_features, in_features]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * self.config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        if use_mixture:
            pi = config.pi
        else:
            pi = 1
        self.weight_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)
        self.bias_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
            bias = self.bias.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # For a single layer network, x would have 2 dimension [n_data, n_feature]
        # But sometime x would be the sampled output from the previous layer,
        # which will have 3 dimension [n_samples, n_data, n_feature]
        n_data = x.shape[-2]
        bias = bias.repeat(1, n_data, 1)
        # If x is 3-d, this expand command will make x remains the same.
        x = x.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: input features size; k: output size
        return torch.einsum('bij,bkj->bik', x, weight) + bias

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

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
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        shared = self.shared_layer(x)
        return self.fc1(shared)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = calculate_baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        #self.l1.reset_parameters()
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

class Mensa(nn.Module):
    def __init__(self,
                 in_features: int,
                 num_time_bins: int,
                 num_events: int = 1,
                 config: dotdict = dotdict()):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.in_features = in_features
        self.num_time_bins = num_time_bins
        self.num_events = num_events
        
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 
                                 (self.num_time_bins - 1) * self.num_events, config)

        #weight = torch.zeros(self.in_features,
        #                     (self.num_time_bins-1) * self.num_events,
        #                     dtype=torch.float)
        #bias = torch.zeros((self.num_time_bins-1) * self.num_events)
        #self.mtlr_weight = nn.Parameter(weight)
        #self.mtlr_bias = nn.Parameter(bias)

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones((self.num_time_bins - 1),
                           (self.num_time_bins), num_events,
                           requires_grad=True)))
        self.reset_parameters()
        
    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        this_batch_size = x.shape[0] # because the last batch may not be a complete batch.
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)
        outputs = outputs.reshape(n_samples, this_batch_size,
                                  (self.num_time_bins)-1, self.num_events) # this can be deleted, just for the safety

        #[10, 32, 66], len(time_bins) = 33
        # forward only returns (w * x + b) for computing nll loss
        # survival curves will be generated using mtlr_survival() function.
        # return outputs
        #return outputs
        G_with_samples = self.G.expand(n_samples, -1, -1, 2)
        # b: n_samples; i: n_data; j: n_bin - 1; k: n_bin
        return torch.einsum('bijm,bjkm->bikm', outputs, G_with_samples)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(
            self,
            x,
            y,
            dataset_size
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples) #4D
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        nll = 0
        # remark if average is needed or not
        for event_id in range(self.num_events):
            event_var = outputs.var(dim=0)[:,:,event_id]
            total_var = torch.mean(event_var)
            event_output = outputs.mean(dim=0)[:,:,event_id]
            event_target = y[:,:,event_id]
            mtlr_loss = mtlr_nll(event_output, event_target, model=self, C1=0, average=False)
            nll += (1/total_var) * mtlr_loss + torch.log(total_var ** (1/2))
        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"hidden_size={self.hidden_size}), "
                f"num_time_bins={self.num_time_bins})")