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

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(input, self.log_weight ** 2, self.bias)
        else:
            return nn.functional.linear(input, self.log_weight ** 2)

def create_representation_positive(inputdim, layers, dropout = 0):
    modules = []
    prevdim = inputdim
    for hidden in layers[:-1]:
        modules.append(PositiveLinear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(nn.Tanh())
        prevdim = hidden
    modules.append(PositiveLinear(prevdim, layers[-1], bias=True))
    return nn.Sequential(*modules)

def create_representation(inputdim, layers, dropout = 0.5):
    modules = []
    prevdim = inputdim
    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(nn.Tanh())
        prevdim = hidden
    return nn.Sequential(*modules)

class NDE(nn.Module):
    def __init__(self, inputdim, layers = [32, 32, 32], layers_surv = [100, 100, 100], 
               dropout = 0., optimizer = "Adam"):
        super(NDE, self).__init__()
        self.input_dim = inputdim
        self.dropout = dropout
        self.optimizer = optimizer
        self.embedding = create_representation(inputdim, layers, self.dropout) 
        self.outcome = create_representation_positive(1 + layers[-1], layers_surv + [1], self.dropout) 

    def forward(self, x, horizon, gradient = False):
        # Go through neural network
        x_embed = self.embedding(x) # Extract unconstrained NN
        time_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
        survival = self.outcome(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1)) # Compute survival
        survival = survival.sigmoid()
        # Compute gradients
        intensity = torch.autograd.grad(survival.sum(), time_outcome, create_graph = True)[0].unsqueeze(1) if gradient else None

        # return 1 - survival, intensity
        return 1 - survival, intensity

    def survival(self, horizon, x):  
        with torch.no_grad():
            horizon = horizon.expand(x.shape[0])
            temp = self.forward(x, horizon)[0]
        return temp.squeeze()

class MultiNDE(nn.Module):
    def __init__(self, inputdim, n_events, layers = [32, 32, 32],
                 layers_surv = [100, 100, 100], dropout = 0., optimizer = "Adam"):
        super(MultiNDE, self).__init__()
        self.input_dim = inputdim
        self.dropout = dropout
        self.optimizer = optimizer
        self.n_events = n_events
        
        # Shared embedding
        self.embedding = create_representation(inputdim, layers, self.dropout)
        
        # Individual outputs  
        self.outcome = nn.ModuleList([create_representation_positive(1 + layers[-1],
                                                                     layers_surv + [1],
                                                                     dropout) for _ in range(n_events)])

    def forward(self, x, horizon, gradient=False):
        # Go through neural network
        x_embed = self.embedding(x) # Extract unconstrained NN
        time_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
        
        survival = [output_layer(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1))
                    for output_layer in self.outcome]

        # survival = self.outcome(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1)) # Compute survival % TODO: Why concat?
        survival = [surv.sigmoid() for surv in survival] # apply sigmoid func
        # Compute gradients
        if gradient:
            intensities = [torch.autograd.grad(surv.sum(), time_outcome, create_graph = True)[0].unsqueeze(1) for surv in survival]
        else:
            intensities = None

        # return 1 - survival, intensity
        return [1 - surv for surv in survival], intensities

    def survival(self, x, horizon):  
        with torch.no_grad():
            horizon = horizon.expand(x.shape[0])
            temp = self.forward(x, horizon)[0]
        return temp

class MensaNDE(nn.Module):
    # with neural density estimators
    def __init__(self, device, n_features, tol,
                 hidden_size=32, hidden_surv = 32, dropout_rate=0, max_iter = 2000):
        super(MensaNDE, self).__init__()
        self.tol = tol
        self.sumo_e = NDE(n_features, layers = [hidden_size, hidden_size, hidden_size],
                          layers_surv = [hidden_surv, hidden_surv, hidden_surv], dropout = dropout_rate)
        self.sumo_c = NDE(n_features, layers = [hidden_size, hidden_size, hidden_size],
                          layers_surv = [hidden_surv, hidden_surv, hidden_surv], dropout = dropout_rate)

    def forward(self, x, t, c, copula, max_iter=2000):
        S_E, density_E = self.sumo_e(x, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()
        
        S_C, density_C = self.sumo_c(x, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )
        
        logL = event_log_density * c + censoring_log_density * (1-c)
        return -torch.sum(logL)

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator

    def survival(self, t, X):
        with torch.no_grad():
            result = self.sumo.survival(X, t)
        return result[0].squeeze()#, result[1].squeeze()