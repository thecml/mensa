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
from dcsurvival.nde import NDE
from mensa.loss import calculate_loss_one_model, calculate_loss_two_models, calculate_loss_three_models

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

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
        # self.sumo = MultiNDE(inputdim=n_features, n_events=2,
        #                      layers = [hidden_size],
        #                      layers_surv = [hidden_surv],
        #                      dropout = dropout_rate)
        # self.sumo_1 = NDE(inputdim=n_features, 
        #                      layers = [hidden_size],
        #                      layers_surv = [hidden_surv],
        #                      dropout = dropout_rate)
        # self.sumo_2 = NDE(inputdim=n_features, 
        #                      layers = [hidden_size],
        #                      layers_surv = [hidden_surv],
        #                      dropout = dropout_rate)        

        
        self.sumo_e = NDE(n_features, layers = [hidden_size, hidden_size, hidden_size],
                          layers_surv = [hidden_surv, hidden_surv, hidden_surv], dropout = dropout_rate)
        self.sumo_c = NDE(n_features, layers = [hidden_size, hidden_size, hidden_size],
                          layers_surv = [hidden_surv, hidden_surv, hidden_surv], dropout = dropout_rate)

    def forward(self, x, t, c, copula, max_iter=2000):
        #S_E, density_E = self.sumo_e(x, t, gradient = True) # S_E = St = Survival marginals
        # surv_marginals, densities = self.sumo(x, t, gradient = True)
        # s1 = surv_marginals[0]
        # s2 = surv_marginals[1]
        # f1 = densities[0]
        # f2 = densities[1]
        # s1, f1 = self.sumo_1(x, t, gradient = True)
        # s2, f2 = self.sumo_2(x, t, gradient = True)
        
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
     
        """
        y, log_densities = list(), list()
        params = ['u', 'v']
        for event_prob, density in zip(surv_marginals, densities): # for each event
            event_prob = event_prob.squeeze()
            S = event_prob.reshape(-1,1).clamp(0.001,0.999)
            event_log_density = torch.log(density).squeeze() + LOG(copula.conditional_cdf(param, S))
            assert (event_prob >= 0.).all() and (
                event_prob <= 1.+1e-10).all(), "t %s, output %s" % (t, event_prob, )
            y.append(event_prob)
            log_densities.append(event_log_density)
        """
        """
        if copula is None:
            p1 = LOG(f1) + LOG(s2)
            p2 = LOG(f2) + LOG(s1)
        else:
            S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
            p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
            p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
        p1[torch.isnan(p1)] = 0
        p2[torch.isnan(p2)] = 0
        #logL = log_densities[0] + c * torch.log(cur1) + log_densities[1] + (1-c) * torch.log(cur2)
        #logL = log_densities[0] * c + log_densities[1] * (1-c)
        return -torch.mean(p1 * c + (1-c)*p2)
        #return torch.sum(logL)
        """

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
            result = self.sumo_1.survival(X, t)
        return result[0].squeeze()#, result[1].squeeze()

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

def train_mensa_model_2_events(train_dict, valid_dict, model1, model2, copula, n_epochs=1000, lr=5e-3):
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
            p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
        
        optimizer.step()
        
        """
        for p in copula.parameters():
            if p <= 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)
        """
        
        with torch.no_grad():
            val_loss = calculate_loss_two_models(model1, model2, valid_dict, copula)
            if itr % 100 == 0:
                print(f"{val_loss} - {copula.theta}")
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                
                """
                best_bh1 = model1.bh.detach().clone()
                best_coeff1 = model1.coeff.detach().clone()
                best_bh2 = model2.bh.detach().clone()
                best_coeff2 = model2.coeff.detach().clone()
                """
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:
                    break
                
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    """
    model1.bh = best_bh1
    model1.coeff = best_coeff1
    model2.bh = best_bh2
    model2.coeff = best_coeff2
    """
    
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