import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import numpy as np

from tqdm import trange

from mensa.loss import conditional_weibull_loss, conditional_weibull_loss_multi

def create_representation(inputdim, layers, activation, bias=True):
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
        self.shape = nn.Parameter(-torch.ones(self.k * risks)) #(k * risk)
        self.scale = nn.Parameter(-torch.ones(self.k * risks))

        self.gate = nn.Linear(lastdim, self.k * self.risks, bias=False)
        self.scaleg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.shapeg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.embedding = create_representation(inputdim, layers, 'ReLU6') # ReLU6

    def forward(self, x):
        xrep = self.embedding(x)
        dim = x.shape[0]
        shape = torch.clamp(self.act(self.shapeg(xrep)) + self.shape.expand(dim,-1), min=-10, max=10)
        scale = torch.clamp(self.act(self.scaleg(xrep)) + self.scale.expand(dim,-1), min=-10, max=10)
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
                                               risks=n_events)
        
    def get_model(self):
        return self.model
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, batch_size=1024, n_epochs=20000, 
            copula_grad_multiplier=1.0, copula_grad_clip=1.0,
            patience=100, optimizer='adam', weight_decay=0.005,
            lr_dict={'network': 5e-4, 'copula': 0.005}, betas=(0.9, 0.999),
            use_wandb=False, verbose=False):

        optim_dict = [{'params': self.model.parameters(), 'lr': lr_dict['network']}]
        if self.copula is not None:
            self.copula.enable_grad()
            optim_dict.append({'params': self.copula.parameters(), 'lr': lr_dict['copula']})
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)
        
        multi_event = True if train_dict['T'].ndim > 1 else False
        
        train_loader = DataLoader(TensorDataset(train_dict['X'].to(self.device),
                                                train_dict['T'].to(self.device),
                                                train_dict['E'].to(self.device)),
                                  batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_dict['X'].to(self.device),
                                                valid_dict['T'].to(self.device),
                                                valid_dict['E'].to(self.device)),
                                  batch_size=batch_size, shuffle=False)
        
        self.model.to(self.device)
        min_delta = 0.001
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        
        pbar = trange(n_epochs, disable=not verbose)
        
        for itr in pbar:
            self.model.train()
            total_train_loss = 0
            
            # Training step
            for xi, ti, ei in train_loader:
                optimizer.zero_grad()
                
                params = self.model.forward(xi) # run forward pass
                if multi_event:
                    f, s = self.compute_risks_multi(params, ti, self.model.risks)
                    loss = conditional_weibull_loss_multi(f, s, ei, self.model.risks)
                else:
                    f, s = self.compute_risks(params, ti, self.model.risks)
                    loss = conditional_weibull_loss(f, s, ei, self.model.risks)
                        
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
        
            avg_train_loss = total_train_loss / len(train_loader)

            # Validation step
            self.model.eval()
            total_valid_loss = 0
            
            with torch.no_grad():
                for xi, ti, ei in valid_loader:
                    params = self.model.forward(xi) # run forward pass
                    if multi_event:
                        f, s = self.compute_risks_multi(params, ti, self.model.risks)
                        loss = conditional_weibull_loss_multi(f, s, ei, self.model.risks)
                    else:
                        f, s = self.compute_risks(params, ti, self.model.risks)
                        loss = conditional_weibull_loss(f, s, ei, self.model.risks)
                    
                    total_valid_loss += loss.item()
                
            avg_valid_loss = total_valid_loss / len(valid_loader)
                
            if use_wandb:
                wandb.log({"train_loss": avg_train_loss})
                wandb.log({"valid_loss": avg_valid_loss})
                
            pbar.set_description(f"[Epoch {itr+1:4}/{n_epochs}]")
            pbar.set_postfix_str(f"Training loss = {avg_train_loss:.4f}, "
                                 f"Validation loss = {avg_valid_loss:.4f}")

            # Check for early stopping
            if avg_valid_loss < best_valid_loss - min_delta:
                best_valid_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at iteration {itr}, best valid loss: {best_valid_loss}")
                break
        
    def compute_risks(self, params, ti, n_risks):
        f_risks = []
        s_risks = []
        ti = ti.reshape(-1,1).expand(-1, self.model.k) #(n, k)
        for i in range(self.model.risks):
            k = params[i][0]
            b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])
            s = - (torch.pow(torch.exp(b)*ti, torch.exp(k)))
            f = k + b + ((torch.exp(k)-1)*(b+torch.log(ti)))
            f = f + s
            s = (s + gate)
            s = torch.logsumexp(s, dim=1) #log_survival
            f = (f + gate)
            f = torch.logsumexp(f, dim=1) #log_density
            f_risks.append(f) #(n,3) each column for one risk
            s_risks.append(s)
        f = torch.stack(f_risks, dim=1)
        s = torch.stack(s_risks, dim=1)
        return f, s
    
    def compute_risks_multi(self, params, ti, n_risks):
        f_risks = []
        s_risks = []
        for i in range(self.model.risks):
            t = ti[:,i].reshape(-1,1).expand(-1, self.model.k) #(n, k)
            k = params[i][0]
            b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])
            s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
            f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
            f = f + s
            s = (s + gate)
            s = torch.logsumexp(s, dim=1)#log_survival
            f = (f + gate)
            f = torch.logsumexp(f, dim=1)#log_density
            f_risks.append(f)#(n,3) each column for one risk
            s_risks.append(s)
        f = torch.stack(f_risks, dim=1)
        s = torch.stack(s_risks, dim=1)
        return f, s

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