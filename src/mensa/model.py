import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import numpy as np

from tqdm import trange

from mensa.loss import conditional_weibull_loss, conditional_weibull_loss_multi, safe_log

def add_transient_state(data_dict):
    # Modify 'E': Add 1 if all columns are 0, else 0
    condition_e = (data_dict['E'] == 0).all(dim=1).unsqueeze(1)
    new_column_e = condition_e.long()
    data_dict['E'] = torch.cat([new_column_e, data_dict['E']], dim=1)
    
    # Modify 'T': Add the maximum or minimum value based on 'E'
    max_values = data_dict['T'].max(dim=1, keepdim=True).values
    
    # If all 'E' columns are 0, take the maximum; otherwise, take the minimum of active events
    new_column_t = torch.where(
        condition_e,  # Condition: all 'E' columns are 0
        max_values,   # If true, take maximum
        torch.where(data_dict['E'][:, 1:] == 1, data_dict['T'], float('inf')).min(dim=1, keepdim=True).values
    )
    
    data_dict['T'] = torch.cat([new_column_t, data_dict['T']], dim=1)
    
    return data_dict

def create_representation(input_dim, layers, dropout_rate, activation, bias=True):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = input_dim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=bias))
        modules.append(nn.BatchNorm1d(hidden))
        modules.append(act)
        modules.append(nn.Dropout(p=dropout_rate))
        prevdim = hidden

    return nn.Sequential(*modules)

class MLP(torch.nn.Module):
    """"
    input_dim: the input dimension, i.e., number of features.
    num_dists: number of Weibull distributions.
    layers: layers and size of the network, e.g., [32, 32].
    temp: 1000 default, temperature for softmax function.
    num_events: number of events (K).
    discount: not used yet.
    """
    def __init__(self, input_dim, n_dists, layers, dropout_rate,
                 temp, n_states, use_shared=True, discount=1.0):
        super(MLP, self).__init__()

        self.n_dists = n_dists
        self.temp = float(temp)
        self.discount = float(discount)
        
        self.n_states = n_states

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0: lastdim = input_dim
        else: lastdim = layers[-1]

        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.n_dists * n_states))
        self.scale = nn.Parameter(-torch.ones(self.n_dists * n_states))

        self.gate = nn.Linear(lastdim, self.n_dists * self.n_states, bias=False)
        self.scaleg = nn.Linear(lastdim, self.n_dists * self.n_states, bias=True)
        self.shapeg = nn.Linear(lastdim, self.n_dists * self.n_states, bias=True)
        
        self.use_shared = use_shared
        
        if self.use_shared:
            self.embedding = create_representation(input_dim, layers, dropout_rate, 'ReLU6')
        else:
            self.embeddings = nn.ModuleList([
                create_representation(input_dim, layers, 'ReLU6') for _ in range(n_states)
            ])

    def forward(self, x):
        outcomes = []
        if self.use_shared:
            xrep = self.embedding(x)
            dim = x.shape[0]
            shape = torch.clamp(self.act(self.shapeg(xrep)) + self.shape.expand(dim, -1), min=-10, max=10)
            scale = torch.clamp(self.act(self.scaleg(xrep)) + self.scale.expand(dim, -1), min=-10, max=10)
            gate = self.gate(xrep) / self.temp
            for i in range(self.n_states):
                outcomes.append((shape[:,i*self.n_dists:(i+1)*self.n_dists],
                                scale[:,i*self.n_dists:(i+1)*self.n_dists],
                                gate[:,i*self.n_dists:(i+1)*self.n_dists]))
        else:
            dim = x.shape[0]
            for i in range(self.n_states):
                xrep = self.embeddings[i](x)
                shape = torch.clamp(self.act(self.shapeg(xrep)) + self.shape.expand(dim, -1), min=-10, max=10)
                scale = torch.clamp(self.act(self.scaleg(xrep)) + self.scale.expand(dim, -1), min=-10, max=10)
                gate = self.gate(xrep) / self.temp
                outcomes.append((shape[:, i * self.n_dists:(i + 1) * self.n_dists],
                                    scale[:, i * self.n_dists:(i + 1) * self.n_dists],
                                    gate[:, i * self.n_dists:(i + 1) * self.n_dists]))
        return outcomes
        
class MENSA:
    """
    This is a wrapper class for the actual model that implements a convenient fit() function.
    n_features: number of features
    n_events: number of events (K)
    n_dists: number of Weibull distributions
    layers: layers and size of the network, e.g., [32, 32].
    device: device to use, e.g., cpu or cuda
    """
    def __init__(self, n_features, n_events, n_dists=5,
                 layers=[32, 32], dropout_rate=0.5,
                 use_shared=True, trajectories=[], device='cpu'):
        self.n_features = n_features
        self.n_states = n_events + 1 # K + 1 states
        self.device = device
        
        self.use_shared = use_shared
        self.trajectories = trajectories
        
        self.model = MLP(n_features, n_dists, layers, dropout_rate, temp=1000,
                         n_states=self.n_states, use_shared=use_shared)
        
    def get_model(self):
        return self.model
    
    def fit(self, train_dict, valid_dict, batch_size=1024, n_epochs=20000, 
            patience=100, optimizer='adam', weight_decay=0.001, learning_rate=5e-4,
            betas=(0.9, 0.999), use_wandb=False, verbose=False):

        optim_dict = [{'params': self.model.parameters(), 'lr': learning_rate}]
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)
        
        multi_event = True if train_dict['T'].ndim > 1 else False
        
        # Added transient state for multi-event scenarios
        if multi_event:
            train_dict = add_transient_state(train_dict)
            valid_dict = add_transient_state(valid_dict)
        
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
                    f, s = self.compute_risks_multi(params, ti)
                    loss = conditional_weibull_loss_multi(f, s, ei, self.model.n_states)
                    for trajectory in self.trajectories:
                        loss += self.compute_risk_trajectory(trajectory[0], trajectory[1], ti, ei, params) 
                else:
                    f, s = self.compute_risks(params, ti)
                    loss = conditional_weibull_loss(f, s, ei, self.model.n_states)

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
                        f, s = self.compute_risks_multi(params, ti)
                        loss = conditional_weibull_loss_multi(f, s, ei, self.model.n_states)
                        for trajectory in self.trajectories:
                            loss += self.compute_risk_trajectory(trajectory[0], trajectory[1], ti, ei, params) 
                    else:
                        f, s = self.compute_risks(params, ti)
                        loss = conditional_weibull_loss(f, s, ei, self.model.n_states)
                    
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
        
    def compute_risks(self, params, ti):
        f_risks = []
        s_risks = []
        ti = ti.reshape(-1,1).expand(-1, self.model.n_dists)
        for i in range(self.model.n_states):
            k = params[i][0]
            b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])
            s = - (torch.pow(torch.exp(b)*ti, torch.exp(k)))
            f = k + b + ((torch.exp(k)-1)*(b+safe_log(ti)))
            f = f + s
            s = (s + gate)
            s = torch.logsumexp(s, dim=1)
            f = (f + gate)
            f = torch.logsumexp(f, dim=1)
            f_risks.append(f)
            s_risks.append(s)
        f = torch.stack(f_risks, dim=1)
        s = torch.stack(s_risks, dim=1)
        return f, s
    
    def compute_risk_trajectory(self, i, j, ti, ei, params): 
        # eg: i = 2, j = 0, j happen before i, S_i(T_j)
        t = ti[:,j].reshape(-1,1).expand(-1, self.model.n_dists) #(n, k)
        k = params[i][0]
        b = params[i][1]
        gate = nn.LogSoftmax(dim=1)(params[i][2])
        s = -(torch.pow(torch.exp(b)*t, torch.exp(k)))
        s = (s + gate)
        s = torch.logsumexp(s, dim=1)#log_survival
        condition = torch.logical_and(ei[:, i] == 1, ei[:, j] == 1)
        result = -torch.sum(condition*s) / ei.shape[0]
        return result
    
    def compute_risks_multi(self, params, ti):
        f_risks = []
        s_risks = []
        for i in range(self.model.n_states):
            t = ti[:,i].reshape(-1,1).expand(-1, self.model.n_dists)
            k = params[i][0]
            b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])
            s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
            f = k + b + ((torch.exp(k)-1)*(b+safe_log(t)))
            f = f + s
            s = (s + gate)
            s = torch.logsumexp(s, dim=1)
            f = (f + gate)
            f = torch.logsumexp(f, dim=1)
            f_risks.append(f)
            s_risks.append(s)
        f = torch.stack(f_risks, dim=1)
        s = torch.stack(s_risks, dim=1)
        return f, s

    def predict(self, x_test, time_bins, risk=0):
        """
        Courtesy of https://github.com/autonlab/DeepSurvivalMachines
        """
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

            for g in range(self.model.n_dists):

                k = k_[:, g]
                b = b_[:, g]
                s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
                lcdfs.append(s)

            lcdfs = torch.stack(lcdfs, dim=1)
            lcdfs = lcdfs+logits
            lcdfs = torch.logsumexp(lcdfs, dim=1)
            cdfs.append(lcdfs.detach().cpu().numpy())
        
        return np.exp(np.array(cdfs)).T