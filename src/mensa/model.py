import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as nn_utils  # at top of file

import wandb
import numpy as np

from tqdm import trange

from mensa.loss import conditional_weibull_loss, conditional_weibull_loss_multi, safe_log

def _exp_safe(x, max_val=20.0):
    # exp(20) ~ 4.85e8; prevents overflow while keeping gradient flow
    return torch.exp(torch.clamp(x, max=max_val))

def add_event_free_column(T, E, n_events, horizon=None):
    """
    Treat 'event-free' as state 0, just like any other event.
    - If a sample has no observed event among 1..K, set E[:,0]=1.
    - For multi-event times (T.ndim==2): create T[:,0].
      Use `horizon` if given, otherwise use per-row max time across events.
    Shapes in -> out:
      E: [N, K]      -> [N, K+1]
      T: [N] or [N,K] -> [N] (single) or [N, K+1] (multi)
    """
    N = E.size(0)
    device = E.device

    # Build E_ext with extra col 0
    if E.ndim != 2 or E.size(1) != n_events:
        raise ValueError("E must be [N, K] with K=n_events.")
    E_ext = torch.zeros((N, n_events + 1), dtype=E.dtype, device=device)
    E_ext[:, 1:] = E
    no_event = (E.sum(dim=1) == 0)
    E_ext[no_event, 0] = 1  # event-free label

    # Times
    if T.ndim == 1:
        # single-time case: nothing to change (your code uses the same t for all states)
        T_ext = T
    elif T.ndim == 2:
        if T.size(1) == n_events + 1:
            # already has the extra column
            T_ext = T
        elif T.size(1) == n_events:
            # create T[:,0]
            if horizon is not None:
                t0 = torch.full((N,), float(horizon), device=T.device, dtype=T.dtype)
            else:
                # reasonable default: the max observed time across event columns
                t0 = T.max(dim=1).values
            T_ext = torch.zeros((N, n_events + 1), dtype=T.dtype, device=T.device)
            T_ext[:, 0] = t0
            T_ext[:, 1:] = T
        else:
            raise ValueError("T has unexpected width. Expected K or K+1 columns.")
    else:
        raise ValueError("T must be 1D or 2D.")

    return T_ext, E_ext

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
        self.use_shared = use_shared

        if layers is None:
            layers = []
        self.layers = layers

        lastdim = input_dim if len(layers) == 0 else layers[-1]

        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.n_dists * n_states))
        self.scale = nn.Parameter(-torch.ones(self.n_dists * n_states))

        if self.use_shared:
            self.embedding = create_representation(input_dim, layers, dropout_rate, 'ReLU6')
        else:
            self.embeddings = nn.ModuleList([
                create_representation(input_dim, layers, dropout_rate, 'ReLU6') for _ in range(n_states)
            ])

        self.shapeg = nn.ModuleList([nn.Linear(lastdim, self.n_dists, bias=True) for _ in range(n_states)])
        self.scaleg = nn.ModuleList([nn.Linear(lastdim, self.n_dists, bias=True) for _ in range(n_states)])
        self.gate   = nn.ModuleList([nn.Linear(lastdim, self.n_dists, bias=False) for _ in range(n_states)])
        
        adapter_hidden = max(16, lastdim // 2)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lastdim, adapter_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(adapter_hidden, lastdim, bias=True),
            ) for _ in range(n_states)
        ])
        
    def forward(self, x):
        outcomes = []
        n_samples = x.shape[0]

        if self.use_shared:
            xrep_shared = self.embedding(x)

        base_shape = self.shape.view(self.n_states, self.n_dists)
        base_scale = self.scale.view(self.n_states, self.n_dists)

        for i in range(self.n_states):
            xrep = xrep_shared if self.use_shared else self.embeddings[i](x)
            
            xrep = xrep + self.adapters[i](xrep)

            shp_lin = self.shapeg[i](xrep)
            scl_lin = self.scaleg[i](xrep)

            shp_act = self.act(shp_lin)
            scl_act = self.act(scl_lin)

            shape = shp_act + base_shape[i].expand(n_samples, -1)
            scale = scl_act + base_scale[i].expand(n_samples, -1)

            gate_logits = self.gate[i](xrep) / self.temp

            outcomes.append((shape, scale, gate_logits))

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
            betas=(0.9, 0.999), verbose=False):

        optim_dict = [{'params': self.model.parameters(), 'lr': learning_rate}]

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)

        multi_event = True if train_dict['T'].ndim > 1 else False

        if multi_event:
            T_tr, E_tr = add_event_free_column(train_dict['T'], train_dict['E'], n_events=self.n_states-1, horizon=None)
            T_va, E_va = add_event_free_column(valid_dict['T'], valid_dict['E'], n_events=self.n_states-1, horizon=None)
        else:
            _, E_tr = add_event_free_column(train_dict['T'].reshape(-1), train_dict['E'], n_events=self.n_states-1, horizon=None)
            _, E_va = add_event_free_column(valid_dict['T'].reshape(-1), valid_dict['E'], n_events=self.n_states-1, horizon=None)
            T_tr, T_va = train_dict['T'], valid_dict['T']

        train_loader = DataLoader(TensorDataset(train_dict['X'].to(self.device),
                                                T_tr.to(self.device),
                                                E_tr.to(self.device)),
                                batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_dict['X'].to(self.device),
                                                T_va.to(self.device),
                                                E_va.to(self.device)),
                                batch_size=batch_size, shuffle=False)

        if multi_event:
            event_counts = torch.sum(E_tr[:, 0:], dim=0).float()  # shape: [K]
            event_weights = 1.0 / (event_counts + 1e-8)
            event_weights = event_weights / event_weights.sum() * event_counts.shape[0]  # normalize
        else:
            event_weights = None

        self.model.to(self.device)
        min_delta = 0.001
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        pbar = trange(n_epochs, disable=not verbose)

        for itr in pbar:
            self.model.train()
            total_train_loss = 0

            # Training step
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(self.device), ti.to(self.device), ei.to(self.device)
                optimizer.zero_grad()

                params = self.model.forward(xi)

                if multi_event: # TODO: Compute weight inside the batch
                    f, s = self.compute_risks_multi(params, ti)
                    loss = conditional_weibull_loss_multi(f, s, ei, self.model.n_states, event_weights)
                    for trajectory in self.trajectories:
                        loss += self.compute_risk_trajectory(trajectory[0], trajectory[1], ti, ei, params)
                else:
                    f, s = self.compute_risks(params, ti)
                    loss = conditional_weibull_loss(f, s, ei, self.model.n_states)  # TODO: Use weights

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                total_norm = nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                if not torch.isfinite(total_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_valid_loss = 0
            
            # Validation step
            with torch.no_grad():
                for xi, ti, ei in valid_loader:
                    xi, ti, ei = xi.to(self.device), ti.to(self.device), ei.to(self.device)
                    params = self.model.forward(xi)

                    if multi_event:
                        f, s = self.compute_risks_multi(params, ti)
                        loss = conditional_weibull_loss_multi(f, s, ei, self.model.n_states, event_weights)
                        for trajectory in self.trajectories:
                            loss += self.compute_risk_trajectory(trajectory[0], trajectory[1], ti, ei, params)
                    else:
                        f, s = self.compute_risks(params, ti)
                        loss = conditional_weibull_loss(f, s, ei, self.model.n_states)

                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)

            pbar.set_description(f"[Epoch {itr+1:4}/{n_epochs}]")
            pbar.set_postfix_str(f"Training loss = {avg_train_loss:.4f}, "
                                f"Validation loss = {avg_valid_loss:.4f}")

            if avg_valid_loss < best_valid_loss - min_delta:
                best_valid_loss = avg_valid_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at iteration {itr}, best valid loss: {best_valid_loss}")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
        
    def compute_risks(self, params, ti):
        f_risks, s_risks = [], []
        eps = 1e-12
        ti = torch.clamp(ti.reshape(-1,1).expand(-1, self.model.n_dists), min=eps)

        for i in range(self.model.n_states):
            k = params[i][0]; b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])

            ek = _exp_safe(k); eb = _exp_safe(b)
            s = -(torch.pow(eb*ti, ek)) # log S mixture terms before logsumexp
            f = k + b + ((ek - 1.0) * (b + safe_log(ti)))
            f = f + s

            s = torch.logsumexp(s + gate, dim=1)
            f = torch.logsumexp(f + gate, dim=1)

            f_risks.append(f); s_risks.append(s)

        return torch.stack(f_risks, 1), torch.stack(s_risks, 1)
        
    def compute_risk_trajectory(self, i, j, ti, ei, params): 
        # eg: i = 2, j = 0, j happen before i, S_i(T_j)
        t = ti[:,j].reshape(-1,1).expand(-1, self.model.n_dists) #(n, k)
        k = params[i][0]
        b = params[i][1]
        gate = nn.LogSoftmax(dim=1)(params[i][2])
        s = -(torch.pow(torch.exp(b)*t, torch.exp(k)))
        s = (s + gate)
        s = torch.logsumexp(s, dim=1) #log_survival
        condition = torch.logical_and(ei[:, i] == 1, ei[:, j] == 1)
        result = -torch.sum(condition*s) / ei.shape[0]
        return result
    
    def compute_risks_multi(self, params, ti):
        f_risks = []
        s_risks = []
        eps = 1e-12
        ti = torch.clamp(ti, min=eps)

        for i in range(self.model.n_states):
            # t_i: [B, 1] -> expand to [B, n_dists]
            t_i = ti[:, i].reshape(-1, 1).expand(-1, self.model.n_dists)

            k = params[i][0]
            b = params[i][1]
            gate_logits = params[i][2]

            gate = nn.LogSoftmax(dim=1)(gate_logits)

            # Safe exponentials
            ek = _exp_safe(k)
            eb = _exp_safe(b)

            # log-survival component per mixture: s_ik = - ( (exp(b)*t)^exp(k) )
            s_comp = -(torch.pow(eb * t_i, ek))

            # log-density component per mixture (before mixing)
            f_comp = k + b + (ek - 1.0) * (b + safe_log(t_i))
            f_comp = f_comp + s_comp

            # Mix in log-space
            s = torch.logsumexp(s_comp + gate, dim=1)
            f = torch.logsumexp(f_comp + gate, dim=1)

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