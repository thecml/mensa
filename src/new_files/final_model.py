import torch 
import torch.nn as nn 
import numpy as np 
from torch.nn.parameter import Parameter
from new_files.Copula_final import Nested_Convex_Copula
import wandb

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

class ParamNet(nn.Module):
    def __init__(self, 
                 n_features=10, 
                 n_risk=3, 
                 hidden_layers=[32,32], 
                 activation_func='relu', 
                 dropout =0.5, 
                 residual=True, 
                 bias=True):
        super().__init__()
        self.n_features = n_features
        self.n_risk = n_risk
        self.dropout_val = dropout
        self.residual = residual
        self.layers = nn.ModuleList()
        #I think relu, leaky, selu will be fine
        if activation_func == 'relu':
            self.activation = nn.ReLU()
        elif activation_func == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_func == 'tanh':
            self.activation = nn.Tanh()
        elif activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_func == 'selu':
            self.activation = nn.SELU()
        elif activation_func == 'gelu':
            self.activation = nn.GELU()
        
        else:
            raise NotImplementedError("activation not available!!!!!")
        d_in = n_features
        for h in hidden_layers:
            l = nn.Linear(d_in, h, bias=bias)
            d_in = h
            l.weight.data.fill_(0.01)
            if bias:
                l.bias.data.fill_(0.01)
            self.layers.append(l)
        self.dropout = nn.Dropout(dropout)
        if residual:
            self.last_layer = nn.Linear(hidden_layers[-1] + n_features, n_risk*2)
        else:
            self.last_layer = nn.Linear(hidden_layers[-1], n_risk*2)
        self.last_layer.weight.data.fill_(0.01)
        if bias:
            self.last_layer.bias.data.fill_(0.01)
        
            
    def forward(self, x):
        tmp = x
        for i, l in enumerate(self.layers):
            x = l(x)
            if self.dropout_val > 0:
                if i != (len(self.layers)-1):
                    x = self.dropout(x)
            x = self.activation(x)
        if self.residual:
            x = torch.cat([x, tmp], dim=1)
        x = self.dropout(x)
        p = self.last_layer(x)
        p = torch.exp(p)
        return p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5]

    
def weibull_log_pdf(t, k, lam):
    return safe_log(k) - safe_log(lam) + (k - 1) * safe_log(t/lam) - (t/lam)**k

def weibull_log_cdf(t, k, lam):
    return safe_log(1 - torch.exp(- (t / lam) ** k))

def weibull_log_survival(t, k, lam):
    return - (t / lam) ** k

def triple_loss_(param_net, X, T, E, copula, device='cpu'):
    
    k1, k2, k3, lam1, lam2, lam3 = param_net(X)
    log_pdf1 = weibull_log_pdf(T, k1, lam1)
    log_pdf2 = weibull_log_pdf(T, k2, lam2)
    log_pdf3 = weibull_log_pdf(T, k3, lam3)
    log_surv1 = weibull_log_survival(T, k1, lam1)
    log_surv2 = weibull_log_survival(T, k2, lam2)
    log_surv3 = weibull_log_survival(T, k3, lam3)
    if copula is None:
        p1 = log_pdf1 + log_surv2 + log_surv3
        p2 = log_surv1 + log_pdf2 + log_surv3
        p3 = log_surv1 + log_surv2 + log_pdf3
    else:
        S = torch.cat([torch.exp(log_surv1).reshape(-1,1), torch.exp(log_surv2).reshape(-1,1), torch.exp(log_surv3).reshape(-1,1)], dim=1).clamp(0.002, 0.998)#todo: clamp removed!!!!!!!
        p1 = log_pdf1 + safe_log(copula.conditional_cdf("u", S))
        p2 = log_pdf2 + safe_log(copula.conditional_cdf("v", S))
        p3 = log_pdf3 + safe_log(copula.conditional_cdf("w", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (E == 0) * 1.0
    e2 = (E == 1) * 1.0
    e3 = (E == 2) * 1.0
    loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
    loss = -loss/E.shape[0]
    return loss

def survival(model, k, lam, x, steps=200):
    u = torch.ones((x.shape[0],)) * 0.001
    time_steps = torch.linspace(1e-4, 1, steps=steps).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x, u)
    t_max = t_max_model.reshape(-1,1).repeat(1, steps)
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps))
    surv2 = torch.zeros((x.shape[0], steps))
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i], x)
        surv2[:,i] = torch.exp(weibull_log_survival(time_steps[:,i], k, lam))
    return surv1, surv2, time_steps, t_max_model


class Mensa:
    def __init__(self,  
                n_features, 
                n_risk,
                dropout = 0.8,
                residual = True,
                bias = True,
                hidden_layers = [32,128], 
                activation_func='relu',
                copula = None,
                device = 'cuda'):
        
        self.n_features = n_features
        self.copula = copula
        
        self.n_risks = n_risk
        self.device = device
        self.paramnet = ParamNet(n_features=n_features, 
                        n_risk=n_risk, 
                        hidden_layers=hidden_layers, 
                        activation_func=activation_func, 
                        dropout =dropout, 
                        residual=residual, 
                        bias=bias).to(self.device)
        
    def get_model(self):
        return self.paramnet
    
    def get_copula(self):
        return self.copula
    
    def fit(self, 
            train_dict,
            val_dict,
            batch_size=10000,
            n_epochs=100, 
            copula_grad_multiplier=1.0,
            copula_grad_clip = 1.0,
            model_path='best.pth',
            patience_tresh=100,
            optimizer='adam',
            weight_decay=0.0,
            lr_dict={'network':0.004, 'copula':0.01},
            betas=(0.9,0.999),
            use_wandb=False
            ):
        optim_dict = [{'params': self.paramnet.parameters(), 'lr':lr_dict['network']}]
        if self.copula is not None:
            self.copula.enable_grad()
            optim_dict.append({'params': self.copula.parameters(), 'lr':lr_dict['copula']})
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)
        min_val_loss = 10000
        patience = 0
        best_theta = []
        for itr in range(n_epochs):
            ###################################################
            epoch_loss = 0
            self.paramnet.train()
            X = train_dict['X']
            T = train_dict['T']
            E = train_dict['E']
            idx = torch.randperm(E.shape[0])
            n_batch = int(np.ceil(E.shape[0]/batch_size))
            for i in range(n_batch):
                ###################################################
                idx_start = batch_size * i
                idx_end = min(X.shape[0], (i+1)*batch_size)
                x = X[idx[idx_start:idx_end]].to(self.device)
                t = T[idx[idx_start:idx_end]].to(self.device)
                e = E[idx[idx_start:idx_end]].to(self.device)
                ###################################################
                optimizer.zero_grad()
                loss = triple_loss_(self.paramnet, x, t, e, self.copula)
                epoch_loss += loss.detach().clone().cpu() * x.shape[0]
                loss.backward()
                ###################################################
                if (copula_grad_multiplier) and (self.copula is not None):
                    if isinstance(self.copula, Nested_Convex_Copula):
                        for p in self.copula.parameters()[:-2]:
                            p.grad= (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip,1 *copula_grad_clip)
                    else:
                        for p in self.copula.parameters():
                            p.grad= (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip,1 *copula_grad_clip)
                ###################################################
                optimizer.step()
                if self.copula is not None:
                    if isinstance(self.copula, Nested_Convex_Copula):
                        for p in self.copula.parameters()[-2]:
                                if p < 0.01:
                                    with torch.no_grad():
                                        p[:] = torch.clamp(p, 0.01,100)
                    else:
                        for p in self.copula.parameters():
                                if p < 0.01:
                                    with torch.no_grad():
                                        p[:] = torch.clamp(p, 0.01,100)
                ###################################################
                
            epoch_loss = epoch_loss / X.shape[0]
            self.paramnet.eval()
            with torch.no_grad():
                val_loss = triple_loss_(self.paramnet, val_dict['X'].to(self.device), val_dict['T'].to(self.device), val_dict['E'].to(self.device), self.copula)
                if use_wandb:
                    wandb.log({"val_loss": val_loss})
                if val_loss < min_val_loss + 1e-6:
                    min_val_loss = val_loss
                    patience = 0
                    torch.save(self.paramnet.state_dict(), model_path)
                    if self.copula is not None:
                        best_theta = [p.detach().clone().cpu() for p in self.copula.parameters()]
                        print(best_theta)
                else:
                    patience += 1
                    if patience == patience_tresh:
                        print('early stopping!!!!!!!!!')
                        break
                
            if itr % 100 == 0:
                    if self.copula is not None:
                        print(itr, "/", n_epochs, "train_loss: ", round(epoch_loss.item(),4), "val_loss: ", round(val_loss.item(),4), "min_val_loss: ",round(min_val_loss.item(),4), self.copula)
                    else:
                        print(itr, "/", n_epochs, "train_loss: ", round(epoch_loss.item(),4), "val_loss: ", round(val_loss.item(),4), "min_val_loss: ",round(min_val_loss.item(),4))

        self.paramnet.load_state_dict(torch.load(model_path))
        self.paramnet.eval()
        if self.copula is not None:
            self.copula.set_params(best_theta)
            print(self.copula.parameters())
        
        return self.paramnet.to('cpu'), self.copula

    def fit_EM(self, 
            train_dict,
            val_dict,
            batch_size=10000,
            n_epochs=10000, 
            copula_grad_multiplier=1.0,
            model_path='best.pth',
            patience_tresh=500,
            optimizer='adamw',
            weight_decay=0.001,
            lr_dict={'network':1e-3, 'copula':1e-2},
            ):
        network_optimizer = torch.optim.Adam(self.paramnet.parameters(),lr=lr_dict['network'], weight_decay = weight_decay)
        self.copula.enable_grad()
        copula_optimizer = torch.optim.Adam(self.copula.parameters(), lr=lr_dict['copula'])
        
        
        min_val_loss = 10000
        patience = 0
        for itr in range(n_epochs):
            ###################################################
            epoch_loss = 0
            self.paramnet.train()
            X = train_dict['X']
            T = train_dict['T']
            E = train_dict['E']
            idx = torch.randperm(E.shape[0])
            n_batch = int(np.ceil(E.shape[0]/batch_size))
            for i in range(n_batch):
                ###################################################
                idx_start = batch_size * i
                idx_end = min(X.shape[0], (i+1)*batch_size)
                x = X[idx[idx_start:idx_end]].to(self.device)
                t = T[idx[idx_start:idx_end]].to(self.device)
                e = E[idx[idx_start:idx_end]].to(self.device)
                ###################################################
                self.paramnet.train()
                network_optimizer.zero_grad()
                loss = triple_loss_(self.paramnet, x, t, e, self.copula)
                #epoch_loss += loss.detach().clone().cpu() * x.shape[0]
                loss.backward()
                network_optimizer.step()
                epoch_loss += loss.detach().clone().cpu() * x.shape[0]
            
                ###################################################
            self.paramnet.eval()
            copula_optimizer.zero_grad()
            loss = triple_loss_(self.paramnet, X.to(self.device), T.to(self.device), E.to(self.device), self.copula)
            loss.backward()
            if (copula_grad_multiplier) and (self.copula is not None):
                for p in self.copula.parameters():
                    if p.shape[0]==1:
                        p.grad= p.grad * copula_grad_multiplier
            copula_optimizer.step()
            ###################################################
            if self.copula is not None:
                for p in self.copula.parameters():
                    if p.shape[0]==1:
                        if p < 0.01:
                            with torch.no_grad():
                                p[:] = torch.clamp(p, 0.01,100)
            ###################################################
                
            epoch_loss = epoch_loss / X.shape[0]
            self.paramnet.eval()
            with torch.no_grad():
                val_loss = triple_loss_(self.paramnet, val_dict['X'].to(self.device), val_dict['T'].to(self.device), val_dict['E'].to(self.device), self.copula)
                if val_loss < min_val_loss + 1e-6:
                    min_val_loss = val_loss
                    patience = 0
                    torch.save(self.paramnet.state_dict(), model_path)
                else:
                    patience += 1
                    if patience == patience_tresh:
                        print('early stopping!!!!!!!!!')
                        break
                
            if itr % 100 == 0:
                    print(itr, "/", n_epochs, epoch_loss, val_loss, min_val_loss, self.copula.parameters())
        self.paramnet.load_state_dict(torch.load(model_path))
        self.paramnet.eval()
        print(min_val_loss, triple_loss_(self.paramnet, val_dict['X'].to(self.device), val_dict['T'].to(self.device), val_dict['E'].to(self.device), self.copula))
        return self.paramnet.to('cpu')
    
    def Evaluate_L1(self, dgp1, dgp2, dgp3, test_dict, steps=200):
        x = test_dict['X']
        self.paramnet = self.paramnet.to('cpu')
        with torch.no_grad():
            k1, k2, k3, lam1, lam2, lam3 = self.paramnet(x)

            surv1, surv2, time_steps, t_m = survival(dgp1, k1, lam1, x, steps)
            integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1)), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
            l1_1 =  torch.mean(integ/t_m)

            surv1, surv2, time_steps, t_m = survival(dgp2, k2, lam2, x, steps)
            integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1)), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
            l1_2 =  torch.mean(integ/t_m)

            surv1, surv2, time_steps, t_m = survival(dgp3, k3, lam3, x, steps)
            integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1)), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
            l1_3 =  torch.mean(integ/t_m)

        return l1_1, l1_2, l1_3


        
if __name__ == '__main__':
    t = torch.rand((10,))
    k = torch.rand((10,))
    lam = torch.rand((10,))
    print(weibull_log_cdf(t, k , lam).shape)