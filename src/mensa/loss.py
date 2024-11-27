import torch
import itertools

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

def conditional_weibull_loss_multi(f, s, e, n_risks: int):
    loss = 0.0
    for k in range(n_risks):
        temp = (e[:, k] == 1)
        loss += torch.sum(temp*f[:, k].T)
        loss += torch.sum(~temp*s[:, k].T)
    loss = -loss / e.shape[0]
    return loss

def conditional_weibull_loss(f, s, e, n_risks: int):    
    loss = 0.0
    for k in range(n_risks):
        temp = (e == k)
        loss += torch.sum(temp*f[:, k].T)
        loss += torch.sum(~temp*s[:, k].T)        
    loss = -loss / e.shape[0]
    return loss