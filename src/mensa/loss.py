import torch
import itertools

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

# TODO: Track the loss per event and report it
def conditional_weibull_loss_multi(f, s, e, n_risks: int, weights=None):
    loss = 0.0
    for k in range(n_risks):
        temp = (e[:, k] == 1)
        uncensored_loss = torch.sum(temp * f[:, k].T)
        censored_loss = torch.sum((~temp) * s[:, k].T)
        if weights is not None:
            loss += weights[k] * (uncensored_loss + censored_loss)
        else:
            loss += (uncensored_loss + censored_loss)
    loss = -loss / e.shape[0]
    return loss

def conditional_weibull_loss(f, s, e, n_risks: int):    
    loss = 0.0
    for k in range(n_risks):
        temp = (e == k)
        uncensored_loss = torch.sum(temp*f[:, k].T)
        censored_loss = torch.sum(~temp*s[:, k].T)
        loss += (uncensored_loss + censored_loss)    
    loss = -loss / e.shape[0]
    return loss