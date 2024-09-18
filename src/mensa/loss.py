import torch
from torch import nn
import itertools

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

def generate_boolean_permutations(n):
    permutations = list(itertools.product([False, True], repeat=n))
    permutations = [list(permutation) for permutation in permutations]
    return permutations

# Conditional Weibull loss for single-event and competing risks
def conditional_weibull_loss(f, s, e, n_risks):
    if n_risks == 4:
        p1 = f[:,0] + s[:,1] + s[:,2] + s[:,3]
        p2 = s[:,0] + f[:,1] + s[:,2] + s[:,3]
        p3 = s[:,0] + s[:,1] + f[:,2] + s[:,3]
        p4 = s[:,0] + s[:,1] + s[:,2] + f[:,3]
        e1 = (e == 0) * 1.0
        e2 = (e == 1) * 1.0
        e3 = (e == 2) * 1.0
        e4 = (e == 2) * 1.0
        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3) + torch.sum(e4 * p4)
        loss = -loss/e.shape[0]
    elif n_risks == 3:
        p1 = f[:,0] + s[:,1] + s[:,2]
        p2 = s[:,0] + f[:,1] + s[:,2]
        p3 = s[:,0] + s[:,1] + f[:,2]
        e1 = (e == 0) * 1.0
        e2 = (e == 1) * 1.0
        e3 = (e == 2) * 1.0
        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
        loss = -loss/e.shape[0]
    elif n_risks == 2:
        p1 = f[:,0] + s[:,1]
        p2 = s[:,0] + f[:,1]
        e1 = (e == 1) * 1.0
        e2 = (e == 0) * 1.0
        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) 
        loss = -loss/e.shape[0]
    elif n_risks == 1:
        e1 = (e == 1) * 1.0
        e2 = (e == 0) * 1.0
        loss = torch.sum(e1 * f[:,0]) + torch.sum(e2 * s[:,0]) 
        loss = -loss/e.shape[0]
    else:
        raise NotImplementedError()
    return loss

# Conditional Weibull loss for multi-event
def conditional_weibull_loss_multi(f, s, e, n_risks: int):    
    permutations = generate_boolean_permutations(n_risks)
    loss = 0.0
    batch_size = e.shape[0]
    for permutation in permutations:
        temp_p = torch.zeros(batch_size, device=f.device, dtype=f.dtype)
        temp_e = torch.ones(batch_size, device=e.device, dtype=torch.bool)
        for idx, bool_num in enumerate(permutation):            
            if bool_num:
                temp_p += f[:, idx]
                temp_e &= e[:, idx] == 1
            else:
                temp_p += s[:, idx]
                temp_e &= e[:, idx] == 0
        # Only accumulate loss where temp_e is True
        if temp_e.any():
            loss += torch.sum(temp_p[temp_e])
    loss = -loss / batch_size
    return loss