import torch
from torch import nn

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

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
    return loss

# Conditional Weibull loss for multi-event
def conditional_weibull_loss_multi(f, s, e, n_risks, device):
    if n_risks == 3:
        p1 = f[:,0] + f[:,1] + f[:,2] # [1 1 1]
        p2 = s[:,0] + s[:,1] + s[:,2] # [0 0 0]
        p3 = f[:,0] + s[:,1] + s[:,2] # [1 0 0]
        p4 = s[:,0] + f[:,1] + s[:,2] # [0 1 0]
        p5 = s[:,0] + s[:,1] + f[:,2] # [0 0 1]
        p6 = f[:,0] + f[:,1] + s[:,2] # [1 1 0]
        p7 = s[:,0] + f[:,1] + f[:,2] # [0 1 1]
        p8 = f[:,0] + s[:,1] + f[:,2] # [1 0 1]
        
        e1 = (e == torch.tensor([1., 1., 1.], device=device)).all(dim=1)
        e2 = (e == torch.tensor([0., 0., 0.], device=device)).all(dim=1)
        e3 = (e == torch.tensor([1., 0., 0.], device=device)).all(dim=1)
        e4 = (e == torch.tensor([0., 1., 0.], device=device)).all(dim=1)
        e5 = (e == torch.tensor([0., 0., 1.], device=device)).all(dim=1)
        e6 = (e == torch.tensor([1., 1., 0.], device=device)).all(dim=1)
        e7 = (e == torch.tensor([0., 1., 1.], device=device)).all(dim=1)
        e8 = (e == torch.tensor([1., 0., 1.], device=device)).all(dim=1)

        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3) \
             + torch.sum(e4 * p4) + torch.sum(e5 * p5) + torch.sum(e6 * p6) \
             + torch.sum(e7* p7) + torch.sum(e8 * p8)
        loss = -loss/e.shape[0]
    elif n_risks == 4:
        p1 = f[:,0] + f[:,1] + f[:,2] + f[:,3] # [1 1 1 1]
        p2 = s[:,0] + s[:,1] + s[:,2] + s[:,3] # [0 0 0 0]
        p3 = f[:,0] + s[:,1] + s[:,2] + s[:,3] # [1 0 0 0]
        p4 = s[:,0] + f[:,1] + s[:,2] + s[:,3] # [0 1 0 0]
        p5 = s[:,0] + s[:,1] + f[:,2] + s[:,3] # [0 0 1 0]
        p6 = s[:,0] + s[:,1] + s[:,2] + f[:,3] # [0 0 0 1]
        p7 = f[:,0] + f[:,1] + s[:,2] + s[:,3] # [1 1 0 0]
        p8 = f[:,0] + s[:,1] + f[:,2] + s[:,3] # [1 0 1 0]
        p9 = f[:,0] + s[:,1] + s[:,2] + f[:,3] # [1 0 0 1]
        p10 = s[:,0] + f[:,1] + f[:,2] + s[:,3] # [0 1 1 0]
        p11 = s[:,0] + f[:,1] + s[:,2] + f[:,3] # [0 1 0 1]
        p12 = s[:,0] + s[:,1] + f[:,2] + f[:,3] # [0 0 1 1]
        p13 = f[:,0] + f[:,1] + f[:,2] + s[:,3] # [1 1 1 0]
        p14 = f[:,0] + f[:,1] + s[:,2] + f[:,3] # [1 1 0 1]
        p15 = f[:,0] + s[:,1] + f[:,2] + f[:,3] # [1 0 1 1]
        p16 = s[:,0] + f[:,1] + f[:,2] + f[:,3] # [0 1 1 1]

        e1 = (e == torch.tensor([1., 1., 1., 1.], device=device)).all(dim=1)
        e2 = (e == torch.tensor([0., 0., 0., 0.], device=device)).all(dim=1)
        e3 = (e == torch.tensor([1., 0., 0., 0.], device=device)).all(dim=1)
        e4 = (e == torch.tensor([0., 1., 0., 0.], device=device)).all(dim=1)
        e5 = (e == torch.tensor([0., 0., 1., 0.], device=device)).all(dim=1)
        e6 = (e == torch.tensor([0., 0., 0., 1.], device=device)).all(dim=1)
        e7 = (e == torch.tensor([1., 1., 0., 0.], device=device)).all(dim=1)
        e8 = (e == torch.tensor([1., 0., 1., 0.], device=device)).all(dim=1)
        e9 = (e == torch.tensor([1., 0., 0., 1.], device=device)).all(dim=1)
        e10 = (e == torch.tensor([0., 1., 1., 0.], device=device)).all(dim=1)
        e11 = (e == torch.tensor([0., 1., 0., 1.], device=device)).all(dim=1)
        e12 = (e == torch.tensor([0., 0., 1., 1.], device=device)).all(dim=1)
        e13 = (e == torch.tensor([1., 1., 1., 0.], device=device)).all(dim=1)
        e14 = (e == torch.tensor([1., 1., 0., 1.], device=device)).all(dim=1)
        e15 = (e == torch.tensor([1., 0., 1., 1.], device=device)).all(dim=1)
        e16 = (e == torch.tensor([0., 1., 1., 1.], device=device)).all(dim=1)

        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3) \
            + torch.sum(e4 * p4) + torch.sum(e5 * p5) + torch.sum(e6 * p6) \
            + torch.sum(e7 * p7) + torch.sum(e8 * p8) + torch.sum(e9 * p9) \
            + torch.sum(e10 * p10) + torch.sum(e11 * p11) + torch.sum(e12 * p12) \
            + torch.sum(e13 * p13) + torch.sum(e14 * p14) + torch.sum(e15 * p15) \
            + torch.sum(e16 * p16)
        loss = -loss / e.shape[0]
    else:
        raise NotImplementedError()
        
    return loss
