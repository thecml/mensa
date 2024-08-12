import torch
from torch import nn
from mensa.utility import *

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

def double_loss(model, X, T, E, copula):
    k1, k2, lam1, lam2 = model(X)
    log_pdf1 = weibull_log_pdf(T, k1, lam1)
    log_pdf2 = weibull_log_pdf(T, k2, lam2)
    log_surv1 = weibull_log_survival(T, k1, lam1)
    log_surv2 = weibull_log_survival(T, k2, lam2)
    if copula is None:
        p1 = log_pdf1 + log_surv2
        p2 = log_surv1 + log_pdf2
    else:
        S = torch.cat([torch.exp(log_surv1).reshape(-1,1), torch.exp(log_surv2).reshape(-1,1)], dim=1)
        p1 = log_pdf1 + safe_log(copula.conditional_cdf("u", S))
        p2 = log_pdf2 + safe_log(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    e1 = (E == 0) * 1.0
    e2 = (E == 1) * 1.0
    loss = torch.sum(e1 * p1) + torch.sum(e2 * p2)
    loss = -loss/E.shape[0]
    return loss

def triple_loss(model, X, T, E, copula):
    k1, k2, k3, lam1, lam2, lam3 = model(X)
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
        S = torch.cat([torch.exp(log_surv1).reshape(-1,1), torch.exp(log_surv2).reshape(-1,1), torch.exp(log_surv3).reshape(-1,1)], dim=1).clamp(0.002, 0.998)
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

def conditional_weibull_loss(model, x, t, E, elbo=True, copula=None):

    alpha = model.discount
    params = model.forward(x)

    t = t.reshape(-1,1).expand(-1, model.k)#(n, k)
    f_risks = []
    s_risks = []

    for i in range(model.risks):
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

    if model.risks == 3:
        if copula is None:
            p1 = f[:,0] + s[:,1] + s[:,2] 
            p2 = s[:,0] + f[:,1] + s[:,2]
            p3 = s[:,0] + s[:,1] + f[:,2]
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
            loss = -loss/E.shape[0]
        else:
            S = torch.exp(s).clamp(0.001,0.999)
            p1 = f[:,0] + safe_log(copula.conditional_cdf("u", S))
            p2 = f[:,1] + safe_log(copula.conditional_cdf("v", S))
            p3 = f[:,2] + safe_log(copula.conditional_cdf("w", S))
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
            loss = -loss/E.shape[0]
    elif model.risks == 2:
        if copula is None:
            p1 = f[:,0] + s[:,1]
            p2 = s[:,0] + f[:,1]
            e1 = (E == 1) * 1.0
            e2 = (E == 0) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) 
            loss = -loss/E.shape[0]
        else:
            S = torch.exp(s).clamp(0.001,0.999)
            p1 = f[:,0] + safe_log(copula.conditional_cdf("u", S))
            p2 = f[:,1] + safe_log(copula.conditional_cdf("v", S))
            e1 = (E == 1) * 1.0
            e2 = (E == 0) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2)
            loss = -loss/E.shape[0]
    elif model.risks == 1:#added single risk 
        e1 = (E == 1) * 1.0
        e2 = (E == 0) * 1.0
        loss = torch.sum(e1 * f[:,0]) + torch.sum(e2 * s[:,0]) 
        loss = -loss/E.shape[0]
    return loss

def conditional_weibull_loss_multi(model, X, T, E, device, elbo=True, copula=None):

    alpha = model.discount
    params = model.forward(X)

    f_risks = []
    s_risks = []

    for i in range(model.risks):
        t = T[:,i].reshape(-1,1).expand(-1, model.k) #(n, k)
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
    
    if model.risks == 3:
        p1 = f[:,0] + f[:,1] + f[:,2] # [1 1 1]
        p2 = s[:,0] + s[:,1] + s[:,2] # [0 0 0]
        p3 = f[:,0] + s[:,1] + s[:,2] # [1 0 0]
        p4 = s[:,0] + f[:,1] + s[:,2] # [0 1 0]
        p5 = s[:,0] + s[:,1] + f[:,2] # [0 0 1]
        p6 = f[:,0] + f[:,1] + s[:,2] # [1 1 0]
        p7 = s[:,0] + f[:,1] + f[:,2] # [0 1 1]
        p8 = f[:,0] + s[:,1] + f[:,2] # [1 0 1]
        
        e1 = (E == torch.tensor([1., 1., 1.], device=device)).all(dim=1)
        e2 = (E == torch.tensor([0., 0., 0.], device=device)).all(dim=1)
        e3 = (E == torch.tensor([1., 0., 0.], device=device)).all(dim=1)
        e4 = (E == torch.tensor([0., 1., 0.], device=device)).all(dim=1)
        e5 = (E == torch.tensor([0., 0., 1.], device=device)).all(dim=1)
        e6 = (E == torch.tensor([1., 1., 0.], device=device)).all(dim=1)
        e7 = (E == torch.tensor([0., 1., 1.], device=device)).all(dim=1)
        e8 = (E == torch.tensor([1., 0., 1.], device=device)).all(dim=1)

        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3) \
             + torch.sum(e4 * p4) + torch.sum(e5 * p5) + torch.sum(e6 * p6) \
             + torch.sum(e7* p7) + torch.sum(e8 * p8)
        loss = -loss/E.shape[0]
    elif model.risks == 4:
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

        e1 = (E == torch.tensor([1., 1., 1., 1.], device=device)).all(dim=1)
        e2 = (E == torch.tensor([0., 0., 0., 0.], device=device)).all(dim=1)
        e3 = (E == torch.tensor([1., 0., 0., 0.], device=device)).all(dim=1)
        e4 = (E == torch.tensor([0., 1., 0., 0.], device=device)).all(dim=1)
        e5 = (E == torch.tensor([0., 0., 1., 0.], device=device)).all(dim=1)
        e6 = (E == torch.tensor([0., 0., 0., 1.], device=device)).all(dim=1)
        e7 = (E == torch.tensor([1., 1., 0., 0.], device=device)).all(dim=1)
        e8 = (E == torch.tensor([1., 0., 1., 0.], device=device)).all(dim=1)
        e9 = (E == torch.tensor([1., 0., 0., 1.], device=device)).all(dim=1)
        e10 = (E == torch.tensor([0., 1., 1., 0.], device=device)).all(dim=1)
        e11 = (E == torch.tensor([0., 1., 0., 1.], device=device)).all(dim=1)
        e12 = (E == torch.tensor([0., 0., 1., 1.], device=device)).all(dim=1)
        e13 = (E == torch.tensor([1., 1., 1., 0.], device=device)).all(dim=1)
        e14 = (E == torch.tensor([1., 1., 0., 1.], device=device)).all(dim=1)
        e15 = (E == torch.tensor([1., 0., 1., 1.], device=device)).all(dim=1)
        e16 = (E == torch.tensor([0., 1., 1., 1.], device=device)).all(dim=1)

        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3) \
            + torch.sum(e4 * p4) + torch.sum(e5 * p5) + torch.sum(e6 * p6) \
            + torch.sum(e7 * p7) + torch.sum(e8 * p8) + torch.sum(e9 * p9) \
            + torch.sum(e10 * p10) + torch.sum(e11 * p11) + torch.sum(e12 * p12) \
            + torch.sum(e13 * p13) + torch.sum(e14 * p14) + torch.sum(e15 * p15) \
            + torch.sum(e16 * p16)
        loss = -loss / E.shape[0]
    else:
        raise NotImplementedError()
        
    return loss
    
    #"""
    """
    event_loss = E * f
    non_event_loss = (1 - E) * s
    total_loss = torch.sum(event_loss + non_event_loss, dim=1)
    total_loss = -torch.mean(total_loss)
    return total_loss
    """