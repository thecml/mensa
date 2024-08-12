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

def conditional_weibull_loss_multi(model, x, t, E, elbo=True, copula=None):

    alpha = model.discount
    params = model.forward(x)

    f_risks = []
    s_risks = []

    for i in range(model.risks):
        t = t[:,i].reshape(-1,1).expand(-1, model.k) #(n, k)
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
    
    """ TO TEST
    total_loss = 0.0
    N = t.shape[0]
    for i in range(N):
        loss_i = 0.0
        for j in range(t.shape[1]):
            if E[i, j] == 1:
                loss_i += f[i, j]
            else:
                loss_i += s[i, j]
        total_loss -= loss_i / N
    return total_loss
    """
    event_loss = E * f
    non_event_loss = (1 - E) * s
    total_loss = torch.sum(event_loss + non_event_loss, dim=1)
    total_loss = -torch.mean(total_loss)
    return total_loss