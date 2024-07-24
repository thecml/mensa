import torch
from mensa.utility import *

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

def double_loss(model, X, T, E, copula, device):
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

def triple_loss(model, X, T, E, copula, device):
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
        S = torch.cat([torch.exp(log_surv1).reshape(-1,1), torch.exp(log_surv2).reshape(-1,1), torch.exp(log_surv3).reshape(-1,1)], dim=1)
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