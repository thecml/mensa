import torch

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

def safe_exp(x: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, max=max_val))

def weibull_log_pdf(t, k, lam):
    return safe_log(k) - safe_log(lam) + (k - 1) * safe_log(t/lam) - (t/lam)**k

def weibull_log_cdf(t, k, lam):
    return safe_log(1 - torch.exp(- (t / lam) ** k))

def weibull_log_survival(t, k, lam):
    return - (t / lam) ** k
