import torch 
from utilities import safe_log


class Weibull_linear:
    def __init__(self, nf, alpha, gamma, device):
        self.nf = nf
        self.alpha = torch.tensor([alpha], device = device).type(torch.float32)
        self.gamma = torch.tensor([gamma], device = device).type(torch.float32)
        self.coeff = torch.rand((nf, ), device = device).type(torch.float32)


    def cum_hazard(self, t, x):
        return ((t / self.alpha) ** self.gamma) * torch.exp(torch.matmul(x, self.coeff))
    
    def hazard(self, t, x):
        return ((self.gamma / self.alpha) * ((t / self.alpha) ** (self.gamma - 1))) * torch.exp(torch.matmul(x, self.coeff))
    
    def survival(self, t, x):
        return torch.exp(-1.0 * self.cum_hazard(t, x))
    
    def PDF(self, t, x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)
    
    def rvs(self, x, u):
        return ((-safe_log(u) / torch.exp(torch.matmul(x, self.coeff))) ** (1/self.gamma)) * self.alpha