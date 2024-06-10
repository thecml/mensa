import torch as torch
import math

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def DIV(x,y):
    return x/(y+(y<=1e-6)*1e-6)

def log1mexp(x):
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

class Frank:
    def __init__(self, theta, eps=1e-3, device='cpu'):
        self.theta = theta
        self.eps = torch.tensor([eps], device=device).type(torch.float32)
        self.device = device

    
    def CDF(self, u):
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) + + log1mexp(-self.theta*u[:,2]) - log1mexp(-self.theta) - log1mexp(-self.theta)
        return -1.0 / self.theta * log1mexp(tmp)
    
    def conditional_cdf(self, condition_on, uv):
        uv_eps = torch.empty_like(uv, device=self.device)
        if condition_on == "u":
            uv_eps[:,0] = uv[:,0] + self.eps
            uv_eps[:,1] = uv[:,1]
            uv_eps[:,2] = uv[:,2]
        elif condition_on == 'v':
            uv_eps[:,1] = uv[:,1] + self.eps
            uv_eps[:,0] = uv[:,0]
            uv_eps[:,2] = uv[:,2]
        else:
            uv_eps[:,2] = uv[:,2] + self.eps
            uv_eps[:,1] = uv[:,1]
            uv_eps[:,0] = uv[:,0]


        return (self.CDF(uv_eps) - self.CDF(uv))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True
    
    def disable_grad(self):
        self.theta.requires_grad = False
    
    def set_theta(self, new_val):
        self.theta = new_val

class Gumbel:
    def __init__(self, theta, eps=1e-3, device='cpu'):
        self.theta = theta
        self.eps = torch.tensor([eps], device=device).type(torch.float32)
        self.device = device

    
    def CDF(self, u):
        tmp = ( (-LOG(u)) ** self.theta ).sum(dim=1)
        tmp = tmp **(1/self.theta)
        return torch.exp(-1.0*tmp)
    
    def conditional_cdf(self, condition_on, uv):
        uv_eps = torch.empty_like(uv, device=self.device)
        if condition_on == "u":
            uv_eps[:,0] = uv[:,0] + self.eps
            uv_eps[:,1] = uv[:,1]
            uv_eps[:,2] = uv[:,2]
        elif condition_on == 'v':
            uv_eps[:,1] = uv[:,1] + self.eps
            uv_eps[:,0] = uv[:,0]
            uv_eps[:,2] = uv[:,2]
        else:
            uv_eps[:,2] = uv[:,2] + self.eps
            uv_eps[:,1] = uv[:,1]
            uv_eps[:,0] = uv[:,0]


        return (self.CDF(uv_eps) - self.CDF(uv))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True
    
    def disable_grad(self):
        self.theta.requires_grad = False
    
    def set_theta(self, new_val):
        self.theta = new_val
        

if __name__ == "__main__":
    x = torch.rand((100,3))
    tmp = (-LOG(x))**2
    tmp = tmp.sum(dim=1)
    print(tmp.shape)