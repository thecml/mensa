import torch as torch 
from pycop import simulation
import math
from Copula import Clayton as triple_cl
from Copula import Frank as triple_fr
import matplotlib.pyplot as plt

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
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) - log1mexp(-self.theta) 
        return -1.0 / self.theta * log1mexp(tmp)
    
    def enable_grad(self):
        self.theta.requires_grad = True
    
    def disable_grad(self):
        self.theta.requires_grad = False
    
    def set_theta(self, new_val):
        self.theta = new_val


class Clayton:#bivariate
    def __init__(self, theta, epsilon=1e-3, device='cpu'):
        self.theta = theta
        self.eps = torch.tensor([epsilon], device=device).type(torch.float32)
        self.device = device

    def CDF(self, u):#(sum(ui**-theta)-2)**(-1/theta)
        u = u + 1e-30*(u<1e-30)
        u = u.clamp(0,1)
        tmp = torch.exp(-self.theta * LOG(u))
        tmp = torch.sum(tmp, dim=1)-1
        return torch.exp((-1/self.theta)*LOG(tmp))
    
    def enable_grad(self):
        self.theta.requires_grad = True

class NestedClayton:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.P_clayton = Clayton(theta1, eps1)
        self.CH_clayton = Clayton(theta2, eps2)
        self.device = device
        self.eps = eps1
    
    def enable_grad(self):
        self.P_clayton.enable_grad()
        self.CH_clayton.enable_grad()
    
    def parameters(self):
        return [self.P_clayton.theta, self.CH_clayton.theta]
    
    def CDF(self, UV):#UV --- > (N,3) shape
        U = self.CH_clayton.CDF(UV[:,:2]).reshape(-1,1)
        new_UV = torch.cat([U, UV[:,2:3]], dim=1)
        return self.P_clayton.CDF(new_UV)
    
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
    

class NestedFrank:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.P_frank = Frank(theta1, eps1)
        self.CH_frank = Frank(theta2, eps2)
        self.device = device
        self.eps = eps1
    
    def enable_grad(self):
        self.P_frank.enable_grad()
        self.CH_frank.enable_grad()
    
    def parameters(self):
        return [self.P_frank.theta, self.CH_frank.theta]
    
    def CDF(self, UV):#UV --- > (N,3) shape
        U = self.CH_frank.CDF(UV[:,:2]).reshape(-1,1)
        new_UV = torch.cat([U, UV[:,2:3]], dim=1)
        return self.P_frank.CDF(new_UV)
    
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
    

class ConvexCopula:
    def __init__(self, copula1, copula2, beta, device):
        self.copula1 = copula1
        self.copula2 = copula2
        self.beta = torch.tensor([beta], device=device).type(torch.float32)

    def CDF(self, uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.copula1.CDF(uv)* p + (1.0-p)*self.copula2.CDF(uv)
    
    def conditional_cdf(self, condition_on, uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.copula1.conditional_cdf(condition_on, uv)* p + (1.0-p)*self.copula2.conditional_cdf(condition_on, uv)
    
    def enable_grad(self):
        self.copula1.enable_grad()
        self.copula2.enable_grad()
        self.beta.requires_grad = True
    
    def parameters(self):
        return self.copula1.parameters() + self.copula2.parameters() + [self.beta]
        



if __name__ == "__main__":
    torch.manual_seed(0)
    u, v, w = simulation.simu_archimedean('frank', 3, 1000, 2.0)
    u = torch.from_numpy(u).reshape(-1,1)
    v = torch.from_numpy(v).reshape(-1,1)
    w = torch.from_numpy(w).reshape(-1,1)
    U = torch.cat([u,v,w], dim=1)
    c2 = NestedFrank(torch.tensor([2]),torch.tensor([2]),1e-4,1e-4, 'cpu')
    c1 = NestedClayton(torch.tensor([2]),torch.tensor([2]),1e-4,1e-4, 'cpu')
    conv = ConvexCopula(c1, c2, beta=10000, device='cpu')
    cdf_conv = conv.CDF(U)
    
    
    NC = NestedFrank(torch.tensor([2]),torch.tensor([2]),1e-4,1e-4, 'cpu')
    t1 = NC.CDF(U)

    c3 = triple_fr(torch.tensor([2]), 1e-4, 'cpu')
    t2 = c3.CDF(U)
    print(t1[:10])
    print(t2[:10])
    print(cdf_conv[:10])
    
    