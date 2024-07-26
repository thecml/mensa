import torch
import math
import numpy as np 

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

class Clayton:
    def __init__(self, theta, eps, device):
        self.theta = torch.tensor([theta], device=device).type(torch.float64)
        self.eps = torch.tensor([eps], device=device).type(torch.float64)
        self.device = device
    

    def CDF(self, u):
        u = u.clamp(self.eps, 1.0-self.eps * 0.0)
        tmp = torch.exp(-self.theta * safe_log(u))
        tmp = torch.sum(tmp, dim = 1) - 2.0
        return torch.exp((-1.0 / self.theta) * safe_log(tmp))
    

    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    

    def enable_grad(self):
        self.theta.requires_grad = True
    
    def parameters(self):
        return [self.theta]


class Clayton_two:
    def __init__(self, theta, eps, device):
        self.theta = torch.tensor([theta], device=device).type(torch.float64)
        self.eps = torch.tensor([eps], device=device).type(torch.float64)
        self.device = device
    

    def CDF(self, u):
        u = u.clamp(self.eps, 1.0-self.eps * 0.0)
        tmp = torch.exp(-self.theta * safe_log(u))
        tmp = torch.sum(tmp, dim = 1) - 1.0
        return torch.exp((-1.0 / self.theta) * safe_log(tmp))
    

    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    

    def enable_grad(self):
        self.theta.requires_grad = True
    
    def parameters(self):
        return [self.theta]


class NestedClayton:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.P_clayton = Clayton_two(theta1, eps1, device)
        self.CH_clayton = Clayton_two(theta2, eps2, device)
        self.device = device
        self.eps=eps1

    
    def enable_grad(self):
        self.P_clayton.enable_grad()
        self.CH_clayton.enable_grad()
    
    def parameters(self):
        return [self.P_clayton.theta, self.CH_clayton.theta]

    def CDF(self, UV):
        U = self.CH_clayton.CDF(UV[:,:2]).reshape(-1,1)
        #print(U, UV[:,:2])
        new_uv = torch.cat([U, UV[:,2:3]], dim=1)
        return self.P_clayton.CDF(new_uv)
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps

        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
class Convex_Nested:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.P_clayton = Convex_2_Clayton(theta1, theta1, eps1, eps1, device)
        self.CH_clayton = Convex_2_Clayton(theta2, theta2, eps2, eps2, device)
        self.device = device
        self.eps = eps1
    
    def enable_grad(self):
        self.P_clayton.enable_grad()
        self.CH_clayton.enable_grad()
    
    def parameters(self):
        return [self.P_clayton.copula_1.theta,self.P_clayton.copula_2.theta, self.CH_clayton.copula_1.theta, self.CH_clayton.copula_2.theta, self.P_clayton.weight, self.CH_clayton.weight]

    def CDF(self, UV):
        U = self.CH_clayton.CDF(UV[:,:2]).reshape(-1,1)
        #print(U, UV[:,:2])
        new_uv = torch.cat([U, UV[:,2:3]], dim=1)
        return self.P_clayton.CDF(new_uv)
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps

        return (self.CDF(u_eps)-self.CDF(u))/self.eps

class Convex_Nested2:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.P_clayton = Convex_2_Clayton(theta1, theta1, eps1, eps1, device)
        self.CH_clayton = Convex_1_Clayton(theta2, eps2, device)
        self.device = device
        self.eps = eps1
    
    def enable_grad(self):
        self.P_clayton.enable_grad()
        self.CH_clayton.enable_grad()
    
    def parameters(self):
        return [self.P_clayton.copula_1.theta, self.P_clayton.copula_2.theta, self.CH_clayton.copula_1.theta, self.P_clayton.weight, self.CH_clayton.weight]

    def CDF(self, UV):
        U = self.CH_clayton.CDF(UV[:,:2]).reshape(-1,1)
        #print(U, UV[:,:2])
        new_uv = torch.cat([U, UV[:,2:3]], dim=1)
        return self.P_clayton.CDF(new_uv)
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps

        return (self.CDF(u_eps)-self.CDF(u))/self.eps

class Convex_2_Clayton:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.copula_1 = Clayton_two(theta1, eps1, device)
        self.copula_2 = Clayton_two(theta2, eps2, device)
        self.weight = torch.nn.parameter.Parameter(torch.rand(2,).to(device))
        self.device = device
        self.eps=eps1

    
    def enable_grad(self):
        self.copula_1.enable_grad()
        self.copula_2.enable_grad()
        self.weight.requires_grad = True
    
    def parameters(self):
        return [self.copula_2.theta, self.copula_1.theta, self.weight]

    def CDF(self, UV):
        cdf1 = self.copula_1.CDF(UV)
        cdf2 = self.copula_2.CDF(UV)
        return (torch.nn.Softmax()(self.weight) * torch.cat([cdf1.reshape(-1,1), cdf2.reshape(-1,1)], dim=1)).sum(dim=1) 
        
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps

        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
class Convex_1_Clayton:
    def __init__(self, theta1, eps1, device):
        self.copula_1 = Clayton_two(theta1, eps1, device)
        self.weight = torch.nn.parameter.Parameter(torch.rand(2,).to(device))
        self.device = device
        self.eps=eps1

    
    def enable_grad(self):
        self.copula_1.enable_grad()
        self.weight.requires_grad = True
    
    def parameters(self):
        return [self.copula_1.theta, self.weight]

    def CDF(self, UV):
        cdf1 = self.copula_1.CDF(UV)
        return (torch.nn.Softmax()(self.weight) * torch.cat([cdf1.reshape(-1,1)], dim=1)).sum(dim=1) 
        
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps

        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
class Convex_clayton:
    def __init__(self, theta1, theta2, eps1, eps2, device):
        self.copula_1 = Clayton(theta1, eps1, device)
        self.copula_2 = Clayton(theta2, eps2, device)
        self.weight = torch.nn.parameter.Parameter(torch.rand(2,).to(device))
        self.device = device
        self.eps=eps1

    
    def enable_grad(self):
        self.copula_1.enable_grad()
        self.copula_2.enable_grad()
        self.weight.requires_grad = True
    
    def parameters(self):
        return [self.copula_2.theta, self.copula_1.theta, self.weight]

    def CDF(self, UV):
        cdf1 = self.copula_1.CDF(UV)
        cdf2 = self.copula_2.CDF(UV)
        return (torch.nn.Softmax()(self.weight) * torch.cat([cdf1.reshape(-1,1), cdf2.reshape(-1,1)], dim=1)).sum(dim=1) 
        
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps

        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    



if __name__ == '__main__':
    from pycop import simulation
    np.random.seed(0)

    """u, v, w = simulation.simu_archimedean('clayton', 3,10,2.0)
    #print(u)
    u = torch.from_numpy(u).reshape(-1,1)
    v = torch.from_numpy(v).reshape(-1,1)
    w = torch.from_numpy(w).reshape(-1,1)
    U = torch.cat([u,v,w], axis=1)
    copula = Clayton(2.0, 1e-3, 'cpu')
    Y = copula.CDF(U)
    print(Y[:10])
    copula = NestedClayton(2.0, 3.0, 1e-3, 1e-3, 'cpu')
    Y = copula.CDF(U)
    print(Y[:10])
    #print(U)
    #print(Y)
    #print(copula.conditional_cdf('u', U))"""
    x = torch.nn.parameter.Parameter(torch.rand(2,))
    y = torch.cat([torch.rand((10,1)), torch.rand((10,1))], dim=1)
    print((y * x).sum(dim=1))
    #print(x)
    #print(torch.nn.Softmax()(x))
    #print(x.requires_grad)

