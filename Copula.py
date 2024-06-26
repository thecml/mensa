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
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) + log1mexp(-self.theta*u[:,2]) - log1mexp(-self.theta) - log1mexp(-self.theta)
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
    
    def parameters(self):
        return [self.theta]

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


class Clayton:
    def __init__(self, theta, epsilon=1e-3, device='cpu'):
        self.theta = theta
        self.eps = torch.tensor([epsilon], device=device).type(torch.float32)
        self.device = device
        self.flag = True

    def CDF(self, u):#(sum(ui**-theta)-2)**(-1/theta)
        u = u + 1e-30*(u<1e-30)
        u = u.clamp(0,1)
        #tmp = torch.sum(u**(-1.0*self.theta), dim=1)-1
        tmp = torch.exp(-self.theta * LOG(u))
        tmp = torch.sum(tmp, dim=1)-2
        return torch.exp((-1/self.theta)*LOG(tmp))
    
    def conditional_cdf(self, condition_on, uv):
        if not self.flag:
            
            U = torch.autograd.Variable(uv,requires_grad=True)
            Y = self.CDF(U)
            grad = torch.autograd.grad(
                    outputs=[Y],
                    inputs=[U],
                    grad_outputs=torch.ones_like(Y),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )[0]
            if condition_on == "u":
                return grad[:,0]
            elif condition_on == 'v':
                return grad[:,1]
            else:
                return grad[:,2]
        else:
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

    def parameters(self):
        return [self.theta]
    
    
        

if __name__ == "__main__":
    from pycop import simulation

    u, v, w = simulation.simu_archimedean('clayton', 3, 5, 3.0)
    u = torch.from_numpy(u).reshape(-1,1)
    v = torch.from_numpy(v).reshape(-1,1)
    w = torch.from_numpy(w).reshape(-1,1)
    U = torch.cat([u,v,w], axis=1)
    U = torch.autograd.Variable(U,requires_grad=True)
    copula = Clayton(torch.tensor([3.0]))
    Y = copula.CDF(U)
    print(torch.autograd.grad(
                outputs=[Y],
                inputs=[U],
                grad_outputs=torch.ones_like(Y),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0][:,0])
    
    print(copula.conditional_cdf('u', U).shape)
    print(copula.conditional_cdf('v', U).shape)
    print(copula.conditional_cdf('w', U))