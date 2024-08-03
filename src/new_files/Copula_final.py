import torch 
import math 
import numpy as np 

def safe_log(x, eps=1e-6):
    return torch.log(x+eps*(x<eps))

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
    
class Clayton_Bivariate:
    def __init__(self, theta, eps, dtype, device):
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device
    
    def CDF(self, u):
        u = u.clamp(self.eps, 1.0 - self.eps)# can play with this range
        tmp = torch.exp(-self.theta * safe_log(u))
        tmp = torch.sum(tmp, dim=1) - 1.0
        return torch.exp((-1.0 / self.theta) * safe_log(tmp))
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1]  
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True

    def disable_grad(self):
        self.theta.requires_grad = False

    def parameters(self):
        return [self.theta]
    

    def __str__(self):
        return "theta = " + copula.theta.detach().clone().item()
    
    def set_params(self, theta):#on cpu
        self.theta = theta
    
class Frank_Bivariate:
    def __init__(self, theta, eps, dtype, device) -> None:
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device
    
    def CDF(self, u):
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) - log1mexp(-self.theta)
        return -1.0 / self.theta * log1mexp(tmp)
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1]  
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True

    def disable_grad(self):
        self.theta.requires_grad = False

    def parameters(self):
        return [self.theta]
    
    def __str__(self):
        return "theta = " + copula.theta.detach().clone().item()
    
    def set_params(self, theta):#on cpu
        self.theta = theta
    
    

class Convex_bivariate:
    def __init__(self, copulas=['cl', 'fr'], thetas=[2.0, 2.0], eps=1e-3, dtype=torch.float32, device='cpu'):
        self.copulas = []
        self.device = device
        self.n_copula = len(copulas)
        for i in range(len(copulas)):
            if copulas[i] == 'fr':
                copula_tmp = Frank_Bivariate(thetas[i], eps=eps, dtype=dtype, device=device)
            elif copulas[i] == 'cl':
                copula_tmp = Clayton_Bivariate(thetas[i], eps=eps, dtype=dtype, device=device)
            else:
                raise NotImplementedError('copula not implemented!!!!')
            self.copulas.append(copula_tmp)
            self.logits = torch.nn.parameter.Parameter(torch.rand(self.n_copula, ).to(device))

    def enable_grad(self):
        for copula in self.copulas:
            copula.enable_grad()
        self.logits.requires_grad = True

    def disable_grad(self):
        for copula in self.copulas:
            copula.disable_grad()
        self.logits.requires_grad = False
        
    def parameters(self):#[theta1, theta2, .., theta_n, logits]
        params = []
        for copula in self.copulas:
            params = params + copula.parameters()
        params = params + [self.logits]
        return params
    
    def CDF(self, u):
        cdf = torch.empty((u.shape[0], self.n_copula), device=self.device)
        for i in range(self.n_copula):
            cdf[:,i] = self.copulas[i].CDF(u)
        weights = torch.nn.Softmax(dim=0)(self.logits)
        
        return cdf @ weights
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1]  
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps

    def set_params(self, params):#on cpu
        for i,p in enumerate(params[:-1]):
            self.copulas[i].theta = p
        self.logits = params[-1]


class Nested_Convex_Copula:
    def __init__(self, child_copulas, parent_copulas, child_thetas, parent_thetas, eps=1e-3, dtype=torch.float32, device='cpu'):
        self.parent_copula = Convex_bivariate(parent_copulas, parent_thetas, eps=eps, dtype=dtype, device=device)
        self.child_copula = Convex_bivariate(child_copulas, child_thetas, eps=eps, dtype=dtype, device=device)
        self.eps = eps
        self.device = device
        self.child_list = child_copulas
        self.parent_list = parent_copulas
    
    def CDF(self, UV):
        U = self.child_copula.CDF(UV[:,:2]).reshape(-1,1)
        new_uv = torch.cat([U, UV[:,2:3]], dim=1)
        return self.parent_copula.CDF(new_uv)
    
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
        self.child_copula.enable_grad()
        self.parent_copula.enable_grad()
    
    def disable_grad(self):
        self.child_copula.disable_grad()
        self.parent_copula.disable_grad()


    def parameters(self):#[child_theta, parent_thetas, parent_logits, child_logits]
        return self.child_copula.parameters()[:-1] + self.parent_copula.parameters() + [self.child_copula.parameters()[-1]]
    
    
    def __str__(self) -> str:
        child_str = "Copulas: "
        for i in self.child_list:
            child_str += i
            child_str += ", "
        child_theta = "Thetas: "
        for i in self.child_copula.parameters()[:-1]:
            child_theta += str(np.round(i.detach().clone().item(),3))
            child_theta += ', '
        

        parent_str = "Copulas: "
        for i in self.parent_list:
            parent_str += i
            parent_str += ", "
        parent_theta = "Thetas: "
        for i in self.parent_copula.parameters()[:-1]:
            parent_theta += str(np.round(i.detach().clone().item(),3))
            parent_theta += ', '
        
        return "Child-->" + child_str + child_theta +' ' + "Parent-->" + parent_str + parent_theta
        
    def set_params(self, params):
        child_prams = params[:len(self.child_list)]
        child_prams.append(params[-1])
        parent_params = params[len(self.child_list):len(self.child_list) + len(self.parent_list)]
        parent_params.append(params[-2])
        self.child_copula.set_params(child_prams)
        self.parent_copula.set_params(parent_params)



class Clayton_Triple:
    def __init__(self, theta, eps, dtype, device):
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
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

    def disable_grad(self):
        self.theta.requires_grad = False
    
    def parameters(self):
        return [self.theta]

    def __str__(self) -> str:
        return "Clayton theta: " + str(np.round(self.theta.detach().clone().item(),3))
    
    def set_params(self, theta):#on cpu
        self.theta = theta
    

class Frank_Triple:
    def __init__(self, theta, eps, dtype, device):
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
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
        
    def parameters(self):
        return [self.theta]

    def __str__(self) -> str:
        return "Frank theta: " + str(np.round(self.theta.detach().clone().item(),3))
    
    def set_params(self, theta):#on cpu
        self.theta = theta
    

if __name__ == "__main__":
    from pycop import simulation
    #np.random.seed(0)
    u, v, w = simulation.simu_archimedean('frank', 3,1000,2.0)
    #print(u)
    u = torch.from_numpy(u).reshape(-1,1)
    v = torch.from_numpy(v).reshape(-1,1)
    w = torch.from_numpy(w).reshape(-1,1)
    U = torch.cat([u,v,w], axis=1)
    copula = Nested_Convex_Copula(['fr'], ['fr'], [1], [1], 1e-3, dtype=torch.float32, device='cpu')

        
    for p in copula.parameters():
        print(p.shape)
    
