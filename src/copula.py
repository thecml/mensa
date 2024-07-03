import torch as torch 
from pycop import simulation
import math
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GumbelCopula

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
    def __init__(self, theta, eps, device, dtype):
        self.theta = theta
        self.eps = torch.tensor([eps], device=device).type(dtype)
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

    def parameters(self):
        return [self.theta]

class Clayton:
    def __init__(self, theta, epsilon, device, dtype):
        self.theta = theta
        self.eps = torch.tensor([epsilon], device=device).type(dtype)
        self.device = device

    def CDF(self, u):#(sum(ui**-theta)-2)**(-1/theta)
        u = u + 1e-30*(u<1e-30)
        u = u.clamp(0,1)
        tmp = torch.exp(-self.theta * LOG(u))
        tmp = torch.sum(tmp, dim=1)-1
        return torch.exp((-1/self.theta)*LOG(tmp))
    
    def enable_grad(self):
        self.theta.requires_grad = True
        
    def parameters(self):
        return [self.theta]

class NestedClayton:
    def __init__(self, theta1, theta2, eps1, eps2, device, dtype):
        self.P_clayton = Clayton(theta1, eps1, device, dtype)
        self.CH_clayton = Clayton(theta2, eps2, device, dtype)
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
    def __init__(self, theta1, theta2, eps1, eps2, device, dtype):
        self.P_frank = Frank(theta1, eps1, device, dtype)
        self.CH_frank = Frank(theta2, eps2, device, dtype)
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
    def __init__(self, copula1, copula2, beta, device, dtype):
        self.copula1 = copula1
        self.copula2 = copula2
        self.beta = torch.tensor([beta], device=device).type(dtype)

    def CDF(self, uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.copula1.CDF(uv)* p + (1.0-p)*self.copula2.CDF(uv)
    
    def conditional_cdf(self, condition_on, uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.copula1.conditional_cdf(condition_on, uv) * p + (1.0-p) * self.copula2.conditional_cdf(condition_on, uv)
    
    def enable_grad(self):
        self.copula1.enable_grad()
        self.copula2.enable_grad()
        self.beta.requires_grad = True
    
    def parameters(self):
        return self.copula1.parameters() + self.copula2.parameters() + [self.beta]
    
class Clayton3:
    def __init__(self, theta, eps, dtype, device):
        self.theta = theta
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device

    def CDF(self, u):#(sum(ui**-theta)-2)**(-1/theta)
        u = u + 1e-30*(u<1e-30)
        u = u.clamp(0,1)
        #tmp = torch.sum(u**(-1.0*self.theta), dim=1)-1
        tmp = torch.exp(-self.theta * LOG(u))
        tmp = torch.sum(tmp, dim=1)-2
        return torch.exp((-1/self.theta)*LOG(tmp))
    
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
        
    def parameters(self):
        return [self.theta]

class Clayton2D:
    def __init__(self, theta, device, dtype):
        self.theta = theta.to(device)
        self.eps = torch.tensor([1e-4], device=device).type(dtype)
        self.device = device

    def rvs(self, n_samples):
        return torch.from_numpy(ClaytonCopula(self.theta.cpu().numpy().item()).rvs(n_samples)).to(self.device)
        #torch.manual_seed(0)
        """x = torch.rand((n_samples, 2))
        v = torch.distributions.gamma.Gamma(1.0/self.theta, torch.tensor(1.0)).sample((n_samples,))
        
        return (1 - DIV(LOG(x), v)) ** (-1. / self.theta)"""
        #return DIV(1, (1.0 - DIV(LOG(x), v))**(1.0/self.theta))
    
    def CDF(self, u):
        u = u + 1e-30*(u<1e-30)
        u = u.clamp(0,1)
        #tmp = torch.sum(u**(-1.0*self.theta), dim=1)-1
        tmp = torch.exp(-self.theta * LOG(u))
        tmp = torch.sum(tmp, dim=1)-1
        return torch.exp((-1/self.theta)*LOG(tmp))
        
        #return tmp**(-1.0/self.theta)
    
    def conditional_cdf(self, condition_on, uv):
        uv_eps = torch.empty_like(uv, device=self.device)
        if condition_on == "u":
            uv_eps[:,0] = uv[:,0] + self.eps
            uv_eps[:,1] = uv[:,1]
        else:
            uv_eps[:,1] = uv[:,1] + self.eps
            uv_eps[:,0] = uv[:,0]

        return (self.CDF(uv_eps) - self.CDF(uv))/self.eps
    
    def conditional_cdf_cf(self, condition_on, uv):
        #uv = uv + 1e-30*(uv<1e-30)
        a = torch.sum(uv**(-1.0*self.theta), dim=1)-1
        b = -1.0 * ((self.theta + 1)/self.theta)
        if condition_on == "u":
            c = (uv[:,0])**(-1.0*(self.theta+1.0))
        else:
            c = (uv[:,1])**(-1.0*(self.theta+1.0))
        return (a**b) * c
    
    def enable_grad(self):
        self.theta.requires_grad = True
    
    def disable_grad(self):
        self.theta.requires_grad = False
    
    def set_theta(self, new_val):
        self.theta = new_val
        
    def parameters(self):
        return [self.theta]

    def PDF(self, u):
        u = u + 1e-30*(u<1e-30)
        a = torch.sum(u**(-1.0*self.theta), dim=1)-1
        b = (self.theta+1) * torch.prod(u**(-1.0*(self.theta+1)),dim=1)
        c = -1.0*(2 + (1/self.theta))
        return (a**c)*b
    
    def Conditional_sampling(self, u, delta_):
        #delta==1 ==> observed u lower bound on v, delta == 0 ==>observed v, lower bound on u
        low_bound = delta_*u[:,0] + (1-delta_)*u[:,1]
        z_prime = torch.rand((u.shape[0],2)) * (1.0-low_bound).reshape(-1,1).repeat(1,2)
        z_prime += low_bound.reshape(-1,1).repeat(1,2)
        #z_prime = torch.rand((u.shape[0],2))
        tmp = ((z_prime*(u**(self.theta+1)))**(-self.theta/(self.theta+1))+1-(u**(-self.theta)))**(-1/self.theta)
        u_hat = tmp[:,0]
        v_hat = tmp[:,1]
        u_ = delta_ * u[:,0] + u_hat * (1.0-delta_)
        v_ = delta_ * v_hat + u[:,1] * (1.0-delta_)
        return torch.cat([u_.reshape(-1,1), v_.reshape(-1,1)],dim=1)
    
    def generator(self, u):
        return ((1/self.theta)*(u**(-self.theta)-1)).detach()
    
    def inv_generator(self, x):
        return ((1+self.theta*x)**(-1/self.theta)).detach()
    
class Frank2D:
    def __init__(self, theta, device):
        self.theta = theta
        self.eps = torch.tensor([1e-4], device=device).type(torch.float32)
        self.device = device

    def rvs(self, n_samples):
        uv = FrankCopula(self.theta.cpu().numpy().item()).rvs(n_samples)
        tmp = torch.from_numpy(uv)
        return tmp.to(self.device)
        torch.manual_seed(0)
        x = torch.rand((n_samples, 2))
        v = torch.tensor(stats.logser.rvs(1. - torch.exp(-self.theta),
                             size=(n_samples,))).type(torch.float32).reshape(-1,1)
        
        #return -1. / self.theta * LOG(1. + torch.exp(-(-LOG(x) / v))
        #                         * (torch.exp(-self.theta) - 1.))
        
        return DIV(1, (1.0 - DIV(LOG(x), v))**(1.0/self.theta))
    
    def CDF(self, u):
        #t = torch.exp(-self.theta*LOG(u))
        #t = torch.sum(t, dim=1) - 1.0
        #return torch.exp(-1.0 * LOG(t)/self.theta)
        #u = u.clamp(0,1.0)
        #print(self.theta.device, u.device)
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) - log1mexp(-self.theta)
        #num = torch.prod(1 - torch.exp(- self.theta * u), dim=-1)
        #den = (1 - torch.exp(-self.theta)) 
        return -1.0 / self.theta * log1mexp(tmp)
        #return -1.0 / self.theta * LOG(1 - num / den)
    
    def conditional_cdf(self, condition_on, uv):
        uv_eps = torch.empty_like(uv, device=self.device)
        if condition_on == "u":
            uv_eps[:,0] = uv[:,0] + self.eps
            uv_eps[:,1] = uv[:,1]
        else:
            uv_eps[:,1] = uv[:,1] + self.eps
            uv_eps[:,0] = uv[:,0]

        return (self.CDF(uv_eps) - self.CDF(uv))/self.eps
    
    def conditional_cdf_cf(self, condition_on, uv):
        uv = uv + 1e-8*(uv<1e-8)
        num = torch.prod(1 - torch.exp(- self.theta * uv), dim=1)
        den = (1 - torch.exp(-self.theta)) 
        p1 = 1/(1 - (num / den))
        p2 = (torch.exp(- self.theta * uv) - 1.0)/(torch.exp(-self.theta)-1)
        p3 = torch.exp(- self.theta * uv)
        if condition_on == "u":
            return p1 * p2[:,1]*p3[:,0]
        else:
            return p1 * p2[:,0]*p3[:,1]
    def enable_grad(self):
        self.theta.requires_grad = True
    
    def disable_grad(self):
        self.theta.requires_grad = False
    
    def set_theta(self, new_val):
        self.theta = new_val

    def PDF(self, u):
        u = u + 1e-8*(u<1e-8)
        g_ = torch.exp(-self.theta * torch.sum(u, axis=1)) - 1
        g1 = torch.exp(-self.theta) - 1

        num = -self.theta * g1 * (1 + g_)
        aux = torch.prod(torch.exp(-self.theta * u) - 1, dim=1) + g1
        den = aux ** 2
        return num / den
        
    def Conditional_sampling(self, u, delta_):
        #delta==1 ==> observed u lower bound on v, delta == 0 ==>observed v, lower bound on u
        low_bound = delta_*u[:,0] + (1-delta_)*u[:,1]
        z_prime = torch.rand((u.shape[0],2)) #* (1.0-low_bound).reshape(-1,1).repeat(1,2)
        z_prime += low_bound.reshape(-1,1).repeat(1,2)
        z_prime = torch.rand((u.shape[0],2))
        

        C = z_prime*torch.exp(self.theta*u)*(torch.exp(-self.theta)-1)
        B = ((1/C)-(torch.exp(-self.theta*u)-1)/(torch.exp(-self.theta)-1))
        v = LOG((B+1)/B)/(-self.theta)
        u_hat = v[:,0]
        v_hat = v[:,1]
        u_ = delta_ * u[:,0] + u_hat * (1.0-delta_)
        v_ = delta_ * v_hat + u[:,1] * (1.0-delta_)
        return torch.cat([u_.reshape(-1,1), v_.reshape(-1,1)],dim=1)
    