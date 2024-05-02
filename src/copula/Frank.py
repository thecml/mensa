import torch as torch
import matplotlib.pyplot as plt
#from Cox import COX_EXP
#import torch.optim as optim
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.archimedean import FrankCopula
import math
from statsmodels.distributions.copula.api import FrankCopula

def LOG(x):
    return torch.log(x+1e-6*(x<=1e-6))

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
        

if __name__ == "__main__":
    #torch.manual_seed(0)
    t = 14.0
    copula = Frank(theta=torch.tensor([t]).type(torch.float32))
    uv = torch.rand((2000,2))
    plt.scatter(uv[:,0], uv[:,1])
    uv = copula.rvs(2000)
    
    
    plt.scatter(uv[:,0], uv[:,1])
    
    plt.show()