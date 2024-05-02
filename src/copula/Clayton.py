import torch as torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd as autograd
from statsmodels.distributions.copula.api import ClaytonCopula

def LOG(x):
    return torch.log(x+1e-30*(x<=1e-30))

def DIV(x,y):
    return x/(y+(y<=1e-30)*1e-30)

class Clayton:
    def __init__(self, theta, device):
        self.theta = theta
        self.eps = torch.tensor([1e-4], device=device).type(torch.float32)
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

        
        

if __name__ == "__main__":
    #torch.manual_seed(0)
    copula = Clayton(theta=torch.tensor([2.0]).type(torch.float32))
    uv = copula.rvs(200)
    print(uv.dtype)
    copula.conditional_cdf_cf("u", uv)
    """SC = COX_EXP(0.3)
    ST = COX_EXP(0.1)
    T = ST.rvs(1000, uv[:,0])
    C = SC.rvs(1000, uv[:,1])
    E = (T < C).type(torch.float32)
    U = T * E + (1.0-E)*C
    copula.set_theta(torch.rand(1).type(torch.float32))

    c_optimizer = optim.Adam([copula.theta], lr=1e-3)
    #h_scheduler = optim.lr_scheduler.StepLR(h_optimizer, 1000, 0.95)
    c_scheduler = optim.lr_scheduler.StepLR(c_optimizer, 1000, 0.95)
    

    for i in range(10000):
        copula.enable_grad()
        c_optimizer.zero_grad()
        loss = -1.0*torch.mean(LOG(copula.PDF(uv)))
        
        loss.backward()
        c_optimizer.step()
        c_scheduler.step()
        if i%200==0:
            print("Epoch: ", i, "Loss: ", loss.detach().numpy(), "Theta: ", copula.theta.detach().numpy())

    """