import numpy as np 
import torch
import matplotlib.pyplot as plt
from pycop import simulation
import argparse
import json



def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class Clayton:
    def __init__(self, theta, epsilon=1e-3, device='cpu'):
        self.theta = theta.to(device)
        self.eps = torch.tensor([epsilon], device=device).type(torch.float32)
        self.device = device

    def CDF(self, u):#(sum(ui**-theta)-2)**(-1/theta)
        u = u.to(self.device)        
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

    

def generate_events(dgp1, dgp2, dgp3, x, device,theta=2.0, family='clayton'):
    if family is None:
        uv = torch.randn((x.shape[0], 3)).to(device)#sample idnependent 
    else:
        u,v,w = simulation.simu_archimedean(family, 3, x.shape[0], theta=theta)
        u = torch.from_numpy(u).type(torch.float32).reshape(-1,1)
        v = torch.from_numpy(v).type(torch.float32).reshape(-1,1)
        w = torch.from_numpy(w).type(torch.float32).reshape(-1,1)
        
        uv = torch.cat([u,v,w], axis=1).to(device)

    x = x.to(device)
    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    t3 = dgp3.rvs(x, uv[:,2])
    T = np.concatenate([t1.cpu().reshape(-1,1),
                        t2.cpu().reshape(-1,1),
                        t3.cpu().reshape(-1,1)], axis=1)
    E = np.argmin(T,axis=1)
    obs_T = T[np.arange(T.shape[0]), E]
    T = torch.from_numpy(T).type(torch.float32).to(device)
    E = torch.from_numpy(E).type(torch.float32).to(device)
    obs_T = torch.from_numpy(obs_T).type(torch.float32).to(device)

    return {'X':x,'E':E, 'T':obs_T, 't1':t1, 't2':t2, 't3':t3}

def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {"x_train":x_train, "x_val":x_val, "x_test":x_test}

def generate_data(x_dict, dgp1, dgp2,dgp3,device, copula='clayton', theta=2.0):
    train_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_train'],device, theta, copula)
    val_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_val'],device, theta, copula)
    test_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_test'],device, theta, copula)
    return train_dict, val_dict, test_dict


def loss_triple(model1, model2, model3, data, copula=None):#estimates the joint loss
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    s3 = model3.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    f3 = model3.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2) + LOG(s3)
        p2 = LOG(f2) + LOG(s1) + LOG(s3)
        p3 = LOG(f3) + LOG(s1) + LOG(s2)
    else:
        
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1), s3.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
        p3 = LOG(f3) + LOG(copula.conditional_cdf("w", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (data['E'] == 0)*1.0
    e2 = (data['E'] == 1)*1.0
    e3 = (data['E'] == 2)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss = -loss/data['E'].shape[0]
    return loss

def single_loss(model, data, event_name='t1'):#estimates loss assuming every thing is observed/no censoring for checking 
    f = model.PDF(data[event_name], data['X'])
    return -torch.mean(LOG(f))



class Exp_linear:
    def __init__(self, bh, nf, device='cpu') -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32).to(device)
        self.coeff = torch.rand((nf,)).to(device)
    
    def hazard(self, t, x):
        device = x.device
        return self.bh.to(device) * torch.exp(torch.matmul(x, self.coeff))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)
    
    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)
    
class Exp_linear_model:
    def __init__(self, bh, nf, device='cpu') -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32).to(device)
        self.coeff = torch.rand((nf,)).to(device)
    
    def hazard(self, t, x):
        device = x.device
        return torch.exp(self.bh.to(device)) * torch.exp(torch.matmul(x, self.coeff))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)
    
    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def enable_grad(self):
        self.bh.requires_grad = True
        self.coeff.requires_grad = True

    def parameters(self):
        return [self.bh, self.coeff]

if __name__ == "__main__":

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Test Coplus Test')

    # 添加参数
    parser.add_argument("--theta_dgp", type=float, default=2.0, help="Description for theta_dgp")
    parser.add_argument("--initial_theta", type=float, default=8.0, help="Description for initial_theta")
    parser.add_argument("--LR_optimal", type=str, default='None', help="Description for LR_optimal")

    args = parser.parse_args()

    print(f"theta_dgp: {args.theta_dgp}, initial_theta: {args.initial_theta}, LR_optimal: {args.LR_optimal}")
    filename = 'experiment.txt'
    # python3 test_script_weijie.py --theta_dgp 2.0 --initial_theta 8.0 --LR_optimal None
    
    theta_dgp = float(args.theta_dgp)
    copula_start_point = float(args.initial_theta)
    LR_optimal = str(args.LR_optimal)
    
    DEVICE = 'cuda:0'#'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    nf = 1
    n_train = 10000
    n_val = 5000
    n_test = 5000
    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    dgp1 = Exp_linear(0.1, nf, device=DEVICE)
    dgp2 = Exp_linear(0.06, nf, device=DEVICE)
    dgp3 = Exp_linear(0.07, nf, device=DEVICE)
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)
    dgp3.coeff = torch.rand((nf,), device=DEVICE)


    copula_dgp = 'clayton'
    # theta_dgp = 2.0
    eps = 1e-4

    train_dict, val_dict, test_dict = \
                generate_data(x_dict, dgp1, dgp2, dgp3,DEVICE, copula_dgp, theta_dgp)
    
    #remove the comment to check the percentage of each event 
    #plt.hist(train_dict['E'])
    #plt.show()
    #assert 0

    #copula for estimation
    # copula_start_point = 8.0
    copula = Clayton(torch.tensor([copula_start_point]).to(DEVICE),eps, DEVICE)
    
    indep_model1 = Exp_linear_model(0.1, nf, device=DEVICE)
    indep_model2 = Exp_linear_model(0.1, nf, device=DEVICE)
    indep_model3 = Exp_linear_model(0.1, nf, device=DEVICE)
    
    indep_model1.enable_grad()
    indep_model2.enable_grad()
    indep_model3.enable_grad()
    copula.enable_grad()
    optimizer = torch.optim.Adam(indep_model3.parameters(), lr=5e-3)

    optimizer = torch.optim.Adam([  {"params": indep_model1.parameters(), "lr": 1e-2},
                                    {"params": indep_model2.parameters(), "lr": 1e-2},
                                    {"params": indep_model3.parameters(), "lr": 1e-2},
                                ])
#     optimizer = torch.optim.SGD(indep_model3.parameters(), lr=5e-3)

#     optimizer = torch.optim.SGD([  {"params": indep_model1.parameters(), "lr": 1e-2},
#                                     {"params": indep_model2.parameters(), "lr": 1e-2},
#                                     {"params": indep_model3.parameters(), "lr": 1e-2},
#                                 ])

    
    if LR_optimal == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-3, max_lr=1e-1, step_size_up=2000, 
                                                      mode='triangular', cycle_momentum=False)
    elif LR_optimal == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=100, epochs=10)
    elif LR_optimal == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    elif LR_optimal == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    
    #add pretraining to make sure its possible to converge to the correct model
    pre_trainn_epochs = 200
    for i in range(pre_trainn_epochs):
        optimizer.zero_grad()
        loss = single_loss(indep_model1, train_dict, 't1')
        loss.backward()
        optimizer.step()
    print("Event 1--> trained model: ", single_loss(indep_model1, val_dict, 't1'), "\tDGP:", single_loss(dgp1, val_dict, 't1'))


    for i in range(pre_trainn_epochs):
        optimizer.zero_grad()
        loss = single_loss(indep_model2, train_dict, 't2')
        loss.backward()
        optimizer.step()
    print("Event 2--> trained model: ", single_loss(indep_model2, val_dict, 't2'), "\tDGP:", single_loss(dgp2, val_dict, 't2'))


    for i in range(pre_trainn_epochs):
        optimizer.zero_grad()
        loss = single_loss(indep_model3, train_dict, 't3')
        loss.backward()
        optimizer.step()
    print("Event 3--> trained model: ", single_loss(indep_model3, val_dict, 't3'), "\tDGP:", single_loss(dgp3, val_dict, 't3'))

    #training loop

    optimizer = torch.optim.Adam([  {"params": indep_model1.parameters(), "lr": 1e-2},
                                    {"params": indep_model2.parameters(), "lr": 1e-2},
                                    {"params": indep_model3.parameters(), "lr": 1e-2},
                                    {"params": [copula.theta], "lr": 1e-3}
                                ])
    n_epochs = 10000
    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_triple(indep_model1, indep_model2, indep_model3, train_dict, copula)
        loss.backward()
        
        copula.theta.grad = copula.theta.grad*1000
        #play with the clip range to see if it makes any differences 
        copula.theta.grad.clamp_(torch.tensor([-0.5]).to(DEVICE), torch.tensor([0.5]).to(DEVICE))
        optimizer.step()
        if LR_optimal in ['CyclicLR', 'OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealingLR']:
            scheduler.step(loss)
        
        if i%500 == 0:
            print(loss, copula.theta)
    train_theta = copula.theta
    print("###############################################################")
    #NLL of all of the events together
    all_nll = loss_triple(indep_model1, indep_model2, indep_model3, val_dict, copula)
    print(all_nll)
    #check the dgp performance
    copula.theta = torch.tensor([theta_dgp]).to(DEVICE)
    dgp_performance = loss_triple(dgp1, dgp2, dgp3, val_dict, copula)
    print(dgp_performance)
    
    
    #NLL assuming every thing is observed, here NLL works correctly
    print("Event 1--> trained model: ", single_loss(indep_model1, val_dict, 't1'), "\tDGP:", single_loss(dgp1, val_dict, 't1'))
    print("Event 2--> trained model: ", single_loss(indep_model2, val_dict, 't2'), "\tDGP:", single_loss(dgp2, val_dict, 't2'))
    print("Event 3--> trained model: ", single_loss(indep_model3, val_dict, 't3'), "\tDGP:", single_loss(dgp3, val_dict, 't3'))
    
    data = {
        'theta_dgp': theta_dgp,
        'inital_theta': copula_start_point,
        'LR_optimal': LR_optimal,
        'train_nll_loss': loss.item(),
        'copula.theta': train_theta.item(),
        'all_nll_loss': all_nll.item(),
        'dgp_performance': dgp_performance.item()
    }
    
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')