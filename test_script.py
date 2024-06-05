import numpy as np 
import torch





def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def generate_events(dgp1, dgp2, dgp3, x, device,copula=None):
    if copula is None:
        uv = torch.rand((x.shape[0],3), device=device)
    else:
        uv = copula.rvs(x.shape[0])
    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    t3 = dgp3.rvs(x, uv[:,0])
    T = np.concatenate([t1.reshape(-1,1),t2.reshape(-1,1),t3.reshape(-1,1)], axis=1)
    E = np.argmin(T,axis=1)
    obs_T = T[np.arange(T.shape[0]), E]
    T = torch.from_numpy(T).type(torch.float32)
    E = torch.from_numpy(E).type(torch.float32)
    obs_T = torch.from_numpy(obs_T).type(torch.float32)

    return {'X':x,'E':E, 'T':obs_T, 't1':t1, 't2':t2, 't3':t3}

def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {"x_train":x_train, "x_val":x_val, "x_test":x_test}

def generate_data(x_dict, dgp1, dgp2,dgp3,device, copula=None):
    train_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_train'],device, copula)
    val_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_val'],device, copula)
    test_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_test'],device, copula)
    return train_dict, val_dict, test_dict


def loss_triple(model1, model2, model3, data, copula=None):
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
        
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (data['E'] == 0)*1.0
    e2 = (data['E'] == 1)*1.0
    e3 = (data['E'] == 2)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss = -loss/data['E'].shape[0]
    return loss

def single_loss(model, data, event_name='t1'):
    f = model.PDF(data[event_name], data['X'])
    return -torch.mean(LOG(f))

def single_loss_2(model, data, e):
    s = model.survival(data['T'], data['X'])
    f = model.PDF(data['T'], data['X'])
    
    E = (data['E'] == e).type(torch.float32)
    
    return -torch.mean((E * LOG(f)) + ((1-E)*LOG(s)))

class Exp_linear:
    def __init__(self, bh, nf) -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32)
        self.coeff = torch.rand((nf,))
    
    def hazard(self, t, x):
        return self.bh * torch.exp(torch.matmul(x, self.coeff))
    
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
    def __init__(self, bh, nf) -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32)
        self.coeff = torch.rand((nf,))
    
    def hazard(self, t, x):
        return torch.exp(self.bh) * torch.exp(torch.matmul(x, self.coeff))
    
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

    DEVICE = 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    nf = 5
    n_train = 10000
    n_val = 5000
    n_test = 5000
    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    dgp1 = Exp_linear(0.1, nf)
    dgp2 = Exp_linear(0.04, nf)
    dgp3 = Exp_linear(0.07, nf)
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)
    dgp3.coeff = torch.rand((nf,), device=DEVICE)


    copula_dgp = None

    train_dict, val_dict, test_dict = \
                generate_data(x_dict, dgp1, dgp2, dgp3,DEVICE, copula_dgp)
    
    indep_model1 = Exp_linear_model(0.1, nf)
    indep_model2 = Exp_linear_model(0.1, nf)
    indep_model3 = Exp_linear_model(0.1, nf)

    indep_model1.enable_grad()
    indep_model2.enable_grad()
    indep_model3.enable_grad()

    model_optimizer = torch.optim.Adam( list(indep_model1.parameters())+list(indep_model2.parameters() + list(indep_model3.parameters())), lr=1e-3, weight_decay=0.01)
    for i in range(10000):
        model_optimizer.zero_grad()
        loss = loss_triple(indep_model1, indep_model2, indep_model3, train_dict)
        loss.backward()
        model_optimizer.step()
        print(loss)

    print(loss_triple(dgp1, dgp2, dgp3, val_dict))
    print(loss_triple(indep_model1, indep_model2, indep_model3, val_dict))
    print(single_loss(dgp1, val_dict, 't1'), single_loss(indep_model1, val_dict, 't1'))
    print(single_loss(dgp2, val_dict, 't2'), single_loss(indep_model2, val_dict, 't2'))
    print(single_loss(dgp3, val_dict, 't3'), single_loss(indep_model3, val_dict, 't3'))

    print(single_loss_2(dgp1, val_dict, 0), single_loss_2(indep_model1, val_dict, 0))
    print(single_loss_2(dgp2, val_dict, 1), single_loss_2(indep_model2, val_dict, 1))
    print(single_loss_2(dgp3, val_dict, 2), single_loss_2(indep_model3, val_dict, 2))
    
