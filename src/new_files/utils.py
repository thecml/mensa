import torch 
from pycop import simulation
import numpy as np



def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

#todo: try randn instead of rand
def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {'x_train':x_train, 'x_val':x_val, 'x_test':x_test}

def generate_events(dgp1, dgp2, dgp3, x, device, theta, family):
    if family is None:
        U = torch.rand((x.shape[0], 3))
    else:
        u, v, w = simulation.simu_archimedean(family, 3, x.shape[0], theta)
        u = torch.from_numpy(u).type(torch.float32).reshape(-1, 1)
        v = torch.from_numpy(v).type(torch.float32).reshape(-1, 1)
        w = torch.from_numpy(w).type(torch.float32).reshape(-1, 1)
        U = torch.cat([u,v,w], axis=1)
    t1 = dgp1.rvs(x, U[:,0])
    t2 = dgp2.rvs(x, U[:,1])
    t3 = dgp3.rvs(x, U[:,2])
    T = np.concatenate([t1.reshape(-1,1), t2.reshape(-1,1), t3.reshape(-1,1)], axis=1)
    E = np.argmin(T, axis=1)
    obs_T = T[np.arange(T.shape[0]), E]
    T = torch.from_numpy(T).type(torch.float32)
    E = torch.from_numpy(E).type(torch.float32)
    obs_T = torch.from_numpy(obs_T).type(torch.float32)
    return {'X': x, 'E':E, 'T':obs_T, 't1':t1, 't2':t2, 't3':t3}

def generate_data(x_dict, dgp1, dgp2, dgp3, device, copula, theta):
    train_dict = generate_events(dgp1, dgp2, dgp3, x_dict['x_train'], device, theta, copula)
    val_dict = generate_events(dgp1, dgp2, dgp3, x_dict['x_val'], device, theta, copula)
    test_dict = generate_events(dgp1, dgp2, dgp3, x_dict['x_test'], device, theta, copula)
    return train_dict, val_dict, test_dict


def loss_triple(model1, model2, model3, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    s3 = model3.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    f3 = model3.PDF(data['T'], data['X'])

    if copula is None:
        p1 = safe_log(f1) + safe_log(s2) + safe_log(s3)
        p2 = safe_log(s1) + safe_log(f2) + safe_log(s3)
        p3 = safe_log(s1) + safe_log(s2) + safe_log(f3)
    
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1), s3.reshape(-1,1)], dim=1)#todo: clamp removed!!!!!!!
        p1 = safe_log(f1) + safe_log(copula.conditional_cdf("u", S))
        p2 = safe_log(f2) + safe_log(copula.conditional_cdf("v", S))
        p3 = safe_log(f3) + safe_log(copula.conditional_cdf("w", S))


    #todo: not sure if its a good idea?????
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (data['E'] == 0) * 1.0
    e2 = (data['E'] == 1) * 1.0
    e3 = (data['E'] == 2) * 1.0
    loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
    loss = -loss/data['E'].shape[0]
    return loss


def loss_triple_cuda(model1, model2, model3, T, X, E, copula=None):
    s1 = model1.survival(T, X)
    s2 = model2.survival(T, X)
    s3 = model3.survival(T, X)
    f1 = model1.PDF(T, X)
    f2 = model2.PDF(T, X)
    f3 = model3.PDF(T, X)

    if copula is None:
        p1 = safe_log(f1) + safe_log(s2) + safe_log(s3)
        p2 = safe_log(s1) + safe_log(f2) + safe_log(s3)
        p3 = safe_log(s1) + safe_log(s2) + safe_log(f3)
    
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1), s3.reshape(-1,1)], dim=1)#todo: clamp removed!!!!!!!
        p1 = safe_log(f1) + safe_log(copula.conditional_cdf("u", S))
        p2 = safe_log(f2) + safe_log(copula.conditional_cdf("v", S))
        p3 = safe_log(f3) + safe_log(copula.conditional_cdf("w", S))


    #todo: not sure if its a good idea?????
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (E == 0) * 1.0
    e2 = (E == 1) * 1.0
    e3 = (E == 2) * 1.0
    loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
    loss = -loss/E.shape[0]
    return loss



def loss_tripleNDE(model, X, T, E, copula=None):
    log_f = model.forward_f(X, T.reshape(-1,1))
    log_s = model.forward_S(X, T.reshape(-1,1), mask=0)
    f1 = log_f[:,0:1]
    f2 = log_f[:,1:2]
    f3 = log_f[:,2:3]
    s1 = log_s[:,0:1]
    s2 = log_s[:,1:2]
    s3 = log_s[:,2:3]
    #todo: maybe inverse weighting can help!!!!!!

    if copula is None:
        p1 = f1 + s2 + s3
        p2 = s1 + f2 + s3
        p3 = s1 + s2 + f3
    else:#todo: check dims, why reshape?????
        S = torch.cat([torch.exp(s1).reshape(-1,1), torch.exp(s2).reshape(-1,1), torch.exp(s3).reshape(-1,1)], dim=1)#todo: clamp removed!!!!!!!
        p1 = f1 + safe_log(copula.conditional_cdf("u", S)).reshape(-1,1)
        p2 = f2 + safe_log(copula.conditional_cdf("v", S)).reshape(-1,1)
        p3 = f3 + safe_log(copula.conditional_cdf("w", S)).reshape(-1,1)
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (E == 0) * 1.0
    e2 = (E == 1) * 1.0
    e3 = (E == 2) * 1.0
    loss = torch.sum(e1.reshape(-1,1) * p1) + torch.sum(e2.reshape(-1,1) * p2) + torch.sum(e3.reshape(-1,1) * p3)
    loss = -loss/E.shape[0]
    return loss

