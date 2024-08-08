import torch 
import numpy as np 
import torch.nn as nn 

def conditional_weibull_loss(model, x, t, E, elbo=True, copula=None):

    alpha = model.discount
    params = model.forward(x)

    t = t.reshape(-1,1).expand(-1, model.k)#(n, k)
    f_risks = []
    s_risks = []

    for i in range(model.risks):
        k = params[i][0]
        b = params[i][1]
        gate = nn.Softmax(dim=1)(params[i][2])
        s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
        f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
        f = f + s
        s = (s * gate).sum(dim=1)#log_survival
        f = (f * gate).sum(dim=1)#log_density
        f_risks.append(f)#(n,3) each column for one risk
        s_risks.append(s)
    f = torch.stack(f_risks, dim=1)
    s = torch.stack(s_risks, dim=1)

    if model.risks == 3:
        
        if copula is None:
            p1 = f[:,0] + s[:,1] + s[:,2] 
            p2 = s[:,0] + f[:,1] + s[:,2]
            p3 = s[:,0] + s[:,1] + f[:,2]
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
            loss = -loss/E.shape[0]
        else:
            S = torch.exp(s).clamp(0.001,0.999)
            p1 = f[:,0] + safe_log(copula.conditional_cdf("u", S))
            p2 = f[:,1] + safe_log(copula.conditional_cdf("v", S))
            p3 = f[:,2] + safe_log(copula.conditional_cdf("w", S))
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
            loss = -loss/E.shape[0]
    elif model.risks == 2:
        p1 = f[:,0] + s[:,1] 
        p2 = s[:,0] + f[:,1] 
        e1 = (E == 1) * 1.0 #event
        e2 = (E == 0) * 1.0#censoring
        loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) 
        loss = -loss/E.shape[0]
    elif model.risks == 1:#added single risk 
        e1 = (E == 1) * 1.0
        e2 = (E == 0) * 1.0
        loss = torch.sum(e1 * f[:,0]) + torch.sum(e2 * s[:,0]) 
        loss = -loss/E.shape[0]
    return loss
    




"""k1 = params[0][0]#(n, k)
  b1 = params[0][1]
  gate1 = nn.Softmax(dim=1)(params[0][2])

  s1 = - (torch.pow(torch.exp(b1)*t, torch.exp(k1)))
  f1 = k1 + b1 + ((torch.exp(k1)-1)*(b1+torch.log(t)))
  f1 = f1 + s1
  s1 = (s1 * gate1).sum(dim=1)
  f1 = (f1 * gate1).sum(dim=1)

  k2 = params[1][0]
  b2 = params[1][1]
  gate2 = nn.Softmax(dim=1)(params[1][2])
  

  s2 = - (torch.pow(torch.exp(b2)*t, torch.exp(k2)))
  f2 = k2 + b2 + ((torch.exp(k2)-1)*(b2+torch.log(t)))
  f2 = f2 + s2

  s2 = (s2 * gate2).sum(dim=1)
  f2 = (f2 * gate2).sum(dim=1)

  k3 = params[2][0]
  b3 = params[2][1]
  gate3 = nn.Softmax(dim=1)(params[2][2])

  s3 = - (torch.pow(torch.exp(b3)*t, torch.exp(k3)))
  f3 = k3 + b3 + ((torch.exp(k3)-1)*(b3+torch.log(t)))
  f3 = f3 + s3

  s3 = (s3 * gate3).sum(dim=1)
  f3 = (f3 * gate3).sum(dim=1)

  p1 = f1 + s2 + s3
  p2 = s1 + f2 + s3
  p3 = s1 + s2 + f3
  
  e1 = (E == 0) * 1.0
  e2 = (E == 1) * 1.0
  e3 = (E == 2) * 1.0
  loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
  loss = -loss/E.shape[0]
  return loss
"""


  

def triple_weibull_loss(model, X, T, E):
   e1 = (E==0)*1.0
   e2 = (E==1)*1.0
   e3 = (E==2)*1.0
   l1 = conditional_weibull_loss(model, X, T, e1, risk='1') 
   l2 = conditional_weibull_loss(model, X, T, e2, risk='2') 
   l3 = conditional_weibull_loss(model, X, T, e3, risk='3')
   print(l1,l2,l3)
   return l1+l2+l3

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