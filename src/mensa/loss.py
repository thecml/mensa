import torch

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def calculate_loss_one_model(model, data, event_name='T1'):
    """
    Calculates loss assuming everything is observed/no censoring
    """
    f = model.PDF(data[event_name], data['X'])
    return -torch.mean(LOG(f))

def calculate_loss_two_models(model1, model2, data, copula=None):
    """
    Calculates loss for two models in singe event scenario
    """
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    return -torch.mean(p1 * data['E'] + (1-data['E'])*p2)

def calculate_loss_three_models(model1, model2, model3, data, copula=None):
    """
    Calculates loss for three models in competing risks (K=3) scenario
    """
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

def calculate_loss_three_models_me(model1, model2, model3, data, copula=None):
    """
    Calculates loss for thee models in multi event scenario
    """
    s1 = model1.survival(data['T'][:,0], data['X'])
    s2 = model2.survival(data['T'][:,1], data['X'])
    s3 = model3.survival(data['T'][:,2], data['X'])
    f1 = model1.PDF(data['T'][:,0], data['X'])
    f2 = model2.PDF(data['T'][:,1], data['X'])
    f3 = model3.PDF(data['T'][:,2], data['X'])
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
    e1 = data['E'][:,0]
    e2 = data['E'][:,1]
    e3 = data['E'][:,2]
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss = -loss/data['E'].shape[0]
    return loss