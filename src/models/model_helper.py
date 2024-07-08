from dgp import Weibull_linear, Weibull_nonlinear, Weibull_log_linear, Exp_linear, EXP_nonlinear, LogNormal_linear, LogNormal_nonlinear, LogNormalCox_linear
import torch

def get_model_from_name(n_features = 10, number_model = 2, model_type = 'Weibull_linear', device = 'cpu', dtype = torch.float64):
    if model_type == 'Weibull_log_linear':
        models = [Weibull_log_linear(n_features, 2, 1, device, dtype) for i in range(number_model)]
    elif model_type == 'Weibull_linear':#not working
        models = [Weibull_linear(n_features, alpha=2, gamma=1, beta=1, device=device, dtype=dtype) for i in range(number_model)]
    elif model_type == 'Weibull_nonlinear':#not working
        models = [Weibull_nonlinear(n_features, alpha=2, gamma=1, beta=1, device=device, dtype=dtype) for i in range(number_model)]
    elif model_type == 'Exp_linear':
        models = [Exp_linear(nf=n_features, bh=0.5, device=device, dtype=dtype) for i in range(number_model)]
    elif model_type == 'EXP_nonlinear':
        models = [EXP_nonlinear(nf=n_features, bh=0.5, hd=1, device=device, dtype=dtype) for i in range(number_model)]
    elif model_type == 'LogNormal_linear':
        models = [LogNormal_linear(n_features, device) for i in range(number_model)]
    elif model_type == 'LogNormal_nonlinear': 
        models = [LogNormal_nonlinear(n_features, 4, device = device) for i in range(number_model)]
    elif model_type == 'LogNormalCox_linear':#not working
        models = [LogNormalCox_linear(nf=n_features, mu=2, sigma=1, device = device) for i in range(number_model)]   
    else:
        raise Exception("Model_type:", model_type, 'incorrect.')
    return models

def get_model_best_params(models, model_type):
    best_params = [{} for i in range(len(models))]
    if model_type == 'Weibull_log_linear':
        for i in range(len(models)):
            best_params[i] = {
                'coeff': models[i].coeff.detach().clone(), 
                'mu': models[i].mu.detach().clone(), 
                'sigma': models[i].sigma.detach().clone()
            }
        # best_c1 = model1.coeff.detach().clone()
        # best_c2 = model2.coeff.detach().clone()
        # best_mu1 = model1.mu.detach().clone()
        # best_mu2 = model2.mu.detach().clone()
        # best_sig1 = model1.sigma.detach().clone()
        # best_sig2 = model2.sigma.detach().clone()
    elif model_type == 'Weibull_linear':
        for i in range(len(models)):
            best_params[i] = {
                'alpha': models[i].alpha.detach().clone(), 
                'gamma': models[i].gamma.detach().clone(), 
                'beta': models[i].beta.detach().clone()
            }
    elif model_type == 'Weibull_nonlinear':
        for i in range(len(models)):
            best_params[i] = {
                'alpha': models[i].alpha.detach().clone(), 
                'gamma': models[i].gamma.detach().clone(), 
                'beta': models[i].beta.detach().clone()
            }
    elif model_type == 'LogNormal_linear':        
        for i in range(len(models)):
            best_params[i] = {
                'mu_coeff': models[i].mu_coeff.detach().clone(), 
                'sigma_coeff': models[i].sigma_coeff.detach().clone(), 
            }
    elif model_type == 'LogNormal_nonlinear':
        for i in range(len(models)):
            best_params[i] = {
                'beta': models[i].beta.detach().clone(), 
                'mu_coeff': models[i].mu_coeff.detach().clone(), 
                'sigma_coeff': models[i].sigma_coeff.detach().clone(), 
            }                
    elif model_type == 'LogNormalCox_linear':
        for i in range(len(models)):
            best_params[i] = {
                'mu': models[i].mu.detach().clone(), 
                'sigma': models[i].sigma.detach().clone(),
                'coeff': models[i].coeff.detach().clone(), 
            }
    elif model_type == 'Exp_linear':
        for i in range(len(models)):
            best_params[i] = {
                'bh': models[i].bh.detach().clone(), 
                'coeff': models[i].coeff.detach().clone(), 
            }
    elif model_type == 'EXP_nonlinear':
        for i in range(len(models)):
            best_params[i] = {
                'bh': models[i].bh.detach().clone(), 
                'beta': models[i].beta.detach().clone(), 
                'coeff': models[i].coeff.detach().clone(), 
            }
    else:
        raise Exception("Model_type:", model_type, 'incorrect.')
    return best_params

def set_model_best_params(models, best_params, model_type='Weibull_log_linear'):
    if model_type == 'Weibull_log_linear':
        for i in range(len(models)):
            models[i].mu = best_params[i]['mu']
            models[i].sigma = best_params[i]['sigma']
            models[i].coeff = best_params[i]['coeff']
        # best_c1 = model1.coeff.detach().clone()
        # best_c2 = model2.coeff.detach().clone()
        # best_mu1 = model1.mu.detach().clone()
        # best_mu2 = model2.mu.detach().clone()
        # best_sig1 = model1.sigma.detach().clone()
        # best_sig2 = model2.sigma.detach().clone()
    elif model_type == 'Weibull_linear':
        for i in range(len(models)):
            models[i].alpha = best_params[i]['alpha']
            models[i].gamma = best_params[i]['gamma']
            models[i].beta = best_params[i]['beta']
    elif model_type == 'Weibull_nonlinear':
        for i in range(len(models)):
            models[i].alpha = best_params[i]['alpha']
            models[i].gamma = best_params[i]['gamma']
            models[i].beta = best_params[i]['beta']            
    elif model_type == 'LogNormal_linear':
        for i in range(len(models)):
            models[i].mu_coeff = best_params[i]['mu_coeff']
            models[i].sigma_coeff = best_params[i]['sigma_coeff']
    elif model_type == 'LogNormal_nonlinear':
        for i in range(len(models)):
            models[i].beta = best_params[i]['beta']
            models[i].mu_coeff = best_params[i]['mu_coeff']
            models[i].sigma_coeff = best_params[i]['sigma_coeff']
    elif model_type == 'LogNormalCox_linear':
        for i in range(len(models)):
            models[i].mu = best_params[i]['mu']
            models[i].sigma = best_params[i]['sigma']
            models[i].coeff = best_params[i]['coeff']
    elif model_type == 'Exp_linear':
        for i in range(len(models)):
            models[i].bh = best_params[i]['bh']
            models[i].coeff = best_params[i]['coeff']
    elif model_type == 'EXP_nonlinear':
        for i in range(len(models)):
            models[i].bh = best_params[i]['bh']
            models[i].beta = best_params[i]['beta']
            models[i].coeff = best_params[i]['coeff']
    else:
        raise Exception("Model_type:", model_type, 'incorrect.')
    return models