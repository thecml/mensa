from distributions import (Weibull_linear, Weibull_nonlinear, Weibull_log_linear, Exp_linear,
                           EXP_nonlinear, LogNormal_linear, LogNormal_nonlinear, LogNormalCox_linear)
import torch

def get_model_from_name(n_features=10, number_model=2, model_type='Weibull_linear',
                        device='cpu', dtype=torch.float64):
    if model_type == 'Weibull_log_linear':
        models = [Weibull_log_linear(n_features, device=device) for i in range(number_model)]
    elif model_type == 'Weibull_linear':
        models = [Weibull_linear(n_features, device=device) for i in range(number_model)]
    elif model_type == 'Weibull_nonlinear':
        models = [Weibull_nonlinear(n_features, n_hidden=4, device=device) for i in range(number_model)]
    elif model_type == 'Exp_linear':
        models = [Exp_linear(n_features, device=device) for i in range(number_model)]
    elif model_type == 'EXP_nonlinear':
        models = [EXP_nonlinear(n_features, n_hidden=4, device=device) for i in range(number_model)]
    elif model_type == 'LogNormal_linear':
        models = [LogNormal_linear(n_features, device=device) for i in range(number_model)]
    elif model_type == 'LogNormal_nonlinear': 
        models = [LogNormal_nonlinear(n_features, n_hidden=4, device=device) for i in range(number_model)]
    elif model_type == 'LogNormalCox_linear': #TODO: Not working
        models = [LogNormalCox_linear(n_features, device=device) for i in range(number_model)]   
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
    elif model_type == 'Weibull_linear':
        for i in range(len(models)):
            best_params[i] = {
                'alpha': models[i].alpha.detach().clone(), 
                'gamma': models[i].gamma.detach().clone(), 
                'coeff': models[i].coeff.detach().clone()
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

def set_model_best_params(models, best_params, model_type):
    if model_type == 'Weibull_log_linear':
        for i in range(len(models)):
            models[i].mu = best_params[i]['mu']
            models[i].sigma = best_params[i]['sigma']
            models[i].coeff = best_params[i]['coeff']
    elif model_type == 'Weibull_linear':
        for i in range(len(models)):
            models[i].alpha = best_params[i]['alpha']
            models[i].gamma = best_params[i]['gamma']
            models[i].coeff = best_params[i]['coeff']
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

def map_model_name(model_name):
    if model_name == "dgp":
        model_name = "DGP"
    elif model_name == "dsm":
        model_name = "DSM"
    elif model_name == "deephit":
        model_name = "DeepHit"
    elif model_name == "deepsurv":
        model_name = "DeepSurv"
    elif model_name == "hierarch":
        model_name = "Hierarch."
    elif model_name == "mtlrcr":
        model_name = "MTLR-CR"
    elif model_name == "mensa":
        model_name = "MENSA"
    else:
        pass
    return model_name