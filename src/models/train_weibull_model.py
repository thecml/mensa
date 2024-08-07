import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
import torch
import random
import warnings
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from utility.config import load_config
from utility.survival import compute_l1_difference, predict_survival_function
from SurvivalEVAL.Evaluations.util import predict_median_survival_time
from SurvivalEVAL.Evaluations.MeanError import mean_error
from utility.loss import single_loss, double_loss
from data_loader import *
from copula import Convex_bivariate, Clayton_Bivariate
from distributions import Weibull_log_linear

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

# Define test parameters
DATASET_VERSIONS = ["linear"]
COPULA_NAMES = ["clayton"] 
#KENDALL_TAUS = np.arange(0, 0.9, 0.1)
KENDALL_TAUS = [0.5]
MODELS = ["weibull-nocop", "weibull-cop"]
N_SAMPLES = 10000
N_FEATURES = 10

def calculate_loss_two_models(model1, model2, data, copula=None):
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

def train_mensa_model_2_events(train_dict, valid_dict, model1, model2, copula, n_epochs=1000, lr=0.01):
    model1.enable_grad()
    model2.enable_grad()
    copula.enable_grad()
    optimizer = torch.optim.Adam([{"params": model1.parameters(), "lr": lr},
                                  {"params": model2.parameters(), "lr": lr},
                                  {"params": copula.parameters(), "lr": lr}])
    
    min_val_loss = 1000
    for itr in range(n_epochs):
        optimizer.zero_grad()
        loss = calculate_loss_two_models(model1, model2, train_dict, copula)
        loss.backward()

        for p in copula.parameters():
            p.grad = p.grad * 100
            p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))

        optimizer.step()

        for p in copula.parameters():
            if p <= 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)

        with torch.no_grad():
            val_loss = calculate_loss_two_models(model1, model2, valid_dict, copula)
            
            if itr % 100 == 0:
                print(f"{val_loss} - {copula.parameters()}")
            
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:
                    break
                
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    
    return model1, model2, copula

def generate_events(dgp1, dgp2, dgp3, x, device,theta=2.0, family='clayton'):
    if family is None:
        uv = torch.randn((x.shape[0], 3))#sample idnependent 
    else:
        u,v,w = simulation.simu_archimedean(family, 3, x.shape[0], theta=theta)
        u = torch.from_numpy(u).type(torch.float32).reshape(-1,1)
        v = torch.from_numpy(v).type(torch.float32).reshape(-1,1)
        w = torch.from_numpy(w).type(torch.float32).reshape(-1,1)
        
        uv = torch.cat([u,v,w], axis=1)

    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    t3 = dgp3.rvs(x, uv[:,2])
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

def generate_data(x_dict, dgp1, dgp2,dgp3,device, copula='clayton', theta=2.0):
    train_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_train'],device, theta, copula)
    val_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_val'],device, theta, copula)
    test_dict = generate_events(dgp1, dgp2,dgp3, x_dict['x_test'],device, theta, copula)
    return train_dict, val_dict, test_dict

if __name__ == "__main__":
    DEVICE = 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    nf = 10
    n_train = 10000
    n_val = 5000
    n_test = 5000
    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    dgp1 = DGP_Weibull_linear(nf, alpha=17, gamma=3, device=DEVICE)
    dgp2 = DGP_Weibull_linear(nf, alpha=16, gamma=3, device=DEVICE)
    dgp3 = DGP_Weibull_linear(nf, alpha=17, gamma=4, device=DEVICE)

    """dgp1 = Exp_linear(0.1, nf)
    dgp2 = Exp_linear(0.09, nf)
    dgp3 = Exp_linear(0.06, nf)"""
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)
    dgp3.coeff = torch.rand((nf,), device=DEVICE)


    copula_dgp = 'clayton'
    theta_dgp = 8.0
    eps = 1e-4
    
    train_dict, valid_dict, test_dict = \
                generate_data(x_dict, dgp1, dgp2, dgp3,DEVICE, copula_dgp, theta_dgp)
    
    print(f"Goal theta: {kendall_tau_to_theta('clayton', 0.25)}")
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=train_dict['E']) # Use first event for time bins
    
    # Make models
    eps = 1e-4
    n_features = 10
    copula_start_point = 2.0
    #copula = Convex_bivariate(copulas=['cl'], dtype=dtype, device=device)
    copula = Clayton_Bivariate(2.0, 1e-3, dtype, device)
    model1 = Weibull_log_linear(n_features, device, dtype)
    model2 = Weibull_log_linear(n_features, device, dtype)
    model1, model2, copula = train_mensa_model_2_events(train_dict, valid_dict, model1, model2,
                                                        copula, n_epochs=5000, lr=0.01)

    # Print NLL of all events together
    print(f"NLL all events: {double_loss(model1, model2, valid_dict, copula)}")
    
    # Check the dgp performance
    #copula.theta = torch.tensor([5.0])
    print(f"DGP loss: {double_loss(dgp1, dgp1, valid_dict, copula)}")

    # Evaluate the L1
    preds_e1 = predict_survival_function(model1, test_dict['X'], time_bins).detach().numpy()
    preds_e2 = predict_survival_function(model2, test_dict['X'], time_bins).detach().numpy()
    #preds_c = predict_survival_function(model3, test_dict['X'], time_bins).detach().numpy()
    
    n_samples = test_dict['X'].shape[0]
    truth_preds_e1 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    for i in range(time_bins.shape[0]):
        truth_preds_e1[:,i] = dgp1.survival(time_bins[i], test_dict['X'])
    l1_e1 = float(compute_l1_difference(truth_preds_e1, preds_e1, n_samples, steps=time_bins))
        
    truth_preds_e2 = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    for i in range(time_bins.shape[0]):
        truth_preds_e2[:,i] = dgp2.survival(time_bins[i], test_dict['X'])
    l1_e2 = float(compute_l1_difference(truth_preds_e2, preds_e2, n_samples, steps=time_bins))

    #truth_preds_c = torch.zeros((n_samples, time_bins.shape[0]), device=device)
    #for i in range(time_bins.shape[0]):
    #    truth_preds_c[:,i] = dgps[2].survival(time_bins[i], test_dict['X'])
    #l1_c = float(compute_l1_difference(truth_preds_c, preds_c, n_samples, steps=time_bins))
    
    print(f"L1 E1: {l1_e1}")
    print(f"L1 E2: {l1_e2}")
    #print(f"L1 C: {l1_c}")
    
    for event_id, surv_preds in enumerate([preds_e1, preds_e2]):
        surv_preds = pd.DataFrame(surv_preds, columns=time_bins.numpy())
        y_test_time = test_dict['T']
        y_test_event = (test_dict['E'] == event_id)*1.0
        y_train_time = train_dict['T']
        y_train_event = (train_dict['E'] == event_id)*1.0
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        mae_hinge = lifelines_eval.mae(method="Hinge")
        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
        median_survs = lifelines_eval.predict_time_from_curve(predict_median_survival_time)
        event_indicators = np.array([1] * len(y_test_time))
        true_mae = float(mean_error(median_survs, y_test_time, event_indicators, method='Uncensored'))
        metrics = [ci, ibs, mae_hinge, mae_pseudo, true_mae]
        print(metrics)
                