import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
import torch
import random
import warnings
from data_loader import *
from utility.survival import preprocess_data
from utility.data import dotdict
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loader
from copula import Clayton
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from tqdm import tqdm
from SurvivalEVAL.Evaluator import LifelinesEvaluator
import copy
from dgp import Weibull_linear

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class Weibull_log_linear:
    def __init__(self, nf, mu, sigma, device) -> None:
        #torch.manual_seed(0)
        self.nf = nf
        self.mu = torch.tensor([mu], device=device).type(torch.float32)
        self.sigma = torch.tensor([sigma], device=device).type(torch.float32)
        self.coeff = torch.rand((nf,), device=device).type(torch.float32)
    
    def survival(self,t,x):
        return torch.exp(-1*torch.exp((LOG(t)-self.mu-torch.matmul(x, self.coeff))/torch.exp(self.sigma)))
    
    def cum_hazard(self, t,x):
        return torch.exp((LOG(t)-self.mu-torch.matmul(x, self.coeff))/torch.exp(self.sigma))
    
    def hazard(self, t,x):
        return self.cum_hazard(t,x)/(t*torch.exp(self.sigma))
    
    def PDF(self,t,x):
        return self.survival(t,x) * self.hazard(t,x)
    
    def CDF(self, t,x ):
        return 1 - self.survival(t,x)
    
    def enable_grad(self):
        self.sigma.requires_grad = True
        self.mu.requires_grad = True
        self.coeff.requires_grad = True
    
    def parameters(self):
        return [self.sigma, self.mu, self.coeff]
    
    def rvs(self, x, u):
        tmp = LOG(-1*LOG(u))*torch.exp(self.sigma)
        tmp1 = torch.matmul(x, self.coeff) + self.mu
        return torch.exp(tmp+tmp1)

def array_to_tensor(array, dtype=None, device='cpu'):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array_c = array.copy()
    tensor = torch.tensor(array_c, dtype=dtype).to(device)
    return tensor

def predict_survival_curve(model, x_test, time_bins, truth=False):
    device = torch.device("cpu")
    if truth == False:
        model = copy.deepcopy(model).to(device)
    surv_estimate = torch.zeros((x_test.shape[0], time_bins.shape[0]), device=device)
    x_test = torch.tensor(x_test)
    time_bins = torch.tensor(time_bins)
    for i in range(time_bins.shape[0]):
        surv_estimate[:,i] = model.survival(time_bins[i], x_test)
    return surv_estimate, time_bins, time_bins.max()

def loss_function(model1, model2, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
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

def independent_train_loop_linear(model1, model2, train_data, val_data,
                                  test_data, n_itr, optimizer1='Adam', optimizer2='Adam',
                                  lr1=1e-3, lr2=1e-3, sub_itr=5, verbose=False):
    train_loss_log = []
    val_loss_log = []
    copula_log = torch.zeros((n_itr,))
    model1.enable_grad()
    model2.enable_grad()
    
    copula_grad_log = []
    mu_grad_log = [[], []]
    sigma_grad_log = [[], []]
    coeff_grad_log = [[], []]
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    stop_itr = 0
    if optimizer1 == 'Adam':
        model_optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr1, weight_decay=0.0)
        
    for itr in tqdm(range(n_itr)):
        model_optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, None)
        loss.backward()
        model_optimizer.step() 
        train_loss_log.append(loss.detach().clone())
    ##########################
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, None)
            val_loss_log.append(val_loss.detach().clone())
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr =0
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
    return model1, model2

if __name__ == "__main__":
    # Load data
    df, params = SingleEventSyntheticDataLoader(copula_name="Frank",
                                           n_features=10,
                                           n_samples=1000)
    X = df.drop(['observed_time', 'event_indicator',
                 'event_time', 'censoring_time'], axis=1)
    y = convert_to_structured(df['observed_time'].values,
                              df['event_indicator'].values)
    cat_features = []
    num_features = list(X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25,
                                                          random_state=0)
    # Make time bins
    time_bins = make_time_bins(y_train['time'], event=y_train['event'])

    # Scale data
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test,
                                               cat_features, num_features, as_array=True)
    
    # Format data
    times_tensor_train = array_to_tensor(y_train['time'], torch.float32)
    event_indicator_tensor_train = array_to_tensor(y_train['event'], torch.float32)
    covariate_tensor_train = torch.tensor(X_train).to(device)
    times_tensor_val = array_to_tensor(y_valid['time'], torch.float32)
    event_indicator_tensor_val = array_to_tensor(y_valid['event'], torch.float32)
    covariate_tensor_val = torch.tensor(X_valid).to(device)
    
    # Make models
    n_features = len(num_features+cat_features)
    indep_model1 = Weibull_log_linear(n_features, 4, 14, device)
    indep_model2 = Weibull_log_linear(n_features, 4, 14, device)
    
    # Train
    train_dict = dict()
    train_dict['X'] = torch.tensor(X_train, dtype=torch.float32)
    train_dict['T'] = torch.tensor(y_train['time'].copy(), dtype=torch.float32)
    train_dict['E'] = torch.tensor(y_train['event'].copy(), dtype=torch.float32)
    valid_dict = dict()
    valid_dict['X'] = torch.tensor(X_valid, dtype=torch.float32)
    valid_dict['T'] = torch.tensor(y_valid['time'].copy(), dtype=torch.float32)
    valid_dict['E'] = torch.tensor(y_valid['event'].copy(), dtype=torch.float32)
    test_dict = dict()
    test_dict['X'] = torch.tensor(X_test, dtype=torch.float32)
    test_dict['T'] = torch.tensor(y_test['time'].copy(), dtype=torch.float32)
    test_dict['E'] = torch.tensor(y_test['event'].copy(), dtype=torch.float32)
    indep_model1, indep_model2 = independent_train_loop_linear(indep_model1,
                                                               indep_model2,
                                                               train_dict,
                                                               valid_dict,
                                                               test_dict, 15000) # 3000
    
    # Evaluate
    surv_pred, _, _ = predict_survival_curve(indep_model1, X_test.astype('float32'),
                                             time_bins, truth=True)
    surv_pred = pd.DataFrame(surv_pred, columns=np.array(time_bins))
    lifelines_eval = LifelinesEvaluator(surv_pred.T, y_test['time'], y_test['event'],
                                        y_train['time'], y_train['event'])
    ci = lifelines_eval.concordance()[0]
    mae_hinge = lifelines_eval.mae(method="Hinge")
    ibs = lifelines_eval.integrated_brier_score()
    print(f"Ali: CI={round(ci, 2)} - MAE={round(mae_hinge, 2)} - IBS={round(ibs, 2)}")