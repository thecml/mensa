import uuid
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import config as cfg
from utility.survival import make_time_bins
import torch
import random
import warnings
from data_loader import *
from utility.survival import preprocess_data
from utility.data import dotdict, array_to_tensor
import torch.optim as optim
import torch.nn as nn
from utility.survival import convert_to_structured
from dcsurvival.dirac_phi import DiracPhi
from dcsurvival.survival import DCSurvival
from tqdm import tqdm
from SurvivalEVAL.Evaluator import LifelinesEvaluator
import copy
from torch.utils.data import DataLoader, TensorDataset
from mensa.model import MensaNDE
from utility.config import load_config
from utility.survival import predict_survival_function, compute_l1_difference
from copula import Clayton2D, Frank2D

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

SHAPE_SCALE = 1
SCALE_SCALE = 1

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class Net(nn.Module):
    """
    Network architecture of DeepWeiSurv
    """

    def __init__(self, input, output):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.1)
        self.linear6 = nn.Linear(8, output)
        self.linear7 = nn.Linear(8, output)
        self.elu = nn.Softplus(output)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = self.dropout2(x)
        beta = self.elu(self.linear6(x))
        eta = self.elu(self.linear7(x))
        return beta, eta
    
class Data(Dataset):
    
    def __init__(self):
        self.X = X_train
        self.Y = y_train
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len

def neg_log_likelihood_loss(beta, eta, target):
    """

    :param beta:
    :param eta:
    :param target:
    :return:
    """
    status = target[0]
    time = target[1]
    beta = torch.mul(beta, SHAPE_SCALE)
    eta = torch.mul(eta, SCALE_SCALE)

    f = torch.log(torch.div(beta, eta)) \
        + (beta - 1) * torch.log(torch.div(time, eta)) \
        - torch.pow(torch.div(time, eta), beta)
    fu = torch.pow(torch.div(time, eta), beta)

    ll = torch.mul(status, f) + torch.mul((1 - status), (-fu))
    # print('time:', time, '\t', 'status:', status, '\t', 'beta:', beta, '\t', 'eta:', eta, '\t',
    # 'f:', f, '\t', 'fu:', fu, '\t', 'll:', - ll)

    return - ll

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

if __name__ == "__main__":
    # Load and split data
    data_config = load_config(cfg.DGP_CONFIGS_DIR, f"synthetic.yaml")
    dl = SingleEventSyntheticDataLoader().load_data(data_config=data_config,
                                                    linear=True, copula_name="clayton",
                                                    k_tau=0.25, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2)
    n_events = data_config['se_n_events']
    dgps = dl.dgps

    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype)

    # Format data to work easier with sksurv API
    X_train = train_dict['X']
    X_valid = valid_dict['X']
    X_test = test_dict['X']
    y_train = convert_to_structured(train_dict['T'], train_dict['E'])
    y_valid = convert_to_structured(valid_dict['T'], valid_dict['E']) 
    y_test = convert_to_structured(test_dict['T'], test_dict['E'])

    data = Data()
    loader = DataLoader(dataset=data, batch_size=32)

    input_dim, output_dim = 10, 1

    model = Net(input_dim, output_dim)
    # print(clf.parameters)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
    loss_list = []
    model.eval()
    
    time_name = 'time'
    status_name ='event'
    
    train_ll_viz, val_ll_viz, shape_viz, scale_viz, ci_viz = [], [], [], [], []
    
    for t in range(200):
        ll = 0
        beta_ave, eta_ave = 0, 0
        # for i, j in enumerate(loader):
        for i, j in enumerate(X_train):
            beta_pred, eta_pred = model(j.unsqueeze(0))
            beta_ave += beta_pred.squeeze(0)[0].item() / X_train.shape[0]
            eta_ave += eta_pred.squeeze(0)[0].item() / X_train.shape[0]

            #loss = neg_log_likelihood_loss(beta_pred, eta_pred, y_train[i])
            model1 = Weibull_log_linear(10, 2, 1, device, dtype)
            model2 = Weibull_log_linear(10, 2, 1, device, dtype)
            loss = calculate_loss_two_models(model1, model2, y_train[i])
            
            loss /= X_train.shape[0]
            # print(beta_pred, eta_pred, data.Y[i], loss.item())
            loss.backward()

        beta_val, eta_val = [], []
        vloss_total = 0
    
        for i, j in enumerate(X_valid):
            beta_t, eta_t = model(j.unsqueeze(0))
            #vloss = neg_log_likelihood_loss(beta_t, eta_t, y_valid[i])
            vloss /= X_valid.shape[0]
            
            vloss_total += vloss.item()

            beta_val.append(beta_t.squeeze(0)[0].item() * SHAPE_SCALE)
            eta_val.append(eta_t.squeeze(0)[0].item() * SCALE_SCALE)

        df_temp = pd.DataFrame()
        df_temp['time'] = y_valid[time_name]
        df_temp['status'] = y_valid[status_name]
        df_temp.reset_index(inplace=True)
        df_temp = pd.concat([df_temp, pd.DataFrame(beta_val, columns=['beta']),
                            pd.DataFrame(eta_val, columns=['eta'])], axis=1)
        # df_temp['hr'] = (df_temp['beta'] / df_temp['eta']) * pow(100 / df_temp['eta'], df_temp['beta'] - 1)
        # df_temp['hr'] = df_temp['eta'] * np.gamma(1 + 1 / df_temp['beta'])
        mean_list = []
        for m, n in enumerate(df_temp['eta']):
            # print(m, n)
            mean_list.append(n * math.gamma(1 + 1 / df_temp['beta'][m]))
        df_temp = pd.concat([df_temp, pd.DataFrame(mean_list, columns=['mean'])], axis=1)
        # df_temp['mean'].fillna(df_temp['mean'].mean(), inplace=True)
        # ci = concordance_index(df_temp['time'], -df_temp['hr'], df_temp['status'])
        ci = concordance_index(df_temp['time'], df_temp['mean'], df_temp['status'])

        # ll = torch.div(ll, data.X.size(0))
        # loss = -ll
        print("epoch:\t", t,
            "\t train loss:\t", "%.8f" % round(loss.item(), 6),
            "\t valid loss:\t", "%.8f" % round(vloss.item(), 6),
            "\t shape:\t", "%.4f" % round(beta_ave * SHAPE_SCALE, 4),
            "\t scale:\t", "%.3f" % round(eta_ave * SCALE_SCALE, 3),
            "\t concordance index:\t", "%.8f" % round(ci, 8))

        train_ll_viz.append(loss.item()), val_ll_viz.append(vloss.item())
        shape_viz.append(beta_ave * SHAPE_SCALE), scale_viz.append(eta_ave * SCALE_SCALE)
        ci_viz.append(ci)

        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()
        # with torch.no_grad():
        #     for param in clf.parameters():
        #         print(param.grad)
        #         param -= learning_rate * param.grad

    fig, axs = plt.subplots(4, figsize=[11, 20])
    axs[0].plot(train_ll_viz)
    axs[0].set_title("Average Train Loss")
    axs[0].plot(val_ll_viz)
    axs[0].set_title("Average Validation Loss")
    axs[1].plot(shape_viz)
    axs[1].set_title("Average Shape Parameter")
    axs[2].plot(scale_viz)
    axs[2].set_title("Average Scale Parameter")
    axs[3].plot(ci_viz)
    axs[3].set_title("Concordance Index")
    plt.show()