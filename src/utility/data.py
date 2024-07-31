import math
import pylab
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.special import lambertw
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class SingleEventDataset(Dataset):
    def __init__(self, feature_num, X, Y_T, Y_E):
        self.feature_num = feature_num
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(np.stack((Y_T[:,0], Y_E[:,0]), axis=1), dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y[idx,:]

class MultiEventDataset(Dataset):
    def __init__(self, feature_num, X, Y_T, Y_E):
        self.feature_num = feature_num
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y1 = torch.tensor(np.stack((Y_T[:,0], Y_E[:,0]), axis=1), dtype=torch.float32)
        self.Y2 = torch.tensor(np.stack((Y_T[:,1], Y_E[:,1]), axis=1), dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y1[idx,:], self.Y2[idx,:]

def format_data_for_survtrace(df_train, df_valid, df_test, n_events):
    y_train, y_valid, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(n_events):
        event_name = "event_{}".format(i)
        y_train[event_name] = df_train['duration']
        y_valid[event_name] = df_valid['duration']
        y_test[event_name] = df_test['duration']
    y_train['duration'] = df_train['duration']
    y_train['proportion'] = df_train['proportion']
    y_valid['duration'] = df_valid['duration']
    y_valid['proportion'] = df_valid['proportion']
    y_test['duration'] = df_test['duration']
    y_test['proportion'] = df_test['proportion']
    return y_train, y_valid, y_test
    
def calculate_vocab_size(df, cols_categorical):
    vocab_size = 0
    for i, feat in enumerate(cols_categorical):
        df[feat] = LabelEncoder().fit_transform(df[feat]).astype(float) + vocab_size
        vocab_size = df[feat].max() + 1
    return int(vocab_size)

'''
Generate synthetic dataset based on Donna's paper:
https://github.com/MLD3/Hierarchical_Survival_Analysis
'''
def make_synthetic(num_event):
    num_data = 5000
    num_feat = 5 #in each segment, total = 15 (5 features x 3 segments)
    
    #construct covariates
    bounds = np.array([-5, -10, 5, 10])
    x_11 = np.random.uniform(bounds[0], bounds[2], size=(num_data//2, num_feat))
    x_12 = np.random.uniform(bounds[0], bounds[2], size=(num_data//2, num_feat))
    x_21 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat))
    x_31 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat)) 
    x_22 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat))
    x_32 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat)) 
    
    x1 = np.concatenate((x_11, x_21, x_31), axis=1)
    x2 = np.concatenate((x_12, x_32, x_22), axis=1)
    x = np.concatenate((x1, x2), axis=0)
    
    #construct time to events
    gamma_components = []
    gamma_const = [1, 1, 1]
    for i in range(num_event + 1):
        gamma_components.append(gamma_const[i] * np.ones((num_feat,)))
    gamma_components.append(gamma_const[-1] * np.ones((num_feat,)))

    distr_noise = 0.4
    distr_noise2 = 0.4
    
    time2_coeffs = np.array([0, 1, 1])
    event_times = [] 
    raw_event_times = []
    raw_event_times2 = []
    for i in range(num_event):
        raw_time = np.power(np.matmul(np.power(np.absolute(x[:, :num_feat]), 1), gamma_components[0]), 2) + \
                   np.power(np.matmul(np.power(np.absolute(x[:, (i + 1)*num_feat:(i+2)*num_feat]), 1), gamma_components[i + 1]), 2)
        raw_event_times.append(raw_time)
        times = np.zeros(raw_time.shape)
        for j in range(raw_time.shape[0]):
            times[j] = np.random.lognormal(mean=np.log(raw_time[j]), sigma=distr_noise)
        event_times.append(times)
        raw_time2 = 1 * (time2_coeffs[2] * np.power(np.matmul(np.absolute(x[:, (0)*num_feat:(1)*num_feat]), gamma_components[2]), 1))
        raw_event_times2.append(raw_time2)

    t = np.zeros((num_data, num_event))
    for i in range(num_event):
        t[:, i] = event_times[i]
    labels = np.ones(t.shape)
    
    #time to event for second event (conditional event time)
    t_original = copy.deepcopy(t)
    num_inconsist = 0
    for i in range(num_data):
        if t_original[i, 0] < t_original[i, 1]:
            t[i, 1] = t_original[i, 1] + np.random.lognormal(mean=np.log(raw_event_times2[1][i]), sigma=distr_noise2)
            if t[i, 1] < t_original[i, 0]:
                num_inconsist += 1 
        elif t_original[i, 1] < t_original[i, 0]: 
            t[i, 0] = t_original[i, 0] + np.random.lognormal(mean=np.log(raw_event_times2[1][i]), sigma=distr_noise2)
            if t[i, 0] < t_original[i, 1]:
                num_inconsist += 1

    #enforce a prediction horizon
    horizon = np.percentile(np.min(t, axis=1), 50) 
    for i in range(t.shape[1]):
        censored = np.where(t[:, i] > horizon)
        t[censored, i] = horizon
        labels[censored, i] = 0
    
    print('label distribution: ', np.unique(labels, return_counts=True, axis=0))
    return x, t, labels

def safe_log(x):
    return np.log(x+1e-20*(x<1e-20))

def inverse_transform(value, risk, shape, scale):
    return (-safe_log(value) / np.exp(risk)) ** (1 / shape) * scale

def inverse_transform_weibull(p, shape, scale):
    return scale * (-safe_log(p)) ** (1 / shape)

def inverse_transform_lognormal(p, shape, scale):
    return stats.lognorm(s=scale*0.25, scale=shape).ppf(p)

def inverse_transform_exp(p, shape, scale):
    return stats.expon(scale).ppf(p)

def relu(z):
    return np.maximum(0, z)

def array_to_tensor(array, dtype=None, device='cpu'):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array_c = array.copy()
    tensor = torch.tensor(array_c, dtype=dtype).to(device)
    return tensor

def format_data(X, y, dtype, device):
    times = array_to_tensor(y['time'], dtype)
    events = array_to_tensor(y['event'], dtype)
    covariates = torch.tensor(X.to_numpy(), dtype=dtype).to(device)
    return (covariates, times, events)

def kendall_tau_to_theta(copula_name, k_tau):
    #TODO: Check this function (Ali)
    if copula_name == "clayton":
        return 2 * k_tau / (1 - k_tau)
    elif copula_name == "frank":
        return -np.log(1 - k_tau * (1 - np.exp(-1)) / 4)
    elif copula_name == "gumbel":
        return -1 / (k_tau - 1)
    else:
        raise ValueError('Copula not implemented')
    
def theta_to_kendall_tau(copula_name, theta):
    #TODO: Check this function (Ali)
    if copula_name == "clayton":
        return theta / (theta + 2)
    elif copula_name == "frank":
        return 1 - 4 * ((1 - np.exp(-theta)) / theta)
    elif copula_name == "gumbel":
        return (theta - 1) / theta
    else:
        raise ValueError('Copula not implemented')
    
def format_data_as_dict_single(X, events, times, dtype):
    data_dict = dict()
    data_dict['X'] = torch.tensor(X, dtype=dtype)
    data_dict['E'] = torch.tensor(events, dtype=dtype)
    data_dict['T'] = torch.tensor(times, dtype=dtype)
    return data_dict

def format_data_as_dict_multi(X, y_e, y_t, dtype):
    data_dict = dict()
    data_dict['X'] = torch.tensor(X, dtype=dtype)
    n_events = y_e.shape[1]
    for i in range(n_events):
        data_dict[f'E{i+1}'] = torch.tensor(y_e[:,i].astype(np.int64), dtype=torch.int64)
        data_dict[f'T{i+1}'] = torch.tensor(y_t[:,i].astype(np.int64), dtype=torch.int64)
    return data_dict

def format_data_deephit_single(train_dict, valid_dict, labtrans):
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].cpu().numpy()
    train_dict_dh['E'] = train_dict['E'].cpu().numpy()
    train_dict_dh['T'] = train_dict['T'].cpu().numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].cpu().numpy()
    valid_dict_dh['E'] = valid_dict['E'].cpu().numpy()
    valid_dict_dh['T'] = valid_dict['T'].cpu().numpy()
    get_target = lambda data: (data['T'], data['E'])
    y_train = labtrans.transform(*get_target(train_dict_dh))
    y_valid = labtrans.transform(*get_target(valid_dict_dh))
    out_features = len(labtrans.cuts)
    duration_index = labtrans.cuts
    train_data = {'X': train_dict_dh['X'], 'T': y_train[0], 'E': y_train[1]}
    valid_data = {'X': valid_dict_dh['X'], 'T': y_valid[0], 'E': y_valid[1]}
    return train_data, valid_data, out_features, duration_index

def format_data_deephit_cr(train_dict, valid_dict, time_bins):
    class LabTransform(LabTransDiscreteTime):
        def transform(self, durations, events):
            durations, is_event = super().transform(durations, events > 0)
            events[is_event == 0] = 0
            return durations, events.astype('int64')
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].numpy()
    train_dict_dh['E'] = train_dict['E'].numpy()
    train_dict_dh['T'] = train_dict['T'].numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].numpy()
    valid_dict_dh['E'] = valid_dict['E'].numpy()
    valid_dict_dh['T'] = valid_dict['T'].numpy()
    labtrans = LabTransform(time_bins.numpy())
    get_target = lambda data: (data['T'], data['E'])
    y_train = labtrans.transform(*get_target(train_dict_dh))
    y_valid = labtrans.transform(*get_target(valid_dict_dh))
    out_features = len(labtrans.cuts)
    duration_index = labtrans.cuts
    train_data = {'X': train_dict_dh['X'], 'T': y_train[0], 'E': y_train[1]}
    valid_data = {'X': valid_dict_dh['X'], 'T': y_valid[0], 'E': y_valid[1]}
    return train_data, valid_data, out_features, duration_index

def make_times_hierarchical(event_times, num_bins):
    min_time = np.min(event_times[event_times != -1]) 
    max_time = np.max(event_times[event_times != -1]) 
    time_range = max_time - min_time
    bin_size = time_range / num_bins
    binned_event_time = np.floor((event_times - min_time) / bin_size)
    binned_event_time[binned_event_time == num_bins] = num_bins - 1
    return binned_event_time

def format_hierarchical_data_cr(train_dict, valid_dict, test_dict,
                                num_bins, n_events, censoring_event=True):
    #If censoring_event=True, encode censoring as a seperate event
    train_times = np.stack([train_dict['T'] for _ in range(n_events)], axis=1)
    valid_times = np.stack([valid_dict['T'] for _ in range(n_events)], axis=1)
    test_times = np.stack([test_dict['T'] for _ in range(n_events)], axis=1)
    train_event_bins = make_times_hierarchical(train_times, num_bins=num_bins)
    valid_event_bins = make_times_hierarchical(valid_times, num_bins=num_bins)
    test_event_bins = make_times_hierarchical(test_times, num_bins=num_bins)
    if censoring_event:
        train_events = np.array(pd.get_dummies(train_dict['E']))
        valid_events = np.array(pd.get_dummies(valid_dict['E']))
        test_events = np.array(pd.get_dummies(test_dict['E']))
    else:
        train_events = np.array(pd.get_dummies(train_dict['E']))[:,1:] # drop censoring event
        valid_events = np.array(pd.get_dummies(valid_dict['E']))[:,1:]
        test_events = np.array(pd.get_dummies(test_dict['E']))[:,1:]
    train_data = [train_dict['X'], train_event_bins, train_events]
    valid_data = [valid_dict['X'], valid_event_bins, valid_events]
    test_data = [test_dict['X'], test_event_bins, test_events]
    return train_data, valid_data, test_data

def format_hierarchical_data_me(train_dict, valid_dict, test_dict, num_bins):
    train_event_bins = make_times_hierarchical(train_dict['T'].numpy(), num_bins=num_bins)
    valid_event_bins = make_times_hierarchical(valid_dict['T'].numpy(), num_bins=num_bins)
    test_event_bins = make_times_hierarchical(test_dict['T'].numpy(), num_bins=num_bins)
    train_events = train_dict['E'].numpy()
    valid_events = valid_dict['E'].numpy()
    test_events = test_dict['E'].numpy()
    train_data = [train_dict['X'], train_event_bins, train_events]
    valid_data = [valid_dict['X'], valid_event_bins, valid_events]
    test_data = [test_dict['X'], test_event_bins, test_events]
    return train_data, valid_data, test_data

def calculate_layer_size_hierarch(layer_size, n_time_bins):
    def find_factors(n):
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                factor1 = i
                factor2 = n // i
                if factor1 < factor2:
                    return factor1, factor2
        return (1, n_time_bins)
    result = find_factors(n_time_bins)
    return [(layer_size, result[0]), (layer_size, result[1])]
    
def format_survtrace_data(train_dict, valid_dict, time_bins, n_events):
    class LabTransform(LabTransDiscreteTime):
        def transform(self, durations, events):
            durations, is_event = super().transform(durations, events > 0)
            events[is_event == 0] = 0
            return durations, events.astype('int64')
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].numpy()
    train_dict_dh['E'] = train_dict['E'].numpy()
    train_dict_dh['T'] = train_dict['T'].numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].numpy()
    valid_dict_dh['E'] = valid_dict['E'].numpy()
    valid_dict_dh['T'] = valid_dict['T'].numpy()
    labtrans = LabTransform(time_bins.numpy())
    get_target = lambda data: (data['T'], data['E'])
    y_train = labtrans.transform(*get_target(train_dict_dh))
    y_valid = labtrans.transform(*get_target(valid_dict_dh))
    out_features = int(labtrans.out_features)
    duration_index = labtrans.cuts
    y_train_df, y_valid_df = pd.DataFrame(), pd.DataFrame()
    y_train_df['duration'] = y_train[0]
    y_valid_df['duration'] = y_valid[0]
    y_train_df['proportion'] = y_train[1]
    y_valid_df['proportion'] = y_valid[1]
    #for i in range(n_events):
    #    event_name = "event_{}".format(i)
        #y_train_df[event_name] = (train_dict['E'] == i) * 1.0
        #y_valid_df[event_name] = (valid_dict['E'] == i) * 1.0#z
    #    y_train_df[event_name] = (train_dict['T'])
    #    y_valid_df[event_name] = (valid_dict['T'])
    return y_train_df, y_valid_df, duration_index, out_features
        