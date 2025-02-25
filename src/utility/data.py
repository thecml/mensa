import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special import lambertw
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GumbelCopula

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
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
    return torch.relu(z)

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
    if copula_name == "clayton":
        return ClaytonCopula().theta_from_tau(k_tau)
    elif copula_name == "frank":
        return FrankCopula().theta_from_tau(k_tau)
    elif copula_name == "gumbel":
        return GumbelCopula().theta_from_tau(k_tau)
    else:
        raise NotImplementedError('Copula not implemented')
    
def theta_to_kendall_tau(copula_name, theta):
    if copula_name == "clayton":
        return ClaytonCopula().tau(theta)
    elif copula_name == "frank":
        return FrankCopula().tau(theta)
    elif copula_name == "gumbel":
        return GumbelCopula().tau(theta)
    else:
        raise NotImplementedError('Copula not implemented')

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

def format_data_deephit_competing(train_dict, valid_dict, time_bins):
    class LabTransform(LabTransDiscreteTime):
        def transform(self, durations, events):
            durations, is_event = super().transform(durations, events > 0)
            events[is_event == 0] = 0
            return durations, events.astype('int64')
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].cpu().numpy()
    train_dict_dh['E'] = train_dict['E'].cpu().numpy()
    train_dict_dh['T'] = train_dict['T'].cpu().numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].cpu().numpy()
    valid_dict_dh['E'] = valid_dict['E'].cpu().numpy()
    valid_dict_dh['T'] = valid_dict['T'].cpu().numpy()
    labtrans = LabTransform(time_bins.cpu().numpy())
    get_target = lambda data: (data['T'], data['E'])
    y_train = labtrans.transform(*get_target(train_dict_dh))
    y_valid = labtrans.transform(*get_target(valid_dict_dh))
    out_features = len(labtrans.cuts)
    duration_index = labtrans.cuts
    train_data = {'X': train_dict_dh['X'], 'T': y_train[0], 'E': y_train[1]}
    valid_data = {'X': valid_dict_dh['X'], 'T': y_valid[0], 'E': y_valid[1]}
    return train_data, valid_data, out_features, duration_index

def format_data_deephit_multi(train_dict, valid_dict, labtrans, risk):
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].cpu().numpy()
    train_dict_dh['E'] = train_dict['E'][:,risk].cpu().numpy()
    train_dict_dh['T'] = train_dict['T'][:,risk].cpu().numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].cpu().numpy()
    valid_dict_dh['E'] = valid_dict['E'][:,risk].cpu().numpy()
    valid_dict_dh['T'] = valid_dict['T'][:,risk].cpu().numpy()
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
    train_times = np.stack([train_dict['T'].cpu().numpy() for _ in range(n_events)], axis=1)
    valid_times = np.stack([valid_dict['T'].cpu().numpy() for _ in range(n_events)], axis=1)
    test_times = np.stack([test_dict['T'].cpu().numpy() for _ in range(n_events)], axis=1)
    train_event_bins = make_times_hierarchical(train_times, num_bins=num_bins)
    valid_event_bins = make_times_hierarchical(valid_times, num_bins=num_bins)
    test_event_bins = make_times_hierarchical(test_times, num_bins=num_bins)
    if censoring_event:
        train_events = np.array(pd.get_dummies(train_dict['E'].cpu().numpy()))
        valid_events = np.array(pd.get_dummies(valid_dict['E'].cpu().numpy()))
        test_events = np.array(pd.get_dummies(test_dict['E'].cpu().numpy()))
    else:
        train_events = np.array(pd.get_dummies(train_dict['E'].cpu().numpy()))[:,1:] # drop censoring event
        valid_events = np.array(pd.get_dummies(valid_dict['E'].cpu().numpy()))[:,1:]
        test_events = np.array(pd.get_dummies(test_dict['E'].cpu().numpy()))[:,1:]
    train_data = [train_dict['X'].cpu().numpy(), train_event_bins, train_events]
    valid_data = [valid_dict['X'].cpu().numpy(), valid_event_bins, valid_events]
    test_data = [test_dict['X'].cpu().numpy(), test_event_bins, test_events]
    return train_data, valid_data, test_data

def format_hierarchical_data_me(train_dict, valid_dict, test_dict, num_bins):
    train_event_bins = make_times_hierarchical(train_dict['T'].cpu().numpy(), num_bins=num_bins)
    valid_event_bins = make_times_hierarchical(valid_dict['T'].cpu().numpy(), num_bins=num_bins)
    test_event_bins = make_times_hierarchical(test_dict['T'].cpu().numpy(), num_bins=num_bins)
    train_events = train_dict['E'].cpu().numpy()
    valid_events = valid_dict['E'].cpu().numpy()
    test_events = test_dict['E'].cpu().numpy()
    train_data = [train_dict['X'].cpu().numpy(), train_event_bins, train_events]
    valid_data = [valid_dict['X'].cpu().numpy(), valid_event_bins, valid_events]
    test_data = [test_dict['X'].cpu().numpy(), test_event_bins, test_events]
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
    