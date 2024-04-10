import numpy as np
import csv
import copy
import matplotlib.pyplot as plt

'''
Synthetic multi-event data, based on DeepHit
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