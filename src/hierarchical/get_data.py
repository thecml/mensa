'''
gets data depending on desired dataset
'''

import numpy as np
import csv
import copy
import matplotlib.pyplot as plt


############################################################################################
'''
read data from file
'''
def read_file(file_path, num_dim):
    file_handle = open(file_path)
    contents = file_handle.read()
    contents = contents.replace('\t', ',')
    contents = contents.replace('\n', ',')
    contents = np.array(contents.split(',')[:-1])
    contents = contents.reshape((-1, num_dim))
    file_handle.close()
    return contents


'''
bin data into qunitiles, puts nans in their own cateogry
'''
def bin_data(raw_data):
    not_nan_ind = np.where(~np.isnan(raw_data))[0]
    not_nan_data = raw_data[not_nan_ind]
    
    quintiles = np.percentile(not_nan_data, [20, 40, 60, 80])
    binned_data = np.ones(not_nan_data.shape)
    binned_data[not_nan_data > quintiles[0]] += 1
    binned_data[not_nan_data > quintiles[1]] += 1
    binned_data[not_nan_data > quintiles[2]] += 1
    binned_data[not_nan_data > quintiles[3]] += 1
    
    binned_data_final = np.zeros(raw_data.shape)
    binned_data_final[not_nan_ind] = binned_data
    
    return binned_data_final


'''
discretize data by turning each entry into a 1 hot vector
'''
def discretize(raw_data):
    if np.unique(raw_data).shape[0] > 10:
        try:
            raw_data[raw_data == ''] = 'nan'
            raw_data = raw_data.astype(float)
            raw_data = bin_data(raw_data.astype(float))
        except ValueError:
            pass
        
    values = np.unique(raw_data)
    discretized = np.zeros((raw_data.shape[0], values.shape[0]))
    
    for i in range(raw_data.shape[0]):
        val_ind = np.where(values == raw_data[i])[0][0]
        discretized[i, val_ind] = 1
  
    return discretized


############################################################################################
'''
synthetic multi-event data, based on deephit
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

    
'''
get ADNI data
'''
def get_ADNI(data_loc):
    global data_root
    data_file = data_loc + 'ADNIMERGE.csv'
    rows = read_file(data_file, 113)[1:, :] 
    rows = np.char.strip(rows, '"')
    
    death_data_file = data_loc + 'NEUROPATH_04_12_18.csv'
    death_rows = np.empty((0, 2)) #get patients who died
    with open(death_data_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            death_rows = np.append(death_rows, [[row['RID'], row['NPDOD']]], axis=0)
    
    #incomplete data is usually > or < some value, they get replaced with the value after the > or <
    greater = np.flatnonzero(np.core.defchararray.find(rows,'>')!=-1)
    greater = np.unravel_index(greater, rows.shape)
    less = np.flatnonzero(np.core.defchararray.find(rows,'<')!=-1)
    less = np.unravel_index(less, rows.shape)
    rows[greater] = np.char.strip(rows[greater], '>')
    rows[less] = np.char.strip(rows[less], '<')
    
    #inclusion/exclusion
    keep = np.where(rows[:, 7] != 'AD')[0]
    rows = rows[keep, :]
    
    #order by participant
    ordering = np.argsort(rows[:, 0])
    rows = rows[ordering, :]
    pats = np.unique(rows[:, 0]).astype(int)
    
    #time to events
    horizon = 60
    labs = 0 * np.ones((pats.shape[0], 2))
    time_to_all = np.zeros((pats.shape[0], 2)) 
    
    for i in range(pats.shape[0]):
        obs_times = rows[rows[:, 0].astype(int) == pats[i], -2].astype(int)
        time_to_all[i, :] = min(horizon - 1, np.max(obs_times)) #time at death or censorship
        diags = rows[rows[:, 0].astype(int) == pats[i], 59]
        if 'Dementia' in diags and np.min(obs_times[diags == 'Dementia']) <= horizon:
            time_to_all[i, 0] = np.min(obs_times[diags == 'Dementia'])
            labs[i, 0] = 1
        if pats[i] in death_rows[:, 0].astype(int):
            pat_i = np.where(death_rows[:, 0].astype(int) == pats[i])[0]
            first_vis_i = np.where(obs_times == 0)[0][0]
            first_vis = np.datetime64(rows[rows[:, 0].astype(int) == pats[i], 6][first_vis_i])
            death_time = ((np.datetime64(death_rows[pat_i, 1][0]) - first_vis) / 30).astype(int)
            if death_time <= horizon:
                time_to_all[i, 1] = death_time
                labs[i, 1] = 1
                if 'Dementia' not in diags:
                    time_to_all[i, 0] = death_time
    time_to_all[time_to_all >= horizon] = horizon - 1
    
    #features, include participant ids
    del_cols = [1, 2, 3, 4, 5, 6, 59, 78, 109, 110, 112]
    formatted_data = np.delete(rows, del_cols, axis=1)
    discretized_data = np.zeros((formatted_data.shape[0], 0))
    for i in range(1, formatted_data.shape[1]): #discretize feature by feature
        raw = formatted_data[:, i]
        discretized_data = np.append(discretized_data, discretize(raw), axis=1) 
     
    #include only data from alignment (nothing after)   
    final_data = np.zeros((pats.shape[0], discretized_data.shape[1]))
    for i in range(pats.shape[0]):
        pat_rows = np.where(rows[:, 0].astype(int) == pats[i])[0]
        obs_times = rows[rows[:, 0].astype(int) == pats[i], -2].astype(int)
        first_vis_i = np.where(obs_times == 0)[0][0]
        final_data[i, :] += discretized_data[pat_rows[first_vis_i], :]
        
    data_sums = np.sum(final_data, axis=0)
    too_few = 1
    too_many = final_data.shape[0] - 1
    del_col = np.where(np.logical_or(data_sums <= too_few, data_sums >= too_many))[0]
    final_data = np.delete(final_data, del_col, axis=1)
    
    print('num patients, num features: ', final_data.shape)
    print('label distribution: ', np.unique(labs, return_counts=True, axis=0))
    print('time to event distribution: ', np.unique(time_to_all[labs != -1], return_counts=True))
    return final_data, time_to_all, labs.astype(int)


'''
get SEER data
'''
def get_SEER(data_loc):
    #read files
    data_file = data_loc + 'yr1975_2016.seer9/RESPIR.TXT'
    feature_indexes = read_file(data_loc + 'feature_index.txt', 1)   
    rows = read_file(data_file, 1)
    
    #extract features
    num_rows = rows.shape[0]
    num_feat = feature_indexes.shape[0]
    formatted_data = np.zeros((num_rows, num_feat), dtype=object)
    for i in range(num_feat):
        indexes = feature_indexes[i][0]
        indexes = indexes.split('-')
        start = int(indexes[0]) - 1
        end = int(indexes[0])
        if len(indexes) == 2:
            end = int(indexes[1]) 
        feature = rows[:, 0].view((str,1)).reshape(len(rows[:, 0]),-1)[:, start:end]
        feature = np.fromstring(feature.tostring(),dtype=(str, end-start))
        formatted_data[:, i] = feature
    
    #remove anyone with unknown observation time (no survival time, no censoring time)
    #104 comes from looking up the survival feature number from the documentation in the feature_index.txt file
    #105 - flag on followup status, only take non-left censored with known followup (https://seer.cancer.gov/survivaltime/3-fields-survival-time-active.pdf)
    #81 - cause of death, 99999 means unknown (get rid of these), breast cancer 26000, 50130 - pulmonary disease, 220200-larynx cancer
    keep = np.where(formatted_data[:, 104].astype(int) < 9999)[0]
    keep = np.intersect1d(keep, np.where(formatted_data[:, 105] == '1')[0])
    keep = np.intersect1d(keep, np.where(np.isin(formatted_data[:, 82], ['00000', '22020', '50130']))[0])
    formatted_data = formatted_data[keep, :]
    
    #keep unique patients 
    unique_pats = np.unique(formatted_data[:, 0].astype(int), return_index=True)[1]
    formatted_data = formatted_data[unique_pats, :]
    
    #align by age of diagnosis, between 60-65
    keep = np.where(np.logical_and(formatted_data[:, 4].astype(int) >= 60, formatted_data[:, 4].astype(int) <= 65))[0]
    formatted_data = formatted_data[keep, :]
    
    #get rid of patients whose obs times are very far out ( > 150)
    horizon = 120
    offset = 12
    keep = np.where(np.logical_and(formatted_data[:, 104].astype(int) > offset - 1, formatted_data[:, 104].astype(int) < 1000))[0] #132 to get rid of horizon censored
    formatted_data = formatted_data[keep, :]
    
    #time to events (align at diagnosis? use the survival months feature (features 106/107, indexes 105/106)?)
    last_obs_time = formatted_data[:, 104].astype(int) - offset
    t = np.zeros((formatted_data.shape[0], 2)) 
    t[:, 0] = last_obs_time
    t[:, 1] = last_obs_time
    t[t > horizon - 1] = horizon - 1
    last_obs_time[last_obs_time > horizon - 1] = horizon - 1
    
    #get labels, including censorship (feature 68 for tumor behavior, dead/alive tag at feature 85/index 84)
    labels = np.zeros((formatted_data.shape[0], 2))
    labels[np.where(np.logical_and(formatted_data[:, 81] == '22020', formatted_data[:, 104].astype(int) < horizon))[0], 0] = 1 
    labels[np.where(np.logical_and(formatted_data[:, 81] == '50130', formatted_data[:, 104].astype(int) < horizon))[0], 1] = 1 
    
    #process features delete year specific and label related features, discretize the rest (keep participant id)
    include = np.arange(0, 28) #- 1
    include = np.setdiff1d(include, [9,10])
    formatted_data = formatted_data[:, include]
    discretized_data = np.zeros((formatted_data.shape[0], 0))
    for i in range(1, formatted_data.shape[1]):
        raw = formatted_data[:, i]
        discretized_data = np.append(discretized_data, discretize(raw), axis=1)
    
    print('num patients, num features: ', discretized_data.shape)
    print('labels ', np.unique(labels, return_counts=True, axis=0))
    return discretized_data, t, labels 


'''
given the dataset name, calls the appropriate get_data or make_synthetic function
returns data in format: covariates, time to all events, labels
'''
def get_data_by_name(dataset_name, params):
    print(dataset_name)
    
    if dataset_name == 'Synthetic': 
        return make_synthetic(params['num_events'])
    
    elif dataset_name == 'ADNI':
        return get_ADNI(params['location'])
    
    elif dataset_name == 'SEER':
        return get_SEER(params['location'])


'''
main block 
'''
if __name__ == '__main__':
    print(':)')