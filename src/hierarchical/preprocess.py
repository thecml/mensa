'''
data preprocessing (binning time to events, split into train/test/validation)
'''

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from hierarchical import get_data


#######################################################################################
'''
gets data and bins the time to events of time to events are continuous
'''
def get_and_bin_data(dataset, params):
    raw_data, event_times, labs = get_data.get_data_by_name(dataset, params) 
    
    if params['discrete']: #nothing needs to be done 
        return raw_data, event_times, labs, 0, params['num_bins']
    
    #bin time to events
    min_time = np.min(event_times[event_times != -1]) 
    max_time = np.max(event_times[event_times != -1]) 
    time_range = max_time - min_time
    bin_size = time_range / params['num_bins']
    
    binned_event_time = np.floor((event_times - min_time) / bin_size)
    binned_event_time[binned_event_time == params['num_bins']] = params['num_bins'] - 1
    
    return raw_data, binned_event_time, labs, min_time, max_time


'''
reformat labels so that each label corresponds to a trajectory (e.g., event1 then event2, event1 only, event2 then event1)
'''
def get_trajectory_labels(labs):
    unique_labs = np.unique(labs, axis=0)
    new_labs = np.zeros((labs.shape[0],))
    
    for i in range(labs.shape[0]):
        for j in range(unique_labs.shape[0]):
            if np.all(unique_labs[j, :] == labs[i, :]):
                new_labs[i] = j
    
    return new_labs
    

'''
split data into training, validation, and test sets
'''
def split_data(raw_data, event_time, labs):
    traj_labs = labs
    if labs.shape[1] > 1: 
        traj_labs = get_trajectory_labels(labs)
    
    #split into training/test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    train_i, test_i = next(splitter.split(raw_data, traj_labs))
    
    train_data = raw_data[train_i, :]
    train_labs = labs[train_i, :]
    train_event_time = event_time[train_i, :]
    
    pretest_data = raw_data[test_i, :]
    pretest_labs = labs[test_i, :]
    pretest_event_time = event_time[test_i, :]
    
    #further split test set into test/validation
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    new_pretest_labs = get_trajectory_labels(pretest_labs)
    test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
    test_data = pretest_data[test_i, :]
    test_labs = pretest_labs[test_i, :]
    test_event_time = pretest_event_time[test_i, :]
    
    val_data = pretest_data[val_i, :]
    val_labs = pretest_labs[val_i, :]
    val_event_time = pretest_event_time[val_i, :]
    
    #package for convenience
    train_package = [train_data, train_event_time, train_labs]
    test_package = [test_data, test_event_time, test_labs]
    validation_package = [val_data, val_event_time, val_labs]
    
    return train_package, test_package, validation_package


'''
main block 
'''
if __name__ == '__main__':
    print(':)')