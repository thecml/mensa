'''
common functions
'''

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
import torch.nn as nn

from hierarchical import preprocess
from hierarchical import simulation
from hierarchical import hierarch
from hierarchical import direct


###################################################################################################
'''
uncensored loss helper function, maximize likelihood of event that occurred
'''
def get_uncensored_loss(num_time_bins, num_extra_bins, output, uncensored_ind, labels, events, time_to, weights=None):
    num_uncensored = uncensored_ind.shape[0]
    uncensored_output = output[torch.LongTensor(uncensored_ind), :]
    uncensored_labels = labels[torch.LongTensor(uncensored_ind)].reshape(-1)
    mapped_uncen_labs = np.zeros(uncensored_labels.shape)
    for i in range(events.shape[0]):
        mapped_uncen_labs[np.where(torch.Tensor(uncensored_labels) == events[i])[0]] = i
        
    uncensored_time_to = time_to[torch.LongTensor(uncensored_ind)].reshape(-1)
    uncensored_time_to[uncensored_time_to > (num_time_bins - num_extra_bins + 1)] = num_time_bins - num_extra_bins + 1
    uncensored_time_bins = (torch.Tensor(mapped_uncen_labs) * num_time_bins) + torch.Tensor(uncensored_time_to) 
    uncensored_filter = np.zeros(uncensored_output.shape)
    uncensored_filter[np.arange(num_uncensored), np.array(uncensored_time_bins).astype(int)] = 1
    
    uncensored_filtered_out = uncensored_output * torch.Tensor(uncensored_filter)
    uncensored_filtered_out = torch.sum(uncensored_filtered_out, 1)
    uncensored_filtered_out[uncensored_filtered_out == 0] = uncensored_filtered_out[uncensored_filtered_out == 0] + 1e-4
    uncensored_loss = -1 * torch.log(uncensored_filtered_out)

    uncensored_loss = torch.sum(uncensored_loss)/uncensored_ind.shape[0]
    
    return uncensored_loss


'''
censored loss helper function, maximize likelihood of occurrence after observation for all events
'''
def get_censored_loss(num_time_bins, num_extra_bins, output, censored_ind, time_to, num_events):
    num_censored = censored_ind.shape[0]
    censored_output = output[torch.LongTensor(censored_ind), :]
    censored_time_to = time_to[torch.LongTensor(censored_ind)]
    censored_time_bins = censored_time_to 
    censored_filter = np.zeros(censored_output.shape)
    for i in range(num_events):
        for j in range(num_censored):
            obs_bin = int(censored_time_bins[j] + 1)
            censored_filter[j, (i * (num_time_bins + num_extra_bins)) + obs_bin:(i +1) * (num_time_bins + num_extra_bins)] = 1
    filtered_censored_out = censored_output * torch.Tensor(censored_filter)
    filtered_censored_out = torch.sum(filtered_censored_out, 1)
    filtered_censored_out[filtered_censored_out <= 0] = filtered_censored_out[filtered_censored_out <= 0] + 1e-4
    censored_loss = -1 * torch.log(filtered_censored_out)
    censored_loss = torch.sum(censored_loss)/censored_loss.shape[0]
    
    return censored_loss


###################################################################################################
'''
training function for neural nets
'''
def train_network(mod, mod_params, loss_fx, hyperparams, train_package, val_package, ranks, \
                  num_event, terminal_events, num_extra, min_epochs, verbose=True):
    #unpack
    num_time_bins = mod.num_bins
    train_data, train_times, train_labels = train_package[0], train_package[1], train_package[2]
    val_data, val_times, val_labels = val_package[0], val_package[1], val_package[2]
    train_data = (torch.Tensor(train_data))
    val_data = torch.Tensor(val_data)
    
    #setup
    l_rate, l2_const, num_batch = hyperparams[0], hyperparams[1], hyperparams[2]
    optimizer = optim.Adam(mod_params, lr=l_rate, weight_decay=l2_const) 
    #min_epochs = 50
    max_epochs = 200
    loss_diff = 1
    loss_prev = 1000
    ctd_tol = 1e-3
    i = 0
    
    #initial evaluation
    val_out = get_surv_curves(val_data, mod)
    val_ctds = eval_overall(val_out, val_times, val_labels, num_event, num_time_bins, num_extra, terminal_events, ranks)
    val_ctd_avg = (np.average(val_ctds['Proposed']) + val_ctds['Local proposed']) / 2
    prev_val_ctd_avg = 0
    
    #train model 
    while (val_ctd_avg - prev_val_ctd_avg > ctd_tol or i < min_epochs) and i < max_epochs:
        loss = 0  
        splitter = KFold(num_batch) 
        for _, batch_ind in splitter.split(train_data.squeeze(0)):
            train_output = mod(train_data[batch_ind, :])
            batch_loss = loss_fx(train_output, torch.Tensor(train_labels[batch_ind, :]), torch.Tensor(train_times[batch_ind, :]), mod, train_data[batch_ind, :])
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += (batch_loss.detach() / num_batch)
        
        #evaluate every 10 epochs on validation data        
        loss_diff = loss_prev - loss
        loss_prev = loss 
        if i % 10 == 0:
            prev_val_ctd_avg = val_ctd_avg
            val_out = get_surv_curves(val_data, mod)
            val_eval = eval_overall(val_out, val_times, val_labels, num_event, num_time_bins, num_extra, terminal_events, ranks)
            val_ctd_avg = (np.average(val_eval['Proposed']) + val_eval['Local proposed']) / 2
            if verbose:
                print('new training evaluation')
                print(i, loss, val_ctd_avg - prev_val_ctd_avg)
                print(val_eval)
        i += 1
    if verbose:   
        print(i, loss, loss_diff, val_ctd_avg)
        print(val_eval)
        
    return mod


###################################################################################################
'''
basic, time-dependent c index
'''
def get_basic_c(mod_out, event_times, labs, num_events, num_bins, num_extra_bin, terminal_events, event_ranks):   
    num_samples = mod_out[0].shape[0] 
    
    #initialize ctds
    for_numerators = np.zeros((num_events,))
    for_denominators = np.zeros((num_events,))
    
    #loop through events
    for i in range(num_events):
        event_out = mod_out[i]
        for j in range(num_samples):
            had_event = labs[j, i] == 1
            if had_event:
                event_time = min(int(event_times[j, i]) + 1, num_bins + num_extra_bin)
                f_compare = np.where(event_times[:, i] > event_time - 1)[0]
                f_compare = np.union1d(f_compare, np.where(np.logical_and(event_times[:, i] == event_time - 1, labs[:, i] == 0))[0])
                if event_out.shape[1] == 1:
                    event_time = 0
                reference_risk = event_out[j, event_time]
                f_comp_risk = event_out[f_compare, event_time]
                for_denominators[i] += f_compare.shape[0]
                for_numerators[i] += np.where(f_comp_risk > reference_risk)[0].shape[0]
                for_numerators[i] += np.where(f_comp_risk == reference_risk)[0].shape[0] / 2
                
    for_denominators[for_denominators == 0] = 1
    c_for = for_numerators / for_denominators
    
    return np.concatenate((c_for, [np.average(c_for)]))
        

'''
proposed metric: consistency
'''
def get_proposed_metric(mod_out, event_times, labs, num_events, num_bin, num_extra_bin, terminal_events, event_ranks):
    num_samples = mod_out[0].shape[0]
    
    #initialize ctds
    for_numerators = np.zeros((num_events,))
    for_denominators = np.zeros((num_events,))
    
    #loop through events
    for i in range(num_events):
        event_out = mod_out[i]
        for j in range(num_samples):
            had_event = labs[j, i] == 1
            if had_event:
                event_time = min(int(event_times[j, i]) + 1, num_bin + num_extra_bin - 0)
                f_compare = np.where(event_times[:, i] > event_time - 1)[0]
                f_compare = np.union1d(f_compare, np.where(np.logical_and(event_times[:, i] == event_time - 1, labs[:, i] == 0))[0])
                if len(f_compare) < 1:
                    continue
                comp_n = np.zeros((f_compare.shape[0],))
                comp_d = np.zeros((f_compare.shape[0],))
                last_time = int(max(event_times[f_compare, i]) + 1)
                #find comparable time points
                if last_time == event_time and last_time + 1 < event_out.shape[1]:
                    last_time = last_time + 1
                if event_out.shape[1] == 1:
                    event_time = 0
                    last_time = 1
                #loop through comparable time points
                for k in range(event_time, last_time):
                    comp_at_k = np.where(event_times[:, i] > k - 1)[0]
                    comp_at_k = np.union1d(comp_at_k, np.where(np.logical_and(event_times[:, i] == k - 1, labs[:, i] == 0))[0])
                    comp_risk = event_out[f_compare, k]
                    reference_risk = event_out[j, k]
                    include_pos = np.isin(f_compare, comp_at_k)
                    comp_d += include_pos
                    comp_n += include_pos * np.greater(comp_risk, reference_risk)
                    comp_n += include_pos * (np.equal(comp_risk, reference_risk) / 2)
                    
                for_numerators[i] += np.sum(comp_n[comp_d > 0] / comp_d[comp_d > 0])
                for_denominators[i] += comp_d[comp_d > 0].shape[0]
                for_numerators[i] += comp_d[comp_d == 0].shape[0] / 2
                for_denominators[i] += comp_d[comp_d == 0].shape[0]
                
    for_denominators[for_denominators == 0] = 1
    values = for_numerators / for_denominators
    return np.concatenate((values, [np.average(values)]))


###################################################################################################
'''
local evaluation
eval types: c, proposed
'''
def get_local_eval(mod_out, event_times, labs, num_events, num_bins, num_extra_bin, eval_type):   
    num_samples = mod_out[0].shape[0] 
    
    #initialize results
    for_numerators = 0
    for_denominators = 0
    
    for i in range(num_samples):
        sample_out = np.concatenate([mod_out[a][i, :].reshape(1, -1) for a in range(len(mod_out))], axis=0)
        for j in range(num_events):
            had_event = labs[i, j] == 1
            if had_event:
                event_time = min(int(event_times[i, j]) + 1, num_bins + num_extra_bin - 0)
                f_compare = np.where(event_times[i, :] > event_time - 1)[0]
                f_compare = np.union1d(f_compare, np.where(np.logical_and(event_times[i, :] == event_time - 1, labs[i, :] == 0))[0])
                
                if f_compare.shape[0] < 1:
                    continue
                comp_n = np.zeros((f_compare.shape[0],))
                comp_d = np.zeros((f_compare.shape[0],))
                stop = int(min(max(event_times[i, f_compare]) + 1, num_bins + num_extra_bin - 1))
                if eval_type == 'c':
                    stop = event_time
                if sample_out.shape[1] == 1:
                    event_time = 0
                    stop = 0
                for k in range(event_time, stop + 1):
                    comp_at_k = np.where(event_times[i, :] > k - 1)[0]
                    comp_at_k = np.union1d(comp_at_k, np.where(np.logical_and(event_times[i, :] == k - 1, labs[i, :] == 0))[0])
                    comp_risk = sample_out[f_compare, k]
                    reference_risk = sample_out[j, k]
                    include_pos = np.isin(f_compare, comp_at_k)
                    comp_d += include_pos
                    comp_n += include_pos * np.greater(comp_risk, reference_risk)
                    comp_n += include_pos * (np.equal(comp_risk, reference_risk) / 2)         
             
                keep = np.where(comp_d > -1)[0]
                for_denominators += 1
                comp_n[comp_d == 0] = 0.5
                comp_d[comp_d == 0] = 1
                for_numerators += np.average(comp_n[keep] / comp_d[keep]) 
    
    eval_result = for_numerators / for_denominators
    
    return eval_result


'''
evaluate and combine local/global numbers
'''
def eval_overall(mod_out, event_times, labs, num_events, num_bin, num_extra_bin, terminal_events, event_ranks):
    basic_c = get_basic_c(mod_out, event_times, labs, num_events, num_bin, num_extra_bin, terminal_events, event_ranks)
    proposed_g = get_proposed_metric(mod_out, event_times, labs, num_events, num_bin, num_extra_bin, terminal_events, event_ranks)
    
    local_p = get_local_eval(mod_out, event_times, labs, num_events, num_bin, num_extra_bin, 'proposed')
    
    results = {'C index': basic_c, \
               'Proposed': proposed_g, \
               'Local proposed': local_p, \
              }
    return results
        

###################################################################################################    
'''
compute survival curves
'''
def get_surv_curves(inp_data, model, data_processed=False, return_tensor=False):
    total_bin = model.num_bins + model.num_extra_bin + 1
    
    probs = inp_data
    if not data_processed:
        probs = model.forward(inp_data)
        probs = [probs[i][-1] for i in range(len(probs))]
    surv_curves = []
    for i in range(model.num_events):
        surv_curves.append(torch.Tensor(np.ones((probs[0].shape[0], total_bin))))
    
    #use network output to get survival curves
    for i in range(1, total_bin):
        for j in range(model.num_events):
            surv_curves[j][:, i] -= torch.sum(probs[j][:, :i], dim=1)
            surv_curves[j][:, i][surv_curves[j][:, i] < 0] = 0
    
    if not return_tensor:
        return [surv_curves[i].detach().numpy() for i in range(len(surv_curves))]
    return surv_curves
    
    
'''
plot survival curves, colors use standard pyplot order
not called anywhere else in the code but may be useful for debugging
'''
def plot_surv_curve(curves, samples, labels, times):
    for i in range(len(samples)):
        for j in range(len(curves)):
            plt.plot(curves[j][samples[i], :].detach().numpy())
        plt.ylabel('Survival Probability')
        plt.xlabel('Time bin')
        plt.title('Sample: ' + str(samples[i]) + ', Event times: ' + str(times[i, :]) + ', Label: ' + str(labels[i, :]))
        plt.show()
    
    return 1
    
    
###################################################################################################                
'''
produce model
'''
def produce_model(method, train_package, val_package, test_package, settings, hyperparams):
    test_data = copy.deepcopy(test_package[0])
    test_event_time = copy.deepcopy(test_package[1])
    test_labs = copy.deepcopy(test_package[2])
    
    num_events, num_time_bins = settings['num_events'], settings['num_bins'] #+ 1
    event_ranks, event_groups = settings['event_ranks'], settings['event_groups']
    terminal_events = settings['terminal_events']
    min_epochs = settings['min_epoch']
    
    layer_sizes = [test_data.shape[1]] + hyperparams[0]
    event_net_sizes = [(hyperparams[0][-1], 1)] + hyperparams[1]
    train_hyperparams = hyperparams[2]
    loss_hyperparams = hyperparams[3]
    multitask = hyperparams[4]
    dh = hyperparams[5]
    num_extra_bins = hyperparams[6]
    
    mod, loss = 0, 0
    if 'hierarch' in method:
        mod = hierarch.hierarch_proposed(layer_sizes, event_net_sizes, num_events, \
                                                num_time_bins, event_groups, num_extra_bins, \
                                                terminal_events, event_ranks, multitask=multitask)
        loss = hierarch.hierarch_loss(terminal_events, event_ranks, loss_hyperparams)
    
    elif 'direct' in method:
        mod = direct.direct_network(layer_sizes, event_net_sizes, num_events, \
                                                num_time_bins, event_groups, num_extra_bins, \
                                                terminal_events, event_ranks, multitask=multitask, dh=dh)
        loss = direct.direct_loss(terminal_events, event_ranks, loss_hyperparams)
        
    all_parameters = mod.get_parameters()
    train_network(mod, all_parameters, loss, train_hyperparams, train_package, val_package, event_ranks, \
        num_events, terminal_events, num_extra_bins, min_epochs, verbose=True)
    test_curves = get_surv_curves(torch.Tensor(test_data), mod)
    return test_curves
    
    #test_results = eval_overall(test_curves, test_event_time, test_labs, num_events, num_time_bins, mod.num_extra_bin, terminal_events, event_ranks)
    #print('test results')
    #print(test_results)
    
    #bootstrap_results(test_curves, test_event_time, test_labs, num_events, num_time_bins, \
    #                  mod.num_extra_bin, terminal_events, event_ranks, 1000)
    
    #return 1


'''
simulate based on ground truth curves
'''
def produce_sim(test_package, settings):
    test_data = test_package[0]
    test_event_time = test_package[1]
    test_labs = test_package[2]
    
    num_events, num_time_bins = settings['num_events'], settings['num_bins']
    event_ranks, event_groups = settings['event_ranks'], settings['event_groups']
    terminal_events = settings['terminal_events']
    min_time, max_time = settings['min_time'], settings['max_time']
    
    sim = simulation.simulator(num_time_bins, event_groups, min_time, max_time)
    test_curves = sim.get_surv_curves(test_data, test_labs)
    return test_curves
    
    #test_labs = test_package[2]
    #test_results = eval_overall(test_curves, test_event_time, test_labs, num_events, num_time_bins, sim.num_extra_bin, terminal_events, event_ranks)
    #print('test results')
    #print(test_results)
    
    #bootstrap_results(test_curves, test_event_time, test_labs, num_events, num_time_bins, \
    #                  sim.num_extra_bin, terminal_events, event_ranks, 1000)

    #return 1


def bootstrap_results(mod_out, times, labs, num_events, num_bin, num_extra_bin, term_events, event_ranks, num_boots=1000):
    c_glob = np.zeros((num_boots, num_events + 1))
    prop_glob = np.zeros((num_boots, num_events + 1))
    
    prop_loc = np.zeros((num_boots,))
    
    traj_labs = preprocess.get_trajectory_labels(labs)
    num_data = mod_out[0].shape[0]
    for i in range(num_boots):
        if i % 20 == 0:
            print(i)
        boot_ind = resample(np.arange(num_data), n_samples=num_data, replace=True, stratify=traj_labs)
        mod_out_boot = [mod_out[j][boot_ind, :] for j in range(num_events)]
        boot_times = times[boot_ind, :]
        boot_labs = labs[boot_ind, :]
        results = eval_overall(mod_out_boot, boot_times, boot_labs, num_events, num_bin, num_extra_bin, term_events, event_ranks)
        
        c_glob[i, :] = results['C index']
        prop_glob[i, :] = results['Proposed']
        prop_loc[i] = results['Local proposed']
    
    print('C index: ', np.percentile(c_glob, [2.5, 50, 97.5], axis=0), np.average(c_glob, axis=0))
    print('Proposed: ', np.percentile(prop_glob, [2.5, 50, 97.5], axis=0), np.average(prop_glob, axis=0))
    print('Local proposed: ', np.percentile(prop_loc, [2.5, 50, 97.5], axis=0), np.average(prop_loc, axis=0))
    
    return 1


################################################################################################### 
'''
produce an event or non-event specific model (combines above 2 functions into 1)
'''
def get_model_and_output(method, train_package, test_package, val_package, params, hyperparams, verbose):
    if 'sim' in method:
        test_curves = produce_sim(test_package, params)
    elif 'hierarch' in method or 'direct' in method:
        test_curves = produce_model(method, train_package, val_package, test_package, params, hyperparams)
    return test_curves


################################################################################################### 
'''
main block 
'''
if __name__ == '__main__':
    print(':)')