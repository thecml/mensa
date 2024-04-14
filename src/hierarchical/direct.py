'''
predicts survival curves at the original time scale only
'''

import numpy as np
import itertools
import torch
import torch.nn as nn

from hierarchical import util


###################################################################################################
'''
event network
'''
class event_network(nn.Module):
    def __init__(self, layer_sizes, num_bins, get_event_prob, multitask, blockable=False):
        super(event_network, self).__init__()
        self.x_layers = []
        self.p_layers = []
        self.layer_sizes = layer_sizes
        self.multitask = multitask
        self.blockable = blockable #true means the event's occurrence can be prevented by another event
        self.get_event_probs = get_event_prob
        
        total_size = 1
        num_reps = 1
        for i in range(len(layer_sizes) - 1):
            self.x_layers.append([])
            self.p_layers.append([])
            total_size = total_size * layer_sizes[i][1]    
            for _ in range(num_reps):
                if num_bins % layer_sizes[i][1] != 0 or (i == len(layer_sizes) - 1 and total_size != num_bins):
                    raise(Exception('Invalid divisions'))     
                in_size = layer_sizes[0][0]
                self.x_layers[i].append(nn.Linear(in_size, layer_sizes[i + 1][0]))
                self.p_layers[i].append(nn.Linear(layer_sizes[i + 1][0], layer_sizes[i + 1][1]))
            num_reps = num_reps * layer_sizes[i + 1][1] 
        self.x_layers = [self.x_layers[-1]]
        self.p_layers = [self.p_layers[-1]]
        
        self.event_indicator = nn.Linear(layer_sizes[0][0], 1)
        if blockable:
            self.event_indicator = nn.Linear(layer_sizes[0][0], 2)
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.x_layers)):
            for j in range(len(self.x_layers[i])):
                nn.init.xavier_uniform_(self.x_layers[i][j].weight)
        for i in range(len(self.p_layers)):
            for j in range(len(self.p_layers[i])):
                nn.init.xavier_uniform_(self.p_layers[i][j].weight)
        nn.init.xavier_uniform_(self.event_indicator.weight)
            
    def get_parameters(self):
        params = [self.event_indicator.parameters()]
        for i in range(len(self.x_layers)):
            for j in range(len(self.x_layers[i])):
                params.append(self.x_layers[i][j].parameters())
        for i in range(len(self.p_layers)):
            for j in range(len(self.p_layers[i])):
                params.append(self.p_layers[i][j].parameters())
                
        return params
    
    def forward(self, inp, return_pre=False):
        got_event = nn.Sigmoid()(self.event_indicator(inp))
        num_data = inp.shape[0]
        
        pre_probs = []
        for i in range(len(self.p_layers)):
            pre_probs.append([])
            for j in range(len(self.p_layers[i])):
                net_inp = inp
                output = self.activation(self.x_layers[i][j](net_inp))
                prob = self.p_layers[i][j](output)
                pre_probs[i].append(prob)
        for i in range(len(self.p_layers)):       
            pre_probs[i] = torch.cat(pre_probs[i], dim=1)
        
        probs = [self.softmax(pre_probs[-1])] #only use the final grain, no guidance from coarser grains
        if self.get_event_probs: #predict if event occurred within horizon
            if self.blockable:
                probs[0] = probs[0] * torch.prod(got_event, dim=1).view(-1, 1)
                current_prob = torch.sum(probs[0], dim=1).view(-1, 1)
                remaining_prob = (got_event[:, 1]).view(-1, 1)
                remaining_prob = remaining_prob - current_prob
                probs[0] = torch.cat([probs[0], remaining_prob], dim=1)
            else:
                probs[0] = torch.cat([probs[0]*got_event, 1 - got_event], dim=1)
        elif self.blockable:
            probs[0] = probs[0] * got_event[:, 1].view(-1, 1)
            
        #here, coarse probabilities are obtained from summing over the fine grained ones (instead of being first predicted)
        for i in range(len(self.layer_sizes)-2): 
            num_bin = self.layer_sizes[i+1][1]
            bin_size = probs[-1].shape[1] // self.layer_sizes[i+1][1]
            coarse_probs = torch.zeros(num_data, num_bin)
            for j in range(num_bin):
                coarse_probs[:, j] = torch.sum(probs[-1][:, bin_size*j:bin_size*(j+1)], dim=1)
            if self.get_event_probs and self.blockable:
                current_prob = torch.sum(coarse_probs, dim=1).view(-1, 1)
                remaining_prob = (got_event[:, 1]).view(-1, 1)
                remaining_prob = remaining_prob - current_prob
                coarse_probs = torch.cat([coarse_probs, remaining_prob], dim=1)
            elif self.get_event_probs:
                coarse_probs = torch.cat([coarse_probs, 1 - got_event], dim=1)
            probs = [coarse_probs] + probs
         
        if return_pre:
            return pre_probs, probs 
        return probs
        

'''
the overall network
'''
class direct_network(nn.Module):
    def __init__(self, layer_sizes, event_net_sizes, num_events, num_time_bins, event_groups, \
                 extra_bin, term_events, ranks, multitask=True, dh=False):
        super(direct_network, self).__init__()
        self.num_bins = num_time_bins
        self.num_extra_bin = extra_bin
        self.multi_task = multitask
        self.dh = dh
        
        self.terminal_events = term_events
        self.ranks = ranks
        self.rank_groups = event_groups
        self.num_event_groups = len(event_groups.keys())
        self.num_events = num_events
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        #joint layers input passes through before event specific networks
        self.main_layers = []
        if multitask:
            for i in range(len(layer_sizes) - 1):
                self.main_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        else:
            for i in range(num_events):
                for j in range(len(layer_sizes) - 1):
                    self.main_layers.append(nn.Linear(layer_sizes[j], layer_sizes[j + 1]))
         
        #event specific networks, 1 for each event
        self.event_networks = []
        for _ in range(num_events):
            event_net = event_network(event_net_sizes, num_time_bins, extra_bin > 0, multitask, len(ranks[i]) > 0)
            self.event_networks.append(event_net)
           
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.main_layers)):
            nn.init.xavier_uniform_(self.main_layers[i].weight)
    
    def forward(self, inp, softmax=True):
        #joint layers
        hidden_outputs = []
        hidden_output = inp
        if self.multi_task:
            for i in range(len(self.main_layers)):
                hidden_output = self.main_layers[i](hidden_output)
                hidden_output = self.activation(hidden_output)
            hidden_outputs = hidden_output
        else:
            num_layer_per_event = len(self.main_layers) / self.num_events
            for i in range(len(self.main_layers)):
                if i % num_layer_per_event == 0:
                    hidden_output = self.main_layers[i](inp)
                else:
                    hidden_output = self.main_layers[i](hidden_output)
                hidden_output = self.activation(hidden_output)
                if i % num_layer_per_event == num_layer_per_event - 1:
                    hidden_outputs.append(hidden_output)
        
        #event-specific layers
        event_net_inp = hidden_outputs
        event_net_out = []
        event_net_out_pre = []
        for i in range(self.num_events):
            if not self.multi_task: 
                event_out = self.event_networks[i](event_net_inp[i], return_pre=True)
            else:
                event_out = self.event_networks[i](event_net_inp, return_pre=True)
            event_net_out.append(event_out[1])
            event_net_out_pre.append(event_out[0])
        
        if self.dh: #if using deephit
            dh_out = event_net_out_pre[0][-1]
            for i in range(1, self.num_events):
                dh_out = torch.cat((dh_out, event_net_out_pre[i][-1]), dim=1)
            dh_out = nn.Softmax(dim=1)(torch.Tensor(dh_out))
            dh_out2 = []
            dh_bins = int(dh_out.shape[1] / self.num_events)
            for i in range(self.num_events):
                dh_out2.append([dh_out[:, i*dh_bins:(i+1)*dh_bins]])
            return dh_out2
            
        return event_net_out
    
    def get_parameters(self):
        net_params = [self.parameters()] + [sub.parameters() for sub in self.main_layers]
        event_net_params = []
        for i in range(self.num_events):
            event_net_params += self.event_networks[i].get_parameters()
        return itertools.chain.from_iterable(net_params + event_net_params)


###################################################################################################
'''
loss function based on cif
'''
class direct_loss(nn.Module):
    def __init__(self, term_event, groups, params):
        super(direct_loss, self).__init__()
        self.terminal_events = term_event
        self.rank_groups = groups
        
        self.back_c = params[0]
        self.hierarch = params[1]
        
        self.alpha_disc = params[2]
        self.sigma_disc = params[3]
    
    #penalizes incorrect rankings, not being used right now, similar the proposed evaluation metric
    def get_c_loss(self, model, complete_output, event_times, labs, event_ranking, num_bin, num_extra_bin):
        num_samples = complete_output[0].shape[0]
        num_events = labs.shape[1]
        sigma = self.sigma_disc
        
        penalty = 0
        total_pairs = 0
        
        #loop through events
        for i in range(num_events):
            event_out = complete_output[i]
            for j in range(num_samples):
                had_event = labs[j, i] == 1
                if had_event:
                    event_time = min(int(event_times[j, i]) + 1, num_bin + num_extra_bin - 1)
                    f_compare = np.where(event_times[:, i] > event_time - 1)[0]
                    f_compare = np.union1d(f_compare, np.where(np.logical_and(event_times[:, i] == event_time - 1, labs[:, i] == 0))[0])
                    total_pairs += f_compare.shape[0]
                    time_window = (event_times[f_compare, i] + 1 - event_time)
                    time_window[time_window == 0] += 1
                    time_window = time_window.view(-1, 1)
                    comp_risk = 1 - torch.sum(event_out[f_compare, :event_time], dim=1)
                    reference_risk = 1 - torch.sum(event_out[j, :event_time])
                    penalize = np.where(comp_risk < reference_risk)[0]
                    penalty += torch.sum(torch.exp(-(comp_risk[penalize] - reference_risk) / sigma)) 
                    
                #backward
                if self.back_c:
                    if event_times[j, i] > 0:
                        event_time = min(int(event_times[j, i]) + 1, num_bin + num_extra_bin - 1)
                        if had_event:
                            event_time = min(int(event_times[j, i]), num_bin + num_extra_bin - 1)
                        b_compare = np.where(np.logical_and(event_times[:, i] <= event_time - 1, labs[:, i] == 1))[0]
                        time_window = (event_times[b_compare, i] + 1 - event_time)
                        time_window[time_window == 0] += 1
                        time_window = time_window.view(-1, 1)
                        
                        comp_risk = 1 - torch.sum(event_out[b_compare, :event_time], dim=1)
                        reference_risk = 1 - torch.sum(event_out[j, :event_time])
                        penalize = np.where(comp_risk > reference_risk)[0]
                        penalty += torch.sum(torch.exp((comp_risk[penalize] - reference_risk) / sigma)) 
                    
        return self.alpha_disc * penalty 
    
    #main loss function
    def forward(self, all_outputs, all_labels, event_ordering_times, model, inp):
        total_loss = torch.Tensor([0])
        event_weights = np.ones((model.num_events,))
        type_weights = np.ones((model.num_events,))
        num_events = model.num_events
        num_time_bins = model.num_bins
        
        for b in range(len(all_outputs[0])): #get loss for each granularity
            num_bin_b = (num_time_bins / (all_outputs[0][b].shape[1] - model.num_extra_bin))
            num_bin_b_total = model.num_bins // num_bin_b
            include_samples = np.arange(all_outputs[0][0].shape[0])   
            
            for a in range(num_events): #get loss for each event
                if b != len(all_outputs[a]) - 1 and not self.hierarch:
                    continue
                output = all_outputs[a][b]
                labels = all_labels[:, a]
                
                time_to_a = event_ordering_times[:, a] // num_bin_b
                events = torch.Tensor([1])
                
                censored_ind = np.where(labels == 0)[0]
                num_censored = censored_ind.shape[0]
                uncensored_ind = np.where(labels == 1)[0]
                num_uncensored = uncensored_ind.shape[0]
                if b < len(all_outputs[a]) - 1 and model.num_extra_bin > 0:
                    exclude = np.where(np.logical_and(time_to_a == all_outputs[0][b].shape[1] - model.num_extra_bin - 1, event_ordering_times[:, a] < num_time_bins - 1))[0]
                    censored_ind = np.setdiff1d(censored_ind, exclude)
                
                event_cen = np.array([])
                if len(model.ranks[a]) > 0: #if blockable
                    had_term = np.where(all_labels[:, model.ranks[a][0]] == 1)[0]
                    event_cen = np.intersect1d(censored_ind, had_term)
                    censored_ind = np.setdiff1d(censored_ind, event_cen)
                
                #uncensored loss - maximize likelihood of first event
                uncensored_loss = 0
                if num_uncensored > 0:
                    uncensored_loss = util.get_uncensored_loss(num_time_bins, model.num_extra_bin, output, uncensored_ind, labels, events, time_to_a)
                    uncensored_loss = uncensored_loss * event_weights[a] * type_weights[0]
                
                #censored loss - maximize likelihood of occurrence after observation for all events
                censored_loss = 0
                if num_censored > 0:
                    censored_loss = censored_loss + util.get_censored_loss(num_time_bins, model.num_extra_bin, output, censored_ind, time_to_a, 1)#, events_had_all)
                    censored_loss = censored_loss * type_weights[1] * event_weights[a]
                
                #event censorship, for events that can be blocked (comp/semicomp risks)
                event_censored_loss = 0
                if event_cen.shape[0] > 0 and len(model.ranks[a]) > 0:
                    probs = 1 - torch.sum(output[event_cen, :int(num_bin_b_total + model.num_extra_bin)], dim=1)
                    probs = probs[probs > 0]
                    if probs.shape[0] > 0:
                        event_censored_loss = -torch.sum(torch.log(probs)) / (probs.shape[0])
                
                total_loss = total_loss + (uncensored_loss + censored_loss + event_censored_loss) 
                include_samples = np.intersect1d(np.union1d(uncensored_ind, censored_ind), include_samples)
        
        last_outs = [all_outputs[i][-1] for i in range(len(all_outputs))]
        c_loss = self.get_c_loss(model, last_outs, event_ordering_times, all_labels, self.rank_groups, model.num_bins, model.num_extra_bin)
        total_loss += c_loss
        
        return total_loss


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')