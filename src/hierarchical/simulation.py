'''
simulation using ground truth
'''

import numpy as np

###################################################################################################
'''
the overall network
'''
class simulator():
    def __init__(self, num_time_bins, event_groups, min_time, max_time):
        self.num_bins = num_time_bins
        self.rank_groups = event_groups
        self.max_time = max_time
        self.min_time = min_time
        self.num_events = len(event_groups[0])
        self.num_extra_bin = 1
    
    def get_surv_curves(self, x, labs):
        return self.get_ground_truth(x, labs)
    
    def get_ground_truth(self, inp, labs): #follows the synthetic data generation code
        num_inp = inp.shape[0]
        num_time_samples = 1000
        num_feat = inp.shape[1] // (labs.shape[1] + 1)
        
        gamma_components = []
        gamma_const = [1, 1, 1]
        for i in range(self.num_events + 1):
            gamma_components.append(gamma_const[i] * np.ones((num_feat,)))
        gamma_components.append(gamma_const[-1] * np.ones((num_feat,)))
        
        distr_noise = 0.4 
        distr_noise2 = 0.4 
        
        raw_times = []
        raw_times2 = []
        time2_coeffs = 1 * np.array([0, 1, 1]) 
        surv_curves = []
        
        for i in range(self.num_events):
            raw_time = np.power(np.matmul(np.power(np.absolute(inp[:, :num_feat]), 1), gamma_components[0]), 2) + \
                   np.power(np.matmul(np.power(np.absolute(inp[:, (i + 1)*num_feat:(i+2)*num_feat]), 1), gamma_components[i + 1]), 2)
            raw_times.append(raw_time)
            raw_time2 = 1 * (time2_coeffs[2] * np.power(np.matmul(np.absolute(inp[:, (0)*num_feat:(1)*num_feat]), gamma_components[2]), 1))
            raw_times2.append(raw_time2)
            
        for i in range(self.num_events):
            event_curves = np.zeros((num_inp, self.num_bins + self.num_extra_bin + 1))
            for j in range(num_inp):
                times0 = np.random.lognormal(mean=np.log(raw_times[0][j]), sigma=distr_noise, size=(num_time_samples,)) 
                times1 = np.random.lognormal(mean=np.log(raw_times[1][j]), sigma=distr_noise, size=(num_time_samples,)) 
                if i == 1 and raw_times[0][j] <= raw_times[1][j]:
                    times2 = 1 * np.random.lognormal(mean=np.log(raw_times2[i][j]), sigma=distr_noise2, size=(num_time_samples,))
                    times = (np.tile(times1, (num_time_samples, 1)) + np.tile(times2, (num_time_samples, 1))).reshape(-1)
                elif i == 0 and raw_times[1][j] <= raw_times[0][j]:
                    times2 = 1 * np.random.lognormal(mean=np.log(raw_times2[i][j]), sigma=distr_noise2, size=(num_time_samples,))
                    times = (np.tile(times0, (num_time_samples, 1)) + np.tile(times2, (num_time_samples, 1))).reshape(-1)
                else:
                    times = [times0, times1][i]
                binned_times = ((times - self.min_time) / (self.max_time - self.min_time) * self.num_bins).astype(int)
                binned_times[binned_times > self.num_bins] = self.num_bins
                for k in range(self.num_bins):
                    event_curves[j, k + 1] = np.where(binned_times >= k)[0].shape[0] / binned_times.shape[0]
            surv_curves.append(event_curves)
        
        return surv_curves


'''
main block 
'''
if __name__ == '__main__':
    print(':)')  