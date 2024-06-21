import torch
from datetime import datetime
from utility.survival import cox_survival
import pandas as pd
import numpy as np
from utility.evaluation import LifelinesEvaluator
from utility.multi_event_ci import all_events_ci, global_ci, local_ci
# import sys,os
# sys.path.append('./SurvivalEVAL/')
# from Evaluator import LifelinesEvaluator

class MultiEventEvaluator():
    def __init__(self, data_test, data_train, model, config, device):
        x_test = data_test.drop(["y1_time", "y2_time", "y1_event", "y2_event"], axis=1).values
        self.x_test = torch.tensor(x_test, dtype=torch.float, device=device)
        self.data_test = data_test
        self.data_train = data_train
        self.model = model
        self.config = config
        
    def predict_survival_curves_gaussian(self):
        self.model.eval()
        start_time = datetime.now()
        with torch.no_grad():
            n_samples = self.config.n_samples_test
            logits_dists = self.model(self.x_test)
            logits_cpd1 = torch.stack([torch.reshape(logits_dists[0].sample(), (self.x_test.shape[0], 1)) for _ in range(n_samples)])
            logits_cpd2 = torch.stack([torch.reshape(logits_dists[1].sample(), (self.x_test.shape[0], 1)) for _ in range(n_samples)])
            logits_mean1 = torch.mean(logits_cpd1, axis=0)
            logits_mean2 = torch.mean(logits_cpd2, axis=0)
            outputs = [logits_dists[0].mean, logits_dists[1].mean]
            end_time = datetime.now()
            inference_time = end_time - start_time
            if self.config.verbose:
                print(f"Inference time: {inference_time.total_seconds()}")
            n_events = 2
            event_survival_curves = list()
            for i in range(n_events):
               survival_curves = cox_survival(self.model.baseline_survivals[i], outputs[i])
               survival_curves = survival_curves.squeeze()
               survival_curves[:,0] = 1
               survival_curves = pd.DataFrame(survival_curves, columns=np.array(self.model.time_bins[i]))
               event_survival_curves.append(survival_curves)
            return event_survival_curves
        
    def predict_survival_curves(self):
        self.model.eval()
        start_time = datetime.now()
        with torch.no_grad():
            pred = self.model(self.x_test)
            end_time = datetime.now()
            inference_time = end_time - start_time
            if self.config.verbose:
                print(f"Inference time: {inference_time.total_seconds()}")
            n_events = len(pred)
            event_survival_curves = list()
            for i in range(n_events):
               survival_curves = cox_survival(self.model.baseline_survivals[i], pred[i])
               survival_curves = survival_curves.squeeze()
               survival_curves[:,0] = 1
               survival_curves = pd.DataFrame(survival_curves, columns=np.array(self.model.time_bins[i]))
               event_survival_curves.append(survival_curves)
            return event_survival_curves

    def calculate_ci(self, surv_preds, event_id):
        time_label, event_label = f"y{event_id+1}_time", f"y{event_id+1}_event"
        evaluator = LifelinesEvaluator(surv_preds.T, self.data_test[time_label], self.data_test[event_label],
                                       self.data_train[time_label], self.data_train[event_label])
        return evaluator.concordance()[0]

    def calculate_multi_ci(self, surv_preds):
        results = {}
        results['CI'] = all_events_ci(surv_preds, self.data_test[time_label], self.data_test[event_label])
        results['global_CI'] = global_ci(surv_preds, self.data_test[time_label], self.data_test[event_label])
        results['local_CI'] = local_ci(surv_preds, self.data_test[time_label], self.data_test[event_label])
        return results
    
    def calculate_mae(self, surv_preds, event_id, method):
        if method == "Hinge":
            time_label, event_label = f"y{event_id+1}_time", f"y{event_id+1}_event"
            evaluator = LifelinesEvaluator(surv_preds.T, self.data_test[time_label], self.data_test[event_label],
                                           self.data_train[time_label], self.data_train[event_label])
            return evaluator.mae(method=method)
        else:
            raise NotImplementedError()

