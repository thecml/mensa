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
    def __init__(self, data_test, data_train):
        self.data_test = data_test
        self.data_train = data_train

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

