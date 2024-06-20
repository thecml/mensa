import sys,os
# sys.path.append('./SurvivalEVAL/')
from evaluator import LifelinesEvaluator
# from SurvivalEVAL.Evaluator import LifelinesEvaluator

def sort_by_time(surv_pred_event, temp_test_time, temp_test_event):
    '''
    Sort by time to make c index calculate faster @Shi-ang may add into the Evaluator
    surv_pred_event: Dataframe
    temp_test_time: np.array
    temp_test_event: np.array
    '''
    surv_pred_event['time'] = temp_test_time
    surv_pred_event['event'] = temp_test_event
    surv_pred_event = surv_pred_event.sort_values('time')

    temp_test_time = surv_pred_event['time'].to_numpy()
    temp_test_event = surv_pred_event['event'].to_numpy()
    surv_pred_event = surv_pred_event.drop(['time', 'event'], axis=1)
    return surv_pred_event, temp_test_time, temp_test_event

def all_events_ci(mod_out, test_time, test_event):
    '''
    all events
    mod_out: List of surv pred
    test_time: np.array of float/int #patient, #event
    test_event: np.array of binary #patient, #event
    '''
    surv_pred_event = pd.concat([pd.DataFrame(surv_pred) for surv_pred in mod_out])
    surv_pred_event = surv_pred_event.reset_index(drop=True)
    temp_test_time = np.concatenate([test_time[:, event_id] for event_id in range(test_time.shape[1])])
    temp_test_event = np.concatenate([test_event[:, event_id] for event_id in range(test_event.shape[1])])

    surv_pred_event, temp_test_time, temp_test_event = sort_by_time(surv_pred_event, temp_test_time, temp_test_event)
    evaluator = LifelinesEvaluator(surv_pred_event.T, temp_test_time, temp_test_event)

    cindex, _, _ = evaluator.concordance()
    return cindex

def global_ci(mod_out, test_time, test_event):
    '''
    each events
    mod_out: List of surv pred
    test_time: np.array of float/int #patient, #event
    test_event: np.array of binary #patient, #event
    '''
    cindex_list = []
    for event_id in range(len(mod_out)):
        surv_pred_event = pd.DataFrame(mod_out[event_id])
        temp_test_time = test_time[:,event_id]
        temp_test_event = test_event[:,event_id]

        surv_pred_event, temp_test_time, temp_test_event = sort_by_time(surv_pred_event, temp_test_time, temp_test_event)
        evaluator = LifelinesEvaluator(surv_pred_event.T, temp_test_time, temp_test_event)

        cindex, _, _ = evaluator.concordance()
        cindex_list.append(cindex)
    print ('each_events CI:', cindex_list)
    return np.mean(cindex_list)

def local_ci(mod_out, test_time, test_event):
    '''
    each patient
    mod_out: List of surv pred
    test_time: np.array of float/int #patient, #event
    test_event: np.array of binary #patient, #event
    '''
    cindex_list = []
    for patient_id in range(test_time.shape[0]):
        surv_pred_patient = np.column_stack([mod_out[event_index][patient_id, :] for event_index in range(len(mod_out))]).T

        surv_pred_event = pd.DataFrame(surv_pred_patient)
        temp_test_time = test_time[patient_id,:]
        temp_test_event = test_event[patient_id,:]
        if np.sum(temp_test_event) != 0:

            zero_indices = np.where(temp_test_event == 0)[0]
            max_value = np.max(temp_test_time)
            for idx in zero_indices:
                if temp_test_time[idx] < max_value:
                    temp_test_time[idx] = max_value+1
            # print (evaluator.predicted_event_times, temp_test_time, temp_test_event)
            evaluator = LifelinesEvaluator(surv_pred_event.T, temp_test_time, temp_test_event)
            cindex, _, _ = evaluator.concordance()
            cindex_list.append(cindex)
            # print (cindex)
    return np.mean(cindex_list)

# print ("all_event C index", all_events_C_index(mod_out, test_time, test_event))
# print ("global C index", global_C_index(mod_out, test_time, test_event))
# print ("local C index", local_C_index(mod_out, test_time, test_event))