import numpy as np
import pandas as pd

from pymsm.datasets import prep_ebmt_long, prep_aidssi, prep_rotterdam, prep_covid_hosp_data

def convert_Multi_stage_to_Multi_event(competing_risk_dataset):
    sample_ids = competing_risk_dataset['sample_id'].unique()
    Y_temp_dict = {'sample_id': np.nan}
    for event_id, event_name in state_labels.items():
        # print (event_id)
        Y_temp_dict[str(event_id)+'_event'] = 0
        Y_temp_dict[str(event_id)+'_time'] = np.nan
    Y_dict_list = []
    for sample_id in sample_ids:
        temp_df = competing_risk_dataset[competing_risk_dataset['sample_id'] == sample_id]
        Y_dict = Y_temp_dict.copy()
        Y_dict['sample_id'] = sample_id
        max_time = max(temp_df['time_transition_to_target'])
        for event_id, event_name in state_labels.items():
            Y_dict[str(event_id)+'_time'] = max_time
        for index, row in temp_df.iterrows():
            target_state = str(int(row['target_state']))
            if target_state != '0':
                Y_dict[target_state+'_event'] = 1
                Y_dict[target_state+'_time'] = row['time_transition_to_target']        
        Y_dict_list.append(Y_dict)

    Y_labels = pd.DataFrame.from_records(Y_dict_list)
    X_features = competing_risk_dataset[['sample_id'] + covariate_cols.tolist()].drop_duplicates('sample_id')
    return X_features, Y_labels

database_name = 'ebmt'
if database_name == 'ebmt':
    competing_risk_dataset, covariate_cols, state_labels = prep_ebmt_long()
elif database_name == 'aidssi':
    competing_risk_dataset, covariate_cols, state_labels = prep_aidssi()
elif database_name == 'covid_hosp_data':    
    competing_risk_dataset, covariate_cols, state_labels = prep_covid_hosp_data()

X, Y = convert_Multi_stage_to_Multi_event(competing_risk_dataset)
all_df = X.merge(Y, on='sample_id')
all_df.to_csv('./ebmt.csv')
