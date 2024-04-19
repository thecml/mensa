import pandas as pd
import numpy as np
import config as cfg
from pathlib import Path

def annotate_event(group, event_col):
    event_observed = True if any(group[event_col] == 1) else False
    if event_observed:
        delta_sum_observed = group.loc[group[event_col] == 1, 'DeltaSum'].iloc[0]
    else:
        delta_sum_observed = group['DeltaSum'].max()
    return pd.Series({'DeltaSum_Observed': delta_sum_observed, 'Event': event_observed})

if __name__ == "__main__":
    alsfrs_fn = "PROACT_ALSFRS.csv"
    alshistory_fn = 'PROACT_ALSHISTORY.csv'
    fvc_fn = 'PROACT_FVC.csv'
    handgrip_str_fn = 'PROACT_HANDGRIPSTRENGTH.csv'
    muscle_str_fn = 'PROACT_MUSCLESTRENGTH.csv'
    riluzole_fn = 'PROACT_RILUZOLE.csv'
    elescorial_fn = 'PROACT_ELESCORIAL.csv'
        
    alsfrs_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, alsfrs_fn))
    history_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, alshistory_fn))
    fvc_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, fvc_fn))
    handgrip_str_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, handgrip_str_fn))
    muscle_str_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, muscle_str_fn))
    riluzole_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, riluzole_fn))
    elescorial_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, elescorial_fn))

    # Create dataframe with subjects
    df = pd.DataFrame()
    df['subject_id'] = alsfrs_df['subject_id'].unique()
    
    # Annotate events
    threshold = 2
    event_names = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
    event_cols = ['Q1_Speech', 'Q3_Swallowing', 'Q4_Handwriting', 'Q8_Walking']
    alsfrs_df['DeltaSum'] = alsfrs_df.groupby('subject_id')['ALSFRS_Delta'].cumsum()
    for event_name, event_col in zip(event_names, event_cols):
        alsfrs_df[f'Event_{event_name}'] = (alsfrs_df[event_col] <= threshold).astype(int)
        event_df = alsfrs_df.groupby('subject_id').apply(annotate_event, f'Event_{event_name}').reset_index()
        event_df = event_df.rename({'DeltaSum_Observed':f'{event_name}_Observed',
                                    'Event': f'{event_name}_Event'}, axis=1)
        #event_df = event_df.loc[event_df[f'{event_name}_Observed']]
        df = pd.merge(df, event_df, on="subject_id", how='left').dropna()
    
    # Record site of onset
    filter_col = [col for col in history_df if col.startswith('Site_of_Onset__')]
    history_df['SOO'] = history_df[filter_col].values.argmax(1)+1
    soo_map = {0: 'Bulbar', 1: 'Limb', 2: 'LimbAndBulbar', 3: 'Other', 4: 'Other_Specify', 5: 'Spine'}
    history_df['SOO'] = history_df['SOO'].map(soo_map)
    history_df = history_df.drop_duplicates(subset='subject_id')
    df = pd.merge(df, history_df[['subject_id', 'SOO']], on="subject_id", how='left')
        
    # Record diagnosis delta
    diagnosis_delta = history_df[['subject_id', 'Diagnosis_Delta']].copy(deep=True)
    diagnosis_delta['Diagnosis_Delta'] = diagnosis_delta['Diagnosis_Delta'].map(abs)
    df = pd.merge(df, diagnosis_delta, on="subject_id", how='left')
    
    # Record Riluzole use
    riluzole_use = riluzole_df[['subject_id', 'Subject_used_Riluzole']].copy(deep=True)
    df = pd.merge(df, riluzole_use, on="subject_id", how='left')
    
    # Record Elescorial information
    elescorial_criteria = elescorial_df[['subject_id', 'el_escorial']].copy(deep=True)
    elescorial_criteria.rename({'el_escorial': 'El_escorial'}, axis=1, inplace=True)
    df = pd.merge(df, elescorial_criteria, on="subject_id", how='left')
    
    # Record FVC
    cols = [f'Subject_Liters_Trial_{i}' for i in range(1,4)]
    fvc_df['FVC_Min'] = fvc_df[cols].min(axis=1)
    fvc_df['FVC_Max'] = fvc_df[cols].max(axis=1)
    fvc_df['FVC_Mean'] = fvc_df[cols].mean(axis=1)
    fvc_df = fvc_df.drop_duplicates(subset='subject_id')
    df = pd.merge(df, fvc_df[['subject_id', 'FVC_Min', 'FVC_Max', 'FVC_Mean']], on="subject_id", how="left")
    
    # Record handgrip strength
    handgrip_str_df = handgrip_str_df.drop_duplicates(subset='subject_id').copy(deep=True)
    handgrip_str_df.rename({'Test_Result': 'Handgrip_Strength'}, axis=1, inplace=True)
    df = pd.merge(df, handgrip_str_df[['subject_id', 'Handgrip_Strength']], on="subject_id", how="left")
    
    # Record muscle strength
    muscle_str_df = muscle_str_df.loc[muscle_str_df['MS_Delta'] == 0] # use first test only
    muscle_str_df = muscle_str_df[['subject_id', 'Test_Name', 'Test_Location', 'Test_Result']].copy(deep=True)
    muscle_str_df['Test_Location'] = muscle_str_df['Test_Location'].str.replace(' JOINT', '')
    muscle_str_df['Test_Location'] = muscle_str_df['Test_Location'].str.replace(' MUSCLE', '')
    muscle_str_df['Test_Type'] = muscle_str_df['Test_Name'].str.split(',', expand=True)[1]
    muscle_str_df.rename({'Test_Location': 'Muscle_Test_Location', 'Test_Result': 'Muscle_Test_Strength',
                          'Test_Type': 'Muscle_Test_Type'}, axis=1, inplace=True)
    muscle_test_res = muscle_str_df.groupby(['subject_id', 'Muscle_Test_Location', 'Muscle_Test_Type'])['Muscle_Test_Strength'] \
                      .mean().unstack(level=1).groupby('subject_id').mean().reset_index()
    test_cols = muscle_test_res.drop(['subject_id'], axis=1).columns
    muscle_test_res = pd.concat([muscle_test_res['subject_id'], muscle_test_res[test_cols].add_suffix('_Strength')], axis=1)
    muscle_test_res.columns = muscle_test_res.columns.str.replace(' ', '_')
    df = pd.merge(df, muscle_test_res, on="subject_id", how='left')

    # Drop subject id
    df = df.drop('subject_id', axis=1)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save df
    df.to_csv(f'{cfg.DATA_DIR}/als.csv')
