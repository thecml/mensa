import pandas as pd
import config as cfg
from pathlib import Path
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def annotate_event(group, event_col):
    event_observed = 1 if any(group[event_col] == 1) else 0
    if event_observed:
        delta_sum_observed = group.loc[group[event_col] == 1, 'ALSFRS_Delta'].iloc[0]
    else:
        delta_sum_observed = group['ALSFRS_Delta'].max()
    return pd.Series({'Delta_Observed': delta_sum_observed, 'Event': event_observed})

def annotate_left_censoring(row, event_name):
    if row[f'TTE_{event_name}'] == 0: # check if left-censored
        tte = row['Onset_Delta'] # left-censored at onset time
        event = -1
    else:
        tte = row[f'TTE_{event_name}']
        event = row[f'Event_{event_name}']
    return pd.Series({f'TTE_{event_name}': tte,
                      f'Event_{event_name}': event})

def convert_weight(row):
    if row['Weight_Units'] in ['Kilograms', 'kg']:
        return row['Weight']  # Keep the value as is
    elif row['Weight_Units'] == 'Pounds':
        return row['Weight'] * 0.453592  # Convert pounds to kg
    else:
        return None  # Handle any unexpected values

def convert_height(row):
    if row['Height_Units'] in ['Centimeters', 'cm']:
        return row['Height']  # Keep the value as is
    elif row['Height_Units'] == 'Inches':
        return row['Height'] * 2.54  # Convert inches to cm
    else:
        return None  # Handle any unexpected values

if __name__ == "__main__":
    alsfrs_fn = "PROACT_ALSFRS.csv"
    alshistory_fn = 'PROACT_ALSHISTORY.csv'
    fvc_fn = 'PROACT_FVC.csv'
    handgrip_str_fn = 'PROACT_HANDGRIPSTRENGTH.csv'
    muscle_str_fn = 'PROACT_MUSCLESTRENGTH.csv'
    riluzole_fn = 'PROACT_RILUZOLE.csv'
    elescorial_fn = 'PROACT_ELESCORIAL.csv'
    deathdata_fn = 'PROACT_DEATHDATA.csv'
    demographics_fn = 'PROACT_DEMOGRAPHICS.csv'
    vital_signs_fn = 'PROACT_VITALSIGNS.csv'
        
    alsfrs_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, alsfrs_fn))
    history_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, alshistory_fn))
    fvc_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, fvc_fn))
    handgrip_str_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, handgrip_str_fn))
    muscle_str_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, muscle_str_fn))
    riluzole_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, riluzole_fn))
    elescorial_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, elescorial_fn))
    deathdata_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, deathdata_fn))
    demographics_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, demographics_fn))
    vital_signs_df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, vital_signs_fn))

    # Create dataframe with subjects
    df = pd.DataFrame()
    df['subject_id'] = alsfrs_df['subject_id'].unique()
    
    # Sort ALSFRS scores by id and delta
    alsfrs_df = alsfrs_df.sort_values(by=['subject_id', 'ALSFRS_Delta'])
    
    # Record onset delta
    onset_delta = history_df[['subject_id', 'Onset_Delta']].copy(deep=True)
    onset_delta['Onset_Delta'] = onset_delta['Onset_Delta'].map(abs)
    df = pd.merge(df, onset_delta, on="subject_id", how='left')
    df = df.dropna(subset='Onset_Delta')
    
    # Annotate events
    threshold = 2
    event_names = ['Speech', 'Swallowing', 'Handwriting', 'Walking']
    event_cols = ['Q1_Speech', 'Q3_Swallowing', 'Q4_Handwriting', 'Q8_Walking']
    for event_name, event_col in zip(event_names, event_cols):
        alsfrs_df[f'Event_{event_name}'] = (alsfrs_df[event_col] <= threshold).astype(int)
        event_df = alsfrs_df.groupby('subject_id').apply(annotate_event, f'Event_{event_name}').reset_index()
        event_df = event_df.rename({'Delta_Observed': f'TTE_{event_name}', 'Event': f'Event_{event_name}'}, axis=1)
        df = pd.merge(df, event_df, on="subject_id", how='left')
        df[[f'TTE_{event_name}', f'Event_{event_name}']] = df.apply(lambda x: annotate_left_censoring(x, event_name), axis=1)
        
    # Record total ALSFRS-R score at baseline
    df = pd.merge(df, alsfrs_df[['subject_id', 'ALSFRS_R_Total']] \
        .drop_duplicates(subset='subject_id'), on="subject_id", how='left')
        
    # Record demographics at baseline
    demographics_df['Age'] = demographics_df['Age'] # age
    demographics_df['Race_Caucasian'] = demographics_df['Race_Caucasian'].fillna(0)
    sex_map = {"Male": "Male", "M": 'Male', "Female": "Female", "F": 'Female'}
    demographics_df['Sex'] = demographics_df['Sex'].map(sex_map)
    df = pd.merge(df, demographics_df[['subject_id', 'Age', 'Race_Caucasian', 'Sex']], on="subject_id", how='left')
    
    # Record vital signs
    vital_signs_df['Weight'] = vital_signs_df.apply(convert_weight, axis=1)
    vital_signs_df['Height'] = vital_signs_df.apply(convert_height, axis=1)
    observed_weights = vital_signs_df.groupby('subject_id')['Weight'].first().reset_index()
    observed_heights = vital_signs_df.groupby('subject_id')['Height'].first().reset_index()
    df = pd.merge(df, observed_weights[['subject_id', 'Weight']] \
         .drop_duplicates(subset='subject_id'), on="subject_id", how='left') # weight
    df = pd.merge(df, observed_heights[['subject_id', 'Height']] \
         .drop_duplicates(subset='subject_id'), on="subject_id", how='left') # height
    min_weight, max_weight = 40, 120  # Normal human weight in kg
    min_height, max_height = 150, 200  # Normal human height in cm
    df['Weight'] = df['Weight'].where(df['Weight'].between(min_weight, max_weight), None)
    df['Height'] = df['Height'].where(df['Height'].between(min_height, max_height), None)

    # Calculate BMI
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Record site of onset
    soo = history_df.drop_duplicates(subset='subject_id')[['subject_id',
                                                           'Site_of_Onset',
                                                           'Onset_Delta']].copy(deep=True)
    soo['Site_of_Onset'] = soo['Site_of_Onset'].str.replace('Onset: ', '', regex=False)
    soo['Site_of_Onset'] = soo['Site_of_Onset'].str.replace('Limb and Bulbar', 'LimbAndBulbar', regex=False)
    df = pd.merge(df, soo[['subject_id', 'Site_of_Onset']], on="subject_id", how='left')
    
    # Calculate disease progression rate
    df['DiseaseProgressionRate'] = (48 - df['ALSFRS_R_Total']) / (abs(df['Onset_Delta'])/30)
    
    # Record Riluzole use
    riluzole_use = riluzole_df[['subject_id', 'Subject_used_Riluzole']].copy(deep=True)
    df = pd.merge(df, riluzole_use, on="subject_id", how='left')
    
    # Record Elescorial information
    elescorial_criteria = elescorial_df[['subject_id', 'el_escorial']].copy(deep=True)
    elescorial_criteria.rename({'el_escorial': 'El_escorial'}, axis=1, inplace=True)
    df = pd.merge(df, elescorial_criteria, on="subject_id", how='left')
    
    # Record time of death
    df = pd.merge(df, deathdata_df, on="subject_id", how='left')
    df = df.rename({'Subject_Died': 'Event_Death', 'Death_Days': 'TTE_Death'}, axis=1)
    df['Event_Death'] = df['Event_Death'].fillna(False)
    tte_columns = [col for col in df.columns if col.startswith('TTE_')]
    df.loc[df['TTE_Death'].isna(), 'TTE_Death'] = df.loc[df['TTE_Death'].isna(), tte_columns].apply(lambda x: max(x), axis=1)
    df['Event_Death'] = df['Event_Death'].replace({'Yes': True, 'No': False})
    
    # Record FVC
    cols = [f'Subject_Liters_Trial_{i}' for i in range(1,4)]
    fvc_df['FVC_Min'] = fvc_df[cols].min(axis=1)
    fvc_df['FVC_Max'] = fvc_df[cols].max(axis=1)
    fvc_df['FVC_Mean'] = fvc_df[cols].mean(axis=1)
    fvc_df = fvc_df.drop_duplicates(subset='subject_id')
    df = pd.merge(df, fvc_df[['subject_id', 'FVC_Mean']], on="subject_id", how="left")
    
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
    df.to_csv(f'{cfg.DATA_DIR}/proact_processed.csv')
