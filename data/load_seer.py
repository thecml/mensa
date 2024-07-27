import pandas as pd
import numpy as np
import paths as pt
from pathlib import Path
from utility.survival import convert_to_structured
from tools.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from tools.baysurv_builder import make_coxnet_model
from utility.metrics import concordance_index_censored
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # Load data
    path = Path.joinpath(pt.DATA_DIR, "seer_breast_2004_2015.csv")
    df = pd.read_csv(path, na_values="Blank(s)")
    
    # Transform the y columns
    df['SEER cause-specific death classification'] = df['SEER cause-specific death classification'] \
                                                    .apply(lambda x: 1 if x=="Dead (attributable to this cancer dx)" else 0)
    df = df.loc[df['Survival months'] != "Unknown"].copy(deep=True)
    df['Survival months'] = df['Survival months'].astype(int)
    
    # Split Grade column
    df[['Differentiate', 'Grade']] = df['Grade (thru 2017)'].str.split(';',expand=True).iloc[:,:2]
    
    # Rename features
    names = {'Age recode with <1 year olds': 'Age',
             'Race recode (White, Black, Other)': 'Race',
             'SEER historic stage A (1973-2015)': 'AStage',
             'Derived AJCC T, 6th ed (2004-2015)': 'TStage',
             'Derived AJCC N, 6th ed (2004-2015)': 'NStage',
             'Derived AJCC Stage Group, 6th ed (2004-2015)': '6thStage',
             'Regional nodes examined (1988+)': 'RegionalNodeExamined',
             'Regional nodes positive (1988+)': 'RegionalNodePositive',
             'ER Status Recode Breast Cancer (1990+)': 'EstrogenStatus',
             'PR Status Recode Breast Cancer (1990+)': 'ProgesteroneStatus',
             'CS tumor size (2004-2015)': 'TumorSize',
             'Reason no cancer-directed surgery': 'Surgery'}
    df = df.rename(columns=names)
    
    # Select relevant features and labels
    df = df[['Age', 'Race', 'Sex', 'AStage', 'TStage', 'NStage', '6thStage', 'RegionalNodeExamined', 'RegionalNodePositive',
             'EstrogenStatus', 'ProgesteroneStatus', 'TumorSize', 'Surgery', 'Survival months', 'SEER cause-specific death classification']]
        
    # Drop invalid data based on AStage and TumorSize
    df = df.loc[(df['AStage'] != 'Unstaged') & (df['TumorSize'] != 999)]
    
    # Rename column Race value for Other
    df['Race'] = df['Race'].replace('Other (American Indian/AK Native, Asian/Pacific Islander)', 'Other')
    df['Race'] = df['Race'].replace('Unknown', None)
    
    # Rename column Surgery value
    df['Surgery'] = df['Surgery'].replace('Surgery performed', 'Yes')
    df['Surgery'] = df['Surgery'].replace('Not recommended', 'No')
    df['Surgery'] = df['Surgery'].replace('Recommended but not performed, patient refused', 'No')
    df['Surgery'] = df['Surgery'].replace('Recommended but not performed, unknown reason', 'No')
    df['Surgery'] = df['Surgery'].replace('Not recommended, contraindicated due to other cond; autopsy only (1973-2002)', 'No')
    df['Surgery'] = df['Surgery'].replace('Not performed, patient died prior to recommended surgery', 'No')
    df['Surgery'] = df['Surgery'].replace('Recommended, unknown if performed', 'Unknown')
    df['Surgery'] = df['Surgery'].replace('Unknown; death certificate; or autopsy only (2003+)', 'Unknown')
    
    # Rename label column
    df = df.rename(columns={"SEER cause-specific death classification": "Status"})
    
    # Label encode the Age feature
    le = LabelEncoder()
    df['Age'] = le.fit_transform(df['Age'])
    
    # Rename EstrogenStatus/ProgesteroneStatus unknowns and fix nans
    df['EstrogenStatus'] = df['EstrogenStatus'].replace('Borderline/Unknown', 'Borderline')
    df['EstrogenStatus'] = df['EstrogenStatus'].replace('Recode not available', None)
    df['ProgesteroneStatus'] = df['ProgesteroneStatus'].replace('Borderline/Unknown', 'Borderline')
    df['ProgesteroneStatus'] = df['ProgesteroneStatus'].replace('Recode not available', None)
    
    # Save file
    path = Path.joinpath(pt.DATA_DIR, "seer.pkl")
    df.to_pickle(path)
