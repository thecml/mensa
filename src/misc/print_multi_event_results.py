import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np
from utility.model_helper import map_model_name

N_DECIMALS = 2
ALPHA = 0.05

def calculate_d_calib(df, model_name, dataset_name):
    results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)]
    num_seeds = int(results['Seed'].nunique())
    event_ids = sorted(results['EventId'].unique())
    num_calib = 0
    for event_id in event_ids:
        num_calib += int(results.loc[results['EventId'] == event_id]['DCalib'].apply(lambda x: (x > ALPHA)).sum())
    event_ratio = f"{num_calib}/{num_seeds*len(event_ids)}"
    return event_ratio

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"multi_event.csv")
    df = pd.read_csv(path)

    cols_to_scale = ["CI", "IBS", "GlobalCI", "LocalCI"]
    df[cols_to_scale] = df[cols_to_scale] * 100
    
    cols_to_replace = ["CI", "GlobalCI", "LocalCI"]
    df[cols_to_replace] = df[cols_to_replace].fillna(0.5)
    
    dataset_names = ['mimic_me', 'rotterdam_me', 'proact_me', 'ebmt_me']
    model_names = ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa']
    metric_names = ["GlobalCI", "LocalCI", "IBS", "MAEM", "DCalib"] 
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            text = ""
            model_name_text = map_model_name(model_name)
            text += f"& {model_name_text} & "
            for i, metric_name in enumerate(metric_names):
                if metric_name == "DCalib":
                    d_calib = calculate_d_calib(df, model_name, dataset_name)
                    text += f"{d_calib}"
                else:
                    if metric_name in ["CI", "IBS", "MAEM"]:
                        avg_seed_df = (df.groupby(["ModelName", "DatasetName", "EventId"], as_index=False).mean(numeric_only=True))
                        results = avg_seed_df.loc[(avg_seed_df['DatasetName'] == dataset_name)
                                                & (avg_seed_df['ModelName'] == model_name)]
                    else:
                        avg_event_df = (df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False).mean(numeric_only=True))
                        results = avg_event_df.loc[(avg_event_df['DatasetName'] == dataset_name)
                                                   & (avg_event_df['ModelName'] == model_name)]
                    results = results[metric_name]
                    if dataset_name in ["mimic_me", "rotterdam_me", "ebmt_me"] and metric_name == "MAEM":
                        results /= 100
                    mean = f"%.{N_DECIMALS}f" % round(np.mean(results), N_DECIMALS)
                    std = f"%.{N_DECIMALS}f" % round(np.std(results), N_DECIMALS)
                    text += f"{mean}$\pm${std} & "
            text += " \\\\"
            print(text)
        print()
            