import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np
from utility.model_helper import map_model_name

N_DECIMALS = 2
ALPHA = 0.05
DATASET_NAME = "proact_me"

def calculate_d_calib(df, model_name, dataset_name):
    results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)]
    num_seeds = df['Seed'].nunique()
    event_ratios = []
    event_ids = sorted(df['EventId'].unique())
    for event_id in event_ids:
        num_calib = results.loc[results['EventId'] == event_id]['DCalib'].apply(lambda x: (x > ALPHA)).sum()
        event_ratio = f"{num_calib}/{num_seeds}"
        event_ratios.append(event_ratio)
    result_string = "(" + ', '.join(event_ratios) + ")"
    return result_string

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"shared_proact.csv")
    df = pd.read_csv(path)
    
    model_names = ["with_shared", "no_shared"]
    metric_names = ["CI", "IBS", "MAE", "GlobalCI", "LocalCI", "DCalib"]
    n_events = 4
    
    for model_name in model_names:
        text = ""
        results = df.loc[(df['DatasetName'] == DATASET_NAME) & (df['ModelName'] == model_name)] \
                  .groupby(['EventId'])[['CI', 'IBS', 'MAE', 'GlobalCI', 'LocalCI']].mean() # take average per eventid, not Dcal
        if results.empty:
            break
        if model_name == "with_shared":
            model_name_display = "Using " + r"$\Phi(X)$" + " &" 
        else:
            model_name_display = "Using " + r"$\Phi_{K}(X)$" + " &"
        text += f"{model_name_display} "
        for i, metric_name in enumerate(metric_names):
            if metric_name == "DCalib":
                d_calib = calculate_d_calib(df, model_name, DATASET_NAME)
                text += f"{d_calib}"
            else:
                metric_result = results[metric_name]
                mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                text += f"{mean}$\pm${std} & "
        text += " \\\\"
        print(text)
    print()
    