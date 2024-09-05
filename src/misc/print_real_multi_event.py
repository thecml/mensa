import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np
from utility.model_helper import map_model_name

N_DECIMALS = 2
ALPHA = 0.05

def calculate_d_calib(df, model_name, dataset_name, event_id):
    results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)]
    num_seeds = df['Seed'].nunique()
    event_ratios = []
    num_calib = results.loc[results['EvenId'] == event_id]['DCalib'].apply(lambda x: (x > ALPHA)).sum()
    event_ratio = f"{num_calib}/{num_seeds}"
    event_ratios.append(event_ratio)
    result_string = "(" + ', '.join(event_ratios) + ")"
    return result_string

def map_event_id(event_id):
    if event_id == 1:
        return "Speech"
    elif event_id == 2:
        return "Swallowing"
    elif event_id == 3:
        return "Handwriting"
    elif event_id == 4:
        return "Walking"
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"real_me.csv")
    df = pd.read_csv(path)
    df = df.round(N_DECIMALS).fillna(0)
    
    dataset_names = ["als_me"]
    model_names = ["deepsurv", 'hierarch', 'mensa']
    metric_names = ["CI", "IBS", "MAEM", "GlobalCI", "LocalCI", "DCalib"]
    n_events = 4
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            for event_id in range(n_events):
                text = ""
                results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)
                                 & (df['EvenId'] == event_id+1)]
                if results.empty:
                    break
                model_name_display = map_model_name(model_name)
                text += f"{model_name_display} & {map_event_id(event_id+1)} & "
                for i, metric_name in enumerate(metric_names):
                    metric_result = results[metric_name]
                    if metric_name == "DCalib":
                        d_calib = calculate_d_calib(df, model_name, dataset_name, event_id)
                        text += f"{d_calib}"
                    else:
                        mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                        std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                        text += f"{mean}$\pm${std} & "
                text += " \\\\"
                print(text)
            print()
            