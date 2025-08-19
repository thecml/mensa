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
    path = Path.joinpath(cfg.RESULTS_DIR, f"competing_risks.csv")
    df = pd.read_csv(path)

    cols_to_scale = ["GlobalCI", "LocalCI", "AUC", "IBS"]
    df[cols_to_scale] = df[cols_to_scale] * 100
    df.loc[df["DatasetName"] == "rotterdam_cr", "MAEM"] /= 100
        
    dataset_names = ["seer_cr", "rotterdam_cr"]
    model_names = ["coxph", "coxnet", "weibullaft", "coxboost", "rsf", "mtlrcr", "deepsurv", "deephit", "dsm", "hierarch", "mensa"]
    metric_names = ["GlobalCI", "LocalCI", "AUC", "IBS", "MAEM", "DCalib"]
    
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
                    if metric_name in ["GlobalCI", "LocalCI", "AUC", "IBS", "MAEM"]:
                        avg_seed_df = (df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False).mean(numeric_only=True))
                        results = avg_seed_df.loc[(avg_seed_df['DatasetName'] == dataset_name)
                                                & (avg_seed_df['ModelName'] == model_name)]
                    else:
                        avg_event_df = (df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False).mean(numeric_only=True))
                        results = avg_event_df.loc[(avg_event_df['DatasetName'] == dataset_name)
                                                   & (avg_event_df['ModelName'] == model_name)]
                    results = results[metric_name]
                    mean = f"%.{1}f" % round(np.mean(results), 1)
                    std = f"%.{2}f" % round(np.std(results), 2)
                    text += f"{mean}$\pm${std} & "
            text += " \\\\"
            print(text)
        print()
        