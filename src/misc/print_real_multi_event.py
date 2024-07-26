import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np
from utility.model_helper import map_model_name

N_DECIMALS = 2
ALPHA = 0.05

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"real_cr.csv")
    df = pd.read_csv(path)
    df = df.round(N_DECIMALS).fillna(0)
    df = df.groupby(['ModelName', 'DatasetName', 'Seed']).mean().reset_index()
    
    dataset_names = ["mimic", "als"]
    model_names = ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa']
    metric_names = ["CI", "GlobalCI", "LocalCI", "MAEH", "MAEM", "MAEPO", "IBS", "DCalib"]
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            text = ""
            results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)]
            if results.empty:
                break
            model_name = map_model_name(model_name)
            text += f"& {model_name} & "
            for i, metric_name in enumerate(metric_names):
                metric_result = results[metric_name]
                if metric_name == "DCalib":
                    sum_d_cal = sum(1 for value in metric_result if value > ALPHA)
                    text += f"{sum_d_cal}/5"
                else:
                    mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                    std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                    text += f"{mean}$\pm${std} & "
            text += " \\\\"
            print(text)
        print()
        