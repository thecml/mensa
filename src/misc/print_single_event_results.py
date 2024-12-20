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
    path = Path.joinpath(cfg.RESULTS_DIR, f"single_event.csv")
    df = pd.read_csv(path)
    
    cols_to_scale = ["CI", "IBS"]
    df[cols_to_scale] = df[cols_to_scale] * 100
    df.loc[df["DatasetName"] == "mimic_se", "MAEM"] /= 100
    
    dataset_names = ["seer_se", "mimic_se"]
    model_names = ["coxph", "coxboost", "rsf", "deepsurv", "deephit", "mtlr", "dsm", "mensa"]
    metric_names = ["CI", "IBS", "MAEM", "DCalib"]
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            text = ""
            model_results = df.loc[(df['DatasetName'] == dataset_name)
                                   & (df['ModelName'] == model_name)]
            if model_results.empty:
                break
            model_name_text = map_model_name(model_name)
            text += f"& {model_name_text} & "
            for i, metric_name in enumerate(metric_names):
                results = model_results[metric_name]
                if metric_name == "DCalib":
                    d_calib_results = df.loc[(df['DatasetName'] == dataset_name) 
                                             & (df['ModelName'] == model_name)]['DCalib']
                    d_calib = sum(1 for value in d_calib_results if value > ALPHA)
                    text += f"{d_calib}/10"
                else:
                    mean = f"%.{N_DECIMALS}f" % round(np.mean(results), N_DECIMALS)
                    std = f"%.{N_DECIMALS}f" % round(np.std(results), N_DECIMALS)
                    text += f"{mean}$\pm${std} & "
            text += " \\\\"
            print(text)
        print()
        