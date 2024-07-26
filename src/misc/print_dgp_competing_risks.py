import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np
from utility.model_helper import map_model_name

N_DECIMALS = 2

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"synthetic_cr.csv")
    df = pd.read_csv(path)
    df = df.round(N_DECIMALS).fillna(0)
    df = df.groupby(['ModelName', 'KTau', 'Seed']).mean().reset_index()
    
    k_taus = [0.25, 0.5, 0.75]
    model_names = ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa', 'dgp']
    metric_names = ["CI", "L1", "MAE"]
    
    for k_tau in k_taus:
        for model_name in model_names:
            text = ""
            results = df.loc[(df['KTau'] == k_tau) & (df['ModelName'] == model_name)]
            if results.empty:
                break
            model_name = map_model_name(model_name)
            text += f"& {model_name} & "
            for i, metric_name in enumerate(metric_names):
                metric_result = results[metric_name]
                mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                if i+1 == len(metric_names):
                    text += f"{mean}$\pm${std}"
                else:
                    text += f"{mean}$\pm${std} &"
            text += " \\\\"
            print(text)
        print()
        