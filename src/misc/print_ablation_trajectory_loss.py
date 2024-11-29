import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np
from utility.model_helper import map_model_name

N_DECIMALS = 2
ALPHA = 0.05
DATASET_NAME = "ebmt_me" # rotterdam_me/ebmt_me

def calculate_d_calib(df, model_name, dataset_name):
    results = df.loc[(df['DatasetName'] == dataset_name) & (df['ModelName'] == model_name)]
    num_seeds = results['Seed'].nunique()
    event_ratios = []
    event_ids = sorted(results['EventId'].unique())
    for event_id in event_ids:
        num_calib = results.loc[results['EventId'] == event_id]['DCalib'].apply(lambda x: (x > ALPHA)).sum()
        event_ratio = f"{num_calib}/{num_seeds}"
        event_ratios.append(event_ratio)
    result_string = "(" + ', '.join(event_ratios) + ")"
    return result_string

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"trajectory_loss.csv")
    df = pd.read_csv(path)
    
    model_names = ["with_trajectory", "no_trajectory"]
    metric_names = ["CI", "IBS", "MAE", "GlobalCI", "LocalCI", "DCalib"]
    
    for model_name in model_names:
        text = ""
        results = df.loc[(df['DatasetName'] == DATASET_NAME) & (df['ModelName'] == model_name)] \
                  .groupby(['EventId'])[['CI', 'IBS', 'MAE', 'GlobalCI', 'LocalCI']].mean() # take average across seeds, not Dcal
        if results.empty:
            break
        if model_name == "with_trajectory":
            model_name_display = "& With " + r"$\mathcal{L}_{trajectory}$" + " &"
        else:
            model_name_display = "& Without " + r"$\mathcal{L}_{trajectory}$" + " &"
        text += f"{model_name_display} "
        for i, metric_name in enumerate(metric_names):
            if metric_name == "DCalib":
                d_calib = calculate_d_calib(df, model_name, DATASET_NAME)
                text += f"{d_calib}"
            else:
                metric_result = results[metric_name]
                if metric_name == "MAE":
                    mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result)/100, N_DECIMALS)
                    std = f"%.{N_DECIMALS}f" % round(np.std(metric_result)/100, N_DECIMALS)
                else:
                    mean = f"%.{N_DECIMALS}f" % round(np.mean(metric_result), N_DECIMALS)
                    std = f"%.{N_DECIMALS}f" % round(np.std(metric_result), N_DECIMALS)
                text += f"{mean}$\pm${std} & "
        text += " \\\\"
        print(text)
    print()
    