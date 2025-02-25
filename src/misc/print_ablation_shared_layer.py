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

def calculate_improvement(metric, baseline, metric_name):
    improvement = round(((float(metric) - float(baseline)) / float(baseline)) * 100, N_DECIMALS)
    sign = "+" if improvement > 0 else ""
    if metric_name in ["CI", "GlobalCI", "LocalCI"]:
        color = "dimRed" if improvement < 0 else "dimGreen"
    elif metric_name == "IBS":
        color = "dimGreen" if improvement < 0 else "dimRed"
    elif metric_name == "MAEM":
        color = "dimGreen" if improvement < 0 else "dimRed"
    elif metric_name == "DCalib":
        color = "dimRed" if improvement < 0 else "dimGreen"
    else:
        color = "black"
    improvement_text = f"({sign}{improvement})"
    improvement_text = "\\textcolor{" + f"{color}" + "}" + "{" + f"{improvement_text}" + "}"
    return improvement_text

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, f"shared_layer.csv")
    df = pd.read_csv(path)

    cols_to_scale = ["CI", "IBS", "GlobalCI", "LocalCI"]
    df[cols_to_scale] = df[cols_to_scale] * 100

    dataset_names = ["proact_me", "rotterdam_me"]
    model_names = ["with_shared", "no_shared"]
    metric_names = ["CI", "IBS", "MAEM", "GlobalCI", "LocalCI", "DCalib"]

    for dataset_name in dataset_names:
        baseline_results = {}
        for model_name in model_names:
            text = ""
            if model_name == "with_shared":
                model_name_display = "With"
            else:
                model_name_display = "Without"
            text += f"& {model_name_display} & "

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

                    if model_name == "with_shared":
                        # Store the baseline value for this metric
                        baseline_results[metric_name] = np.mean(results)
                        text += f"{mean}$\\pm${std} & "
                    else:
                        # Calculate improvement relative to the baseline
                        baseline = baseline_results.get(metric_name, None)
                        if baseline is not None:
                            improvement_text = calculate_improvement(np.mean(results), baseline, metric_name)
                            text += f"{mean}$\\pm${std} {improvement_text} & "
                        else:
                            text += f"{mean}$\\pm${std} & "

            text += " \\\\"
            print(text)
        print()
