import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
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
    if metric_name in ["GlobalCI", "LocalCI", "AUC"]:
        color = "myred" if improvement < 0 else "mygreen"
    elif metric_name in ["IBS", "MAEM"]:
        color = "mygreen" if improvement < 0 else "myred"
    elif metric_name == "DCalib":
        color = "myred" if improvement < 0 else "mygreen"
    else:
        color = "black"
    improvement_text = f"({sign}{improvement})"
    improvement_text = "\\textcolor{" + f"{color}" + "}" + "{" + f"{improvement_text}" + "}"
    return improvement_text

if __name__ == "__main__":
    path = Path.joinpath(cfg.RESULTS_DIR, "independent_model.csv")
    df = pd.read_csv(path)

    # Scale selected metrics by 100
    cols_to_scale = ["GlobalCI", "LocalCI", "AUC", "IBS"]
    df[cols_to_scale] = df[cols_to_scale] * 100

    dataset_names = ["rotterdam_me", "proact_me"]
    model_names = ["not_independent", "independent"]
    metric_names = ["GlobalCI", "LocalCI", "AUC", "IBS", "MAEM", "DCalib"]

    for dataset_name in dataset_names:
        baseline_results = {}
        for model_name in model_names:
            model_name_display = "Jointly" if model_name == "not_independent" else "Indep."
            text = f"& {model_name_display} & "
            
            # Average over events per seed
            avg_per_seed_df = df.groupby(
                ["ModelName", "DatasetName", "Seed"], as_index=False
            ).mean(numeric_only=True)
            
            results_df = avg_per_seed_df.loc[
                (avg_per_seed_df["DatasetName"] == dataset_name) &
                (avg_per_seed_df["ModelName"] == model_name)
            ]
            
            for metric_name in metric_names:
                if metric_name == "DCalib":
                    dcalib_str = calculate_d_calib(df, model_name, dataset_name)
                    text += f"{dcalib_str} & "
                else:
                    if dataset_name in ["mimic_me", "rotterdam_me", "ebmt_me"] and metric_name == "MAEM":
                        metric_values = results_df[metric_name] / 100
                    else:
                        metric_values = results_df[metric_name]

                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values)

                    mean_str = f"{mean_val:.{N_DECIMALS}f}"
                    std_str = f"{std_val:.{N_DECIMALS}f}"

                    if model_name == "not_independent":
                        baseline_results[metric_name] = mean_val
                        text += f"{mean_str}$\\pm${std_str} & "
                    else:
                        baseline = baseline_results.get(metric_name)
                        if baseline is not None:
                            improvement_text = calculate_improvement(mean_val, baseline, metric_name)
                            text += f"{mean_str}$\\pm${std_str} {improvement_text} & "
                        else:
                            text += f"{mean_str}$\\pm${std_str} & "

            text = text.rstrip("& ") + " \\\\"
            print(text)
        print()
