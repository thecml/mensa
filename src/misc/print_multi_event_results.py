import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.model_helper import map_model_name, map_model_type

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

    cols_to_scale = ["GlobalCI", "LocalCI", "AUC", "IBS"]
    df[cols_to_scale] = df[cols_to_scale] * 100
    
    dataset_names = ['mimic_me', 'rotterdam_me', 'proact_me', 'ebmt_me']
    
    # Sorted in desired order
    model_names = ["coxph", "coxnet", "weibullaft", "coxboost", "rsf", "mtlr", "deepsurv", "deephit", "dsm", "hierarch", "mensa"]
    disc_metrics = ["GlobalCI", "LocalCI", "AUC"]
    for model_name in model_names:
        row_parts = [map_model_name(model_name), map_model_type(model_name)]
        
        for dataset_name in dataset_names:
            for metric_name in disc_metrics:
                if metric_name in ["CI", "AUC", "IBS", "MAEM"]:
                    avg_event_df = (
                        df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False)
                        .mean(numeric_only=True)
                    )
                    results = avg_event_df.loc[
                        (avg_event_df['DatasetName'] == dataset_name) &
                        (avg_event_df['ModelName'] == model_name)
                    ]
                else:
                    avg_event_df = (
                        df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False)
                        .mean(numeric_only=True)
                    )
                    results = avg_event_df.loc[
                        (avg_event_df['DatasetName'] == dataset_name) &
                        (avg_event_df['ModelName'] == model_name)
                    ]
                
                results = results[metric_name]
                mean = f"{np.mean(results):.1f}"
                std = f"{np.std(results):.2f}"
                row_parts.append(f"{mean}\\text{{\\tiny{{$\\pm${std}}}}}")
        
        print(" & ".join(row_parts) + " \\\\")
    print()

    for model_name in model_names:
        row_parts = [map_model_name(model_name), map_model_type(model_name)]
        for dataset_name in dataset_names:
            
            avg_event_df = (
                df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False)
                  .mean(numeric_only=True)
            )
            
            results = avg_event_df.loc[
                (avg_event_df['DatasetName'] == dataset_name) &
                (avg_event_df['ModelName'] == model_name)
            ]["IBS"]

            mean = f"{np.mean(results):.1f}"
            std  = f"{np.std(results):.2f}"
            row_parts.append(f"{mean}\\text{{\\tiny{{$\\pm${std}}}}}")

            results = avg_event_df.loc[
                (avg_event_df['DatasetName'] == dataset_name) &
                (avg_event_df['ModelName'] == model_name)
            ]["MAEM"].copy()

            if dataset_name in ["mimic_me", "rotterdam_me", "ebmt_me"]:
                results = results / 100.0

            mean = f"{np.mean(results):.1f}"
            std  = f"{np.std(results):.2f}"
            row_parts.append(f"{mean}\\text{{\\tiny{{$\\pm${std}}}}}")

            row_parts.append(calculate_d_calib(df, model_name, dataset_name))

        print(" & ".join(row_parts) + " \\\\")
    print()