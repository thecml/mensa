import pandas as pd
import numpy as np
import config as cfg
from pathlib import Path

if __name__ == "__main__":
    filenames = [f"mimic_static_feature_fold_{i}.csv.gz" for i in range(5)]
    
    df = pd.DataFrame()
    for filename in filenames:
        data = pd.read_csv(Path.joinpath(cfg.DATA_DIR, filename), compression='gzip', index_col=[0])
        df = pd.concat([df, data], axis=0)
    
    df.to_csv(Path.joinpath(cfg.DATA_DIR, 'mimic.csv.gz'), compression='gzip')
    