import pandas as pd
import numpy as np
import config as cfg
from pathlib import Path
from utility.data import make_synthetic

if __name__ == "__main__":
    params = cfg.SYNTHETIC_SETTINGS
    raw_data, event_times, labels = make_synthetic(params['num_events'])
    print(0)


