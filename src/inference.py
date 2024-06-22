import numpy as np
import pandas as pd
import math
import torch
from typing import List, Tuple, Optional, Union
import rpy2.robjects as robjects
import scipy.integrate as integrate
from sklearn.utils import shuffle
from dataclasses import InitVar, dataclass, field
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import reformat_survival
from utility.loss import mtlr_nll, cox_nll
from utility.survival import compute_unique_counts, make_monotonic, cox_survival
from utility.data import MultiEventDataset
from dgp import CoxPH, MultiEventCoxPH
from utility.data import dotdict
from utility.survival import mtlr_survival, mtlr_survival_multi

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def make_cox_prediction(
        model: CoxPH,
        x: torch.Tensor,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        if config.verbose:
            print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = cox_survival(model.baseline_survival, pred)
        survival_curves = survival_curves.squeeze()

    time_bins = model.time_bins
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def make_cox_prediction_multi(
        model: MultiEventCoxPH,
        x: torch.Tensor,
        config: argparse.Namespace,
        event_id: int
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        if config.verbose:
            print(f"Inference time: {inference_time.total_seconds()}")
        if event_id == 0:
            survival_curves = cox_survival(model.baseline_survival_net1, pred[0])
            time_bins = model.time_bins_net1
        elif event_id == 1:
            survival_curves = cox_survival(model.baseline_survival_net1, pred[1])
            time_bins = model.time_bins_net2
        else:
            assert False, 'Bad event id passed'
        survival_curves = survival_curves.squeeze()
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)
