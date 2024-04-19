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
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split_multi, cox_survival
from utility.data import MultiEventDataset
from models import CoxPH, MultiEventCoxPH
from models import BayesianBaseModel
from utility.data import dotdict
from utility.survival import mtlr_survival, mtlr_survival_multi

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def make_ensemble_mtlr_prediction_multi(
        model: BayesianBaseModel,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: dotdict
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()

    with torch.no_grad():
        # ensemble_output should have size: n_samples * dataset_size * n_bin
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        #logits_outputs = logits_outputs.reshape(config.n_samples_test,
        #                                        x.shape[0],
        #                                        (len(time_bins))+1,
        #                                        2)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = mtlr_survival_multi(logits_outputs, with_sample=True)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = time_bins.to(survival_outputs.device)
    return mean_survival_outputs, time_bins, survival_outputs

def make_ensemble_mtlr_prediction(
        model: BayesianBaseModel,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: dotdict
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()

    with torch.no_grad():
        # ensemble_output should have size: n_samples * dataset_size * n_bin
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = mtlr_survival(logits_outputs, with_sample=True)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = time_bins.to(survival_outputs.device)
    return mean_survival_outputs, time_bins, survival_outputs

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
