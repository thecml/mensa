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
from utility.survival import compute_unique_counts, make_monotonic, make_stratified_split

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def cox_survival(
        baseline_survival: torch.Tensor,
        linear_predictor: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the individual survival distributions based on the baseline survival curves and the liner prediction values.
    :param baseline_survival: (n_time_bins, )
    :param linear_predictor: (n_samples, n_data)
    :return:
    The invidual survival distributions. shape = (n_samples, n_time_bins)
    """
    n_sample = linear_predictor.shape[0]
    n_data = linear_predictor.shape[1]
    risk_score = torch.exp(linear_predictor)
    survival_curves = torch.empty((n_sample, n_data, baseline_survival.shape[0]), dtype=torch.float).to(linear_predictor.device)
    for i in range(n_sample):
        for j in range(n_data):
            survival_curves[i, j, :] = torch.pow(baseline_survival, risk_score[i, j])
    return survival_curves

def baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival

class CoxPH(nn.Module):
    """Cox proportional hazard model for individualised survival prediction."""

    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = nn.Linear(self.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.l1(x)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()

class MultiEventCoxPH(nn.Module):
    """Cox proportional hazard model for individualised survival prediction."""

    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        
        self.output_dim_0 = 1
        self.output_dim_1 = 1
        self.hidden_dim = 100
        
        self.time_bins_dim_0 = None
        self.time_bins_dim_1 = None
        
        self.cum_baseline_hazard_dim_0 = None
        self.cum_baseline_hazard_dim_1 = None
        
        self.baseline_survival_dim_0 = None
        self.baseline_survival_dim_1 = None
        
        self.hidden = nn.Linear(self.in_features, self.hidden_dim)
        
        self.final_0 = nn.Linear(self.hidden_dim, self.output_dim_0)
        self.final_1 = nn.Linear(self.hidden_dim, self.output_dim_1)

    def forward(self, x: torch.Tensor, event_id:int) -> torch.Tensor:
        x = self.hidden(x)
        if event_id == 0:
            x = self.final_0(x)
        elif event_id == 1:
            x = self.final_1(x)
        else:
            assert False, 'Bad event id passed'
        return x

    def calculate_baseline_survival(self, x, t, e, event_id):
        outputs = self.forward(x, event_id=event_id)
        if event_id == 0:
            self.time_bins_dim_0, self.cum_baseline_hazard_dim_0, self.baseline_survival_dim_0 = baseline_hazard(outputs, t, e)
        elif event_id == 1:
            self.time_bins_dim_1, self.cum_baseline_hazard_dim_1, self.baseline_survival_dim_1 = baseline_hazard(outputs, t, e)
        else:
            assert False, 'Bad event id passed'
            
    def reset_parameters(self):
        self.final_0.reset_parameters()
        self.final_1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()


class mtlr(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")

    def get_name(self):
        return self._get_name()

def mtlr_survival(
        logits: torch.Tensor,
        with_sample: bool = True
) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module)
        with size (n_samples, n_data, n_bins) or (n_data, n_bins).

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)

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
        pred = model.forward(x, event_id=event_id)
        end_time = datetime.now()
        inference_time = end_time - start_time
        if config.verbose:
            print(f"Inference time: {inference_time.total_seconds()}")
        if event_id == 0:
            survival_curves = cox_survival(model.baseline_survival_dim_0, pred)
            time_bins = model.time_bins_dim_0
        elif event_id == 1:
            survival_curves = cox_survival(model.baseline_survival_dim_1, pred)
            time_bins = model.time_bins_dim_1
        else:
            assert False, 'Bad event id passed'
        survival_curves = survival_curves.squeeze()

    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def make_mtlr_prediction(
        model: mtlr,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        survival_curves = mtlr_survival(pred, with_sample=False)

    time_bins = torch.cat([torch.tensor([0]), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def train_multi_model(
        model: nn.Module,
        data_train_speech: pd.DataFrame,
        data_train_walking: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    data_train_speech, _, data_val_speech = make_stratified_split(data_train_speech, stratify_colname='both',
                                                                  frac_train=0.9, frac_test=0.1,
                                                                  random_state=random_state)
    data_train_walking, _, data_val_walking = make_stratified_split(data_train_walking, stratify_colname='both',
                                                                    frac_train=0.9, frac_test=0.1,
                                                                    random_state=random_state)

    train_size = data_train_speech.shape[0]
    val_size = data_val_speech.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x_train_s, t_train_s, e_train_s = (torch.tensor(data_train_speech.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                 torch.tensor(data_train_speech["time"].values, dtype=torch.float),
                                 torch.tensor(data_train_speech["event"].values, dtype=torch.float))
    x_val_s, t_val_s, e_val_s = (torch.tensor(data_val_speech.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                           torch.tensor(data_val_speech["time"].values, dtype=torch.float).to(device),
                           torch.tensor(data_val_speech["event"].values, dtype=torch.float).to(device))
    x_train_w, t_train_w, e_train_w = (torch.tensor(data_train_walking.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                 torch.tensor(data_train_walking["time"].values, dtype=torch.float),
                                 torch.tensor(data_train_walking["event"].values, dtype=torch.float))
    x_val_w, t_val_w, e_val_w = (torch.tensor(data_val_walking.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                           torch.tensor(data_val_walking["time"].values, dtype=torch.float).to(device),
                           torch.tensor(data_val_walking["event"].values, dtype=torch.float).to(device))

    train_loader_speech = DataLoader(TensorDataset(x_train_s, t_train_s, e_train_s), batch_size=config.batch_size, shuffle=True)
    train_loader_walking = DataLoader(TensorDataset(x_train_w, t_train_w, e_train_w), batch_size=config.batch_size, shuffle=True)
    
    zipped_dls = zip(train_loader_speech, train_loader_walking)

    for i in pbar:
        dataloader_iterator = iter(train_loader_speech)
        for i, (xi_w, ti_w, ei_w) in enumerate(train_loader_walking):
            try:
                (xi_s, ti_s, ei_s) = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader_speech)
                (xi_s, ti_s, ei_s) = next(dataloader_iterator)
            
            xi_s, ti_s, ei_s = xi_s.to(device), ti_s.to(device), ei_s.to(device)
            xi_w, ti_w, ei_w = xi_w.to(device), ti_w.to(device), ei_w.to(device)
            
            optimizer.zero_grad()
            y_pred_s = model.forward(xi_s, event_id=0)
            y_pred_w = model.forward(xi_w, event_id=1)
            
            nll_loss_speech = cox_nll(y_pred_s, ti_s, ei_s, model, C1=config.c1)
            nll_loss_walking = cox_nll(y_pred_w, ti_w, ei_w, model, C1=config.c1)
            
            nll_loss = nll_loss_speech + nll_loss_walking
            
            nll_loss.backward()
            optimizer.step()
            # here should have only one iteration
        
        logits_outputs_s = model.forward(x_val_s, event_id=0)
        eval_nll_s = cox_nll(logits_outputs_s, t_val_s, e_val_s, model, C1=0)
        
        logits_outputs_w = model.forward(x_val_w, event_id=1)
        eval_nll_w = cox_nll(logits_outputs_w, t_val_w, e_val_w, model, C1=0)
        
        total_eval_nll = eval_nll_s + eval_nll_w
        
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss.item():.4f}; "
                             f"Validation nll = {total_eval_nll.item():.4f};")
        
        if config.early_stop:
            if best_val_nll > total_eval_nll:
                best_val_nll = total_eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break
        
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    model.calculate_baseline_survival(x_train_s.to(device), t_train_s.to(device), e_train_s.to(device), event_id=0)
    model.calculate_baseline_survival(x_train_w.to(device), t_train_w.to(device), e_train_w.to(device), event_id=1)
        
    return model

def train_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    data_train, _, data_val = make_stratified_split(data_train, stratify_colname='both',
                                                    frac_train=0.9, frac_test=0.1,
                                                    random_state=random_state)

    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                 torch.tensor(data_train["time"].values, dtype=torch.float),
                                 torch.tensor(data_train["event"].values, dtype=torch.float))
    x_val, t_val, e_val = (torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                           torch.tensor(data_val["time"].values, dtype=torch.float).to(device),
                           torch.tensor(data_val["event"].values, dtype=torch.float).to(device))

    train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
    model.config.batch_size = train_size

    for i in pbar:
        nll_loss = 0
        for xi, ti, ei in train_loader:
            xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            nll_loss = cox_nll(y_pred, ti, ei, model, C1=config.c1)

            nll_loss.backward()
            optimizer.step()
            # here should have only one iteration
        logits_outputs = model.forward(x_val)
        eval_nll = cox_nll(logits_outputs, t_val, e_val, model, C1=0)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss.item():.4f}; "
                                f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    if isinstance(model, CoxPH):
        model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model