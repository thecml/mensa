import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import reformat_survival
from utility.loss import mtlr_nll
from torchmtlr.model import mtlr_neg_log_likelihood, mtlr_survival
from torchmtlr.utils import reset_parameters
from torch.optim import Adam

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer

class mtlr(nn.Module):
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
    
    time_bins = torch.cat([torch.tensor([0], device=survival_curves.device), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0)

def train_mtlr_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        dtype: torch.dtype,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
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
    x, y = reformat_survival(data_train, time_bins, dtype)
    x_val, y_val = reformat_survival(data_val, time_bins, dtype)
    x_val, y_val = x_val.to(device), y_val.to(device)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
    for i in pbar:
        nll_loss = 0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            loss = mtlr_nll(y_pred, yi, model, C1=config.c1, average=False)

            loss.backward()
            optimizer.step()

            nll_loss += (loss / train_size).item()
        logits_outputs = model.forward(x_val)
        eval_nll = mtlr_nll(logits_outputs, y_val, model, C1=0, average=True)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss:.4f}; "
                                f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                break
    end_time = datetime.now()
    training_time = end_time - start_time
    # model.eval()
    return model

def train_mtlr_cr(x_train, y_train, x_valid, y_valid, model, time_bins,
                  num_epochs=1000, lr=.01, weight_decay=0., C1=1., batch_size=32,
                  verbose=True, device="cpu", early_stop=True, patience=10):
    optimizer = make_optimizer(Adam, model, lr=lr, weight_decay=weight_decay)
    reset_parameters(model)
    model = model.to(device)
    model.train()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False)
    
    best_val_nll = np.inf
    best_ep = -1
    
    pbar =  trange(num_epochs, disable=not verbose)
    for i in pbar:
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            y_pred = model(xi)
            loss = mtlr_neg_log_likelihood(y_pred, yi, model, C1, average=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        logits_outputs = model.forward(x_valid.to(device))
        eval_nll = mtlr_neg_log_likelihood(logits_outputs, y_valid.to(device), model, C1, average=True)
        
        pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
        pbar.set_postfix_str(f"loss = {loss.item():.4f}")
        
        if early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > patience:
                break
        
    model.eval()
    return model