import numpy as np
import pandas as pd
import math
import torch
import rpy2.robjects as robjects
import scipy.integrate as integrate
from sklearn.utils import shuffle
from dataclasses import InitVar, dataclass, field
from skmultilearn.model_selection import iterative_train_test_split
from typing import List, Tuple, Optional, Union
from sksurv.linear_model.coxph import BreslowEstimator
from utility.preprocessor import Preprocessor
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class LabTransform(LabTransDiscreteTime): # for DeepHit CR
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

def digitize_and_convert(data, time_bins, y_col_names=['time', 'event']):
    df = pd.DataFrame(data[0]).astype(np.float32)
    df[y_col_names[0]] = np.digitize(data[1][:,0], bins=time_bins).astype(int)
    df[y_col_names[1]] = convert_to_competing_risk(data[2]).astype(int)
    return df

def convert_to_competing_risk(data):
    return np.array([next((i+1 for i, val in enumerate(subarr)
                           if val == 1), 0) for subarr in data])

def calculate_event_times(t_train, e_train):
    unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
    if 0 not in unique_times:
        unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
    return unique_times.numpy() 

def compute_survival_curve(model, X_train, X_test, e_train, t_train, event_times):
    train_logits = model.predict(X_train).reshape(-1)
    test_logits = model.predict(X_test).reshape(-1)
    breslow = BreslowEstimator().fit(train_logits, e_train, t_train)
    surv_fn = breslow.get_survival_function(test_logits)
    breslow_surv_times = np.row_stack([fn(event_times) for fn in surv_fn])
    return breslow_surv_times

def make_stratified_split_multi(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = 0
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df[0].values  # Contains all columns.
    columns = df[0].columns
    labs = df[2]
    
    if labs.shape[1] > 1: 
        traj_labs = get_trajectory_labels(labs)

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=traj_labs, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

'''
Impute missing values and scale
'''
def preprocess_data(X_train, X_valid, X_test, cat_features,
                    num_features, as_array=False) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean', scaling_strategy="minmax")
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_valid = transformer.transform(X_valid)
    X_test = transformer.transform(X_test)
    if as_array:
        return (np.array(X_train), np.array(X_valid), np.array(X_test))
    return (X_train, X_valid, X_test)

'''
Reformat labels so that each label corresponds to a trajectory (e.g., event1 then event2, event1 only, event2 then event1)
'''
def get_trajectory_labels(labs):
    unique_labs = np.unique(labs, axis=0)
    new_labs = np.zeros((labs.shape[0],))
    
    for i in range(labs.shape[0]):
        for j in range(unique_labs.shape[0]):
            if np.all(unique_labs[j, :] == labs[i, :]):
                new_labs[i] = j
    
    return new_labs

def mtlr_survival_multi(
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
        assert logits.dim() == 4, "The logits should have dimension with with size (n_samples, n_data, n_bins, n_events)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1, 2)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        #torch.einsum('bij,bjk->bik', density_event, G_with_samples)
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 3, "The logits should have dimension with with size (n_data, n_bins, n_events)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)

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

def calculate_baseline_hazard(
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
    baseline_hazard = hazard.cpu()
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival

def split_time_event(y):
    y_t = np.array(y['time'])
    y_e = np.array(y['event'])
    return (y_t, y_e)

def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None
) -> torch.Tensor:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins

def make_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array

def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

def make_stratified_split_single(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

def make_stratification_label(df):
    t = df["Survival_time"]
    bins = np.linspace(start=t.min(), stop=t.max(), num=20)
    t = np.digitize(t, bins, right=True)
    e = df["Event"]
    stra_lab = np.stack([t, e], axis=1)
    return stra_lab

def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike
) -> (torch.Tensor, torch.Tensor):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y

def coverage(time_bins, upper, lower, true_times, true_indicator) -> float:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    time_bins = check_and_convert(time_bins)
    upper, lower = check_and_convert(upper, lower)
    true_times, true_indicator = check_and_convert(true_times, true_indicator)
    true_indicator = true_indicator.astype(bool)
    covered = 0
    upper_median_times = predict_median_survival_times(upper, time_bins, round_up=True)
    lower_median_times = predict_median_survival_times(lower, time_bins, round_up=False)
    covered += 2 * np.logical_and(upper_median_times[true_indicator] >= true_times[true_indicator],
                                  lower_median_times[true_indicator] <= true_times[true_indicator]).sum()
    covered += np.sum(upper_median_times[~true_indicator] >= true_times[~true_indicator])
    total = 2 * true_indicator.sum() + (~true_indicator).sum()
    return covered / total

def predict_median_survival_times(
        survival_curves: np.ndarray,
        times_coordinate: np.ndarray,
        round_up: bool = True
):
    median_probability_times = np.zeros(survival_curves.shape[0])
    max_time = times_coordinate[-1]
    slopes = (1 - survival_curves[:, -1]) / (0 - max_time)

    if round_up:
        # Find the first index in each row that are smaller or equal than 0.5
        times_indices = np.where(survival_curves <= 0.5, survival_curves, -np.inf).argmax(axis=1)
    else:
        # Find the last index in each row that are larger or equal than 0.5
        times_indices = np.where(survival_curves >= 0.5, survival_curves, np.inf).argmin(axis=1)

    need_extend = survival_curves[:, -1] > 0.5
    median_probability_times[~need_extend] = times_coordinate[times_indices][~need_extend]
    median_probability_times[need_extend] = (max_time + (0.5 - survival_curves[:, -1]) / slopes)[need_extend]

    return median_probability_times

def convert_to_structured (T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def make_event_times (t_train, e_train):
    unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
    if 0 not in unique_times:
        unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
    return unique_times.numpy()

def make_time_bins_hierarchical(event_times, num_bins):
    min_time = np.min(event_times[event_times != -1]) 
    max_time = np.max(event_times[event_times != -1]) 
    time_range = max_time - min_time
    bin_size = time_range / num_bins
    binned_event_time = np.floor((event_times - min_time) / bin_size)
    binned_event_time[binned_event_time == num_bins] = num_bins - 1
    return binned_event_time
    
def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None
) -> torch.Tensor:
    """
    Courtesy of https://ieeexplore.ieee.org/document/10158019
    
    Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins

def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

def check_and_convert(*args):
    """ Makes sure that the given inputs are numpy arrays, list,
        tuple, panda Series, pandas DataFrames, or torch Tensors.

        Also makes sure that the given inputs have the same shape.

        Then convert the inputs to numpy array.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    last_length = ()
    for i, arg in enumerate(args):

        if len(arg) == 0:
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)

        else:

            if isinstance(arg, np.ndarray):
                x = (arg.astype(np.double),)
            elif isinstance(arg, list):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, tuple):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, pd.Series):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, pd.DataFrame):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, torch.Tensor):
                x = (arg.cpu().numpy().astype(np.double),)
            else:
                error = """{arg} is not a valid data format. Only use 'list', 'tuple', 'np.ndarray', 'torch.Tensor', 
                        'pd.Series', 'pd.DataFrame'""".format(arg=type(arg))
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, """Shapes between {}-th input array and 
                    {}-th input array are not consistent""".format(i - 1, i)
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result
