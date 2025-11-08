import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple
import torch.nn.utils as nn_utils

import numpy as np
from tqdm import trange

from mensa.loss import conditional_weibull_loss, conditional_weibull_loss_multi, trajectory_loss
from mensa.mlp import MLP
from mensa.utility import safe_exp, safe_log

def add_event_free_column(
    T: torch.Tensor,
    E: torch.Tensor,
    n_events: int,
    horizon: float = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add an 'event-free' (transient) state column to time and event tensors.

    Extends event indicators and times to include a state representing
    subjects who remain event-free (no observed event among 1..K).
    This transient state becomes index 0, with other events shifted to 1..K.

    Parameters
    ----------
    T : torch.Tensor
        Event times, shape (N,) or (N, K).
    E : torch.Tensor
        Event indicators, shape (N, K).
    n_events : int
        Number of observed event types (K).
    horizon : float, optional
        Fixed time assigned to event-free samples for state 0.
        If None, uses the per-sample maximum time across events.

    Returns
    -------
    T_ext : torch.Tensor
        Extended time tensor of shape (N,) or (N, K+1).
    E_ext : torch.Tensor
        Extended event indicator tensor of shape (N, K+1),
        where column 0 marks event-free cases.

    Notes
    -----
    - If a sample has no event among 1..K, then E_ext[n, 0] = 1.
    - For multi-event data (T.ndim == 2), a new time column T[:,0]
      is created using `horizon` or max(T, dim=1).
    - This function is used when `use_transient=True` in MENSA.
    """
    N = E.size(0)
    device = E.device

    # Build E_ext with extra col 0
    if E.ndim != 2 or E.size(1) != n_events:
        raise ValueError("E must be [N, K] with K=n_events.")
    E_ext = torch.zeros((N, n_events + 1), dtype=E.dtype, device=device)
    E_ext[:, 1:] = E
    no_event = (E.sum(dim=1) == 0)
    E_ext[no_event, 0] = 1  # event-free label

    # Times
    if T.ndim == 1:
        # single-time case: nothing to change (your code uses the same t for all states)
        T_ext = T
    elif T.ndim == 2:
        if T.size(1) == n_events + 1:
            # already has the extra column
            T_ext = T
        elif T.size(1) == n_events:
            # create T[:,0]
            if horizon is not None:
                t0 = torch.full((N,), float(horizon), device=T.device, dtype=T.dtype)
            else:
                # reasonable default: the max observed time across event columns
                t0 = T.max(dim=1).values
            T_ext = torch.zeros((N, n_events + 1), dtype=T.dtype, device=T.device)
            T_ext[:, 0] = t0
            T_ext[:, 1:] = T
        else:
            raise ValueError("T has unexpected width. Expected K or K+1 columns.")
    else:
        raise ValueError("T must be 1D or 2D.")

    return T_ext, E_ext

class MENSA:
    def __init__(
        self,
        n_features: int,
        n_events: int,
        n_dists: int = 5,
        layers: list = [32, 32],
        dropout_rate: float = 0.5,
        trajectories: list = [],
        use_transient: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize a MENSA model wrapper.

        This class wraps the underlying MLP-based survival model and provides
        a convenient interface for fitting and evaluating MENSA.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_events : int
            Number of distinct events (K) to model.
        n_dists : int, optional (default=5)
            Number of Weibull mixture components per event.
        layers : list of int, optional (default=[32, 32])
            Sizes of hidden layers in the shared MLP backbone.
        dropout_rate : float, optional (default=0.5)
            Dropout rate applied between hidden layers.
        trajectories : list of tuple, optional
            List of known event trajectories (i → j) used to enforce temporal consistency.
        use_transient : bool, optional (default=True)
            Whether to include an additional transient state (K + 1 total states).
        device : str, optional (default='cpu')
            Computational device to use, e.g., "cpu" or "cuda".

        Notes
        -----
        - When `use_transient=True`, the model includes an additional latent
        transient state, increasing the total number of modeled states to K + 1.
        - The underlying model is constructed as an `MLP` instance that outputs
        the Weibull mixture parameters (shape, scale, gate) for each state.
        """
        self.n_features = n_features
        self.use_transient = use_transient
        self.n_events = n_events
        self.device = device

        # Determine total number of modelled states (P)
        # In MENSA, P = K + 1 when a transient state is included,
        # otherwise P = K, where K is the number of observed events.
        if self.use_transient:
            self.n_states = n_events + 1
        else:
            self.n_states = n_events

        self.trajectories = trajectories

        self.model = MLP(
            n_features, n_dists, layers, dropout_rate,
            temp=1000, n_states=self.n_states
        )

    def get_model(self):
        """
        Return the underlying neural survival model.

        Returns
        -------
        torch.nn.Module
            The core model containing all learnable parameters and layers.
        """
        return self.model
    
    def fit(
        self,
        train_dict: dict,
        valid_dict: dict,
        batch_size: int = 32,
        n_epochs: int = 100,
        patience: int = 10,
        optimizer: str = 'adam',
        weight_decay: float = 0,
        learning_rate: float = 0.001,
        betas: tuple = (0.9, 0.999),
        traj_lambda: float = 0.0,
        verbose: bool = False
    ):
        """
        Train the MENSA model with early stopping on a validation set.

        This method wraps the full training loop, including loss computation,
        optimization, gradient clipping, and early stopping based on validation loss.
        It supports both single-event and multi-event survival modeling, with an
        optional trajectory-consistency regularization term.

        Parameters
        ----------
        train_dict : dict
            Training data containing:
                'X' : torch.Tensor, shape (N, D)
                    Input features.
                'T' : torch.Tensor, shape (N,) or (N, K)
                    Observed event or censoring times.
                'E' : torch.Tensor, shape (N,) or (N, K)
                    Event indicators (1 if event occurred, 0 if censored).
        valid_dict : dict
            Validation data in the same format as `train_dict`.
        batch_size : int, optional (default=32)
            Mini-batch size for stochastic optimization.
        n_epochs : int, optional (default=100)
            Maximum number of training epochs.
        patience : int, optional (default=10)
            Number of epochs without improvement in validation loss
            before triggering early stopping.
        optimizer : {'adam', 'adamw'}, optional (default='adam')
            Optimization algorithm to use.
        weight_decay : float, optional (default=0)
            L2 regularization coefficient.
        learning_rate : float, optional (default=0.001)
            Initial learning rate for the optimizer.
        betas : tuple of float, optional (default=(0.9, 0.999))
            Beta coefficients for Adam/AdamW optimizers.
        traj_lambda : float, optional (default=0.0)
            Weight assigned to the trajectory consistency loss term.
            Set to > 0 to enforce temporal ordering between related events.
        verbose : bool, optional (default=False)
            Whether to display a progress bar with training and validation losses.

        Notes
        -----
        - MENSA models transitions between P states, where P = K (+1 if a transient
        event-free state is included). Each state p ∈ 𝒫 corresponds to either
        a terminal or intermediate event.
        - If `use_transient=True`, an additional event-free (transient) state is added
        to the data. This allows the model to capture transitions from “no event”
        to any absorbing or intermediate state.
        - When the input times `T` have multiple columns, the model trains under a
        **multi-state (multi-event)** likelihood, assuming conditional independence
        between transitions given the covariates. Otherwise, it defaults to the
        **single-transition (competing-risks)** formulation, where only one state
        can be reached per subject.
        - The total training loss combines the conditional Weibull likelihood
        with an optional trajectory-consistency penalty that enforces valid
        temporal ordering between states:

            L_total = (1 - λ) * L_Weibull + λ * L_traj

        - The training loop performs:
            1. Forward pass to compute Weibull mixture parameters for each state.
            2. Computation of log-density and log-survival terms.
            3. Weighted accumulation of per-state losses.
            4. Optional addition of trajectory loss for predefined state pairs (i → j).
            5. Gradient clipping (max norm = 1.0) for stability.
            6. Early stopping based on validation loss.

        Returns
        -------
        None
            The model is trained in-place. The best-performing weights (on the
            validation set) are automatically restored at the end of training.
        """

        optim_dict = [{'params': self.model.parameters(), 'lr': learning_rate}]

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)

        multi_event = True if train_dict['T'].ndim > 1 else False

        # Add transient state if toggled
        if self.use_transient and multi_event:
            train_times, train_events = add_event_free_column(train_dict['T'], train_dict['E'],
                                                            n_events=self.n_events, horizon=None)
            valid_times, valid_events = add_event_free_column(valid_dict['T'], valid_dict['E'],
                                                            n_events=self.n_events, horizon=None)
        else:
           train_events, valid_events = train_dict['E'], valid_dict['E']
           train_times, valid_times = train_dict['T'], valid_dict['T']

        train_loader = DataLoader(
            TensorDataset(train_dict['X'].to(self.device),
                        train_times.to(self.device),
                        train_events.to(self.device)),
            batch_size=batch_size, shuffle=True)

        valid_loader = DataLoader(
            TensorDataset(valid_dict['X'].to(self.device),
                        valid_times.to(self.device),
                        valid_events.to(self.device)),
            batch_size=batch_size, shuffle=False)
        
        # Compute event weights
        if multi_event:
            event_counts = torch.sum(train_events[:, 0:], dim=0).float()
        else:
            event_counts = torch.bincount(train_events)
        event_weights = 1.0 / (event_counts + 1e-8)
        event_weights = event_weights / event_weights.sum() * event_counts.shape[0]

        self.model.to(self.device)
        min_delta = 0.001
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        pbar = trange(n_epochs, disable=not verbose)

        for itr in pbar:
            self.model.train()
            total_train_loss = 0
            
            # Training step
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(self.device), ti.to(self.device), ei.to(self.device)
                optimizer.zero_grad()

                params = self.model.forward(xi)

                if multi_event:
                    f, s = self.compute_risks_multi(params, ti)
                    dens_loss = conditional_weibull_loss_multi(f, s, ei, self.model.n_states, event_weights)
                    traj_loss = 0.0
                    for (i, j) in self.trajectories:
                        traj_loss += trajectory_loss(i, j, ei, s)
                    loss = (1 - traj_lambda) * dens_loss + traj_lambda * traj_loss
                else:
                    f, s = self.compute_risks(params, ti)
                    dens_loss = conditional_weibull_loss(f, s, ei, self.model.n_states, event_weights)
                    loss = dens_loss

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                total_norm = nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                if not torch.isfinite(total_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_valid_loss = 0
            
            # Validation step
            with torch.no_grad():
                for xi, ti, ei in valid_loader:
                    xi, ti, ei = xi.to(self.device), ti.to(self.device), ei.to(self.device)
                    params = self.model.forward(xi)

                    if multi_event:
                        f, s = self.compute_risks_multi(params, ti)
                        dens_loss = conditional_weibull_loss_multi(f, s, ei, self.model.n_states, event_weights)
                        traj_loss = 0.0
                        for (i, j) in self.trajectories:
                            traj_loss += trajectory_loss(i, j, ei, s)
                        loss = (1 - traj_lambda) * dens_loss + traj_lambda * traj_loss
                    else:
                        f, s = self.compute_risks(params, ti)
                        dens_loss = conditional_weibull_loss(f, s, ei, self.model.n_states, event_weights)
                        loss = dens_loss

                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)

            pbar.set_description(f"[Epoch {itr+1:4}/{n_epochs}]")
            pbar.set_postfix_str(f"Training loss = {avg_train_loss:.4f}, "
                                f"Validation loss = {avg_valid_loss:.4f}")

            if avg_valid_loss < best_valid_loss - min_delta:
                best_valid_loss = avg_valid_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at iteration {itr}, best valid loss: {best_valid_loss}")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
        
    def compute_risks(
        self,
        params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ti: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-density (f) and log-survival (s) terms for each risk
        under a mixture of Weibull distributions. This is for single/competing-risks.
        For multi-event, see compute_risks_multi().

        Parameters
        ----------
        params : list of tuples [(k, b, gate), ...]
            Model parameters per risk/state.
            - k: log shape parameter tensor
            - b: log scale parameter tensor
            - gate: mixture logits per risk (before log-softmax)
        ti : torch.Tensor, shape (N,)
            Observed times (censored or event).

        Returns
        -------
        f_risks : torch.Tensor, shape (N, K)
            Log-density terms per risk.
        s_risks : torch.Tensor, shape (N, K)
            Log-survival terms per risk.
        """
        f_risks, s_risks = [], []
        eps = 1e-12
        ti = torch.clamp(ti.reshape(-1,1).expand(-1, self.model.n_dists), min=eps)

        for i in range(self.model.n_states):
            k = params[i][0]; b = params[i][1]
            gate = nn.LogSoftmax(dim=1)(params[i][2])

            ek = safe_exp(k); eb = safe_exp(b)
            s = -(torch.pow(eb*ti, ek)) # log S mixture terms before logsumexp
            f = k + b + ((ek - 1.0) * (b + safe_log(ti)))
            f = f + s

            s = torch.logsumexp(s + gate, dim=1)
            f = torch.logsumexp(f + gate, dim=1)

            f_risks.append(f); s_risks.append(s)

        return torch.stack(f_risks, 1), torch.stack(s_risks, 1)
            
    def compute_risks_multi(
        self,
        params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ti: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-density (f) and log-survival (s) terms for multiple events,
        each with its own time-to-event measurement.

        This extends `compute_risks()` to handle the multi-event case,
        where each event k has a distinct observed time t_k.  
        Each event may follow its own Weibull mixture distribution
        parameterized by (shape, scale, gate).

        Parameters
        ----------
        params : list of tuples [(k, b, gate), ...]
            Model parameters for each event/risk:
            - k : torch.Tensor, log(shape) parameter tensor per mixture component.
            - b : torch.Tensor, log(scale) parameter tensor per mixture component.
            - gate : torch.Tensor, logits for mixture weights per sample and component.
        ti : torch.Tensor, shape (N, E)
            Observed times for all events, where ti[n, e] is the observed (or censored)
            time for event e in sample n.

        Returns
        -------
        f : torch.Tensor, shape (N, E)
            Log-density terms per sample and event.
            Represents log f(t_e | x; θ_e) = log likelihood of observing event e at time t_e.
        s : torch.Tensor, shape (N, E)
            Log-survival terms per sample and event.
            Represents log S(t_e | x; θ_e) = log probability of *not* having event e by time t_e.
        """
        f_risks = []
        s_risks = []
        eps = 1e-12
        ti = torch.clamp(ti, min=eps)

        for i in range(self.model.n_states):
            # Expand per-event times to match mixture dimension
            t_i = ti[:, i].reshape(-1, 1).expand(-1, self.model.n_dists)

            k = params[i][0]
            b = params[i][1]
            gate_logits = params[i][2]

            gate = nn.LogSoftmax(dim=1)(gate_logits)

            # Safe exponentials
            ek = safe_exp(k)
            eb = safe_exp(b)

            # Log-survival per mixture component
            s_comp = -(torch.pow(eb * t_i, ek))

            # Log-density per mixture component (before mixture aggregation)
            f_comp = k + b + (ek - 1.0) * (b + safe_log(t_i))
            f_comp = f_comp + s_comp

            # Mixture aggregation in log-space
            s = torch.logsumexp(s_comp + gate, dim=1)
            f = torch.logsumexp(f_comp + gate, dim=1)

            f_risks.append(f)
            s_risks.append(s)

        f = torch.stack(f_risks, dim=1)
        s = torch.stack(s_risks, dim=1)
        return f, s

    def predict_survival(
        self,
        x_test: torch.Tensor,
        time_bins: torch.Tensor,
        risk: int = 0
    ) -> np.ndarray:
        """
        Predict the survival function S(t) = P(T > t | x) for a specified risk
        using the fitted Weibull mixture model.

        This implementation follows the formulation used in
        DeepSurvivalMachines (AutonLab), adapted for multi-risk settings.

        Parameters
        ----------
        x_test : torch.Tensor, shape (N, D)
            Input feature matrix for test samples.
        time_bins : torch.Tensor, shape (T,)
            Discretized grid of times (in ascending order) at which survival
            probabilities should be estimated.
        risk : int, optional (default=0)
            Index of the risk/event for which to predict the survival curve.

        Returns
        -------
        surv : np.ndarray, shape (N, T)
            Predicted survival probabilities for each sample and time.
            Each entry represents P(T > t | x) ∈ [0, 1].
            Decreases monotonically with time.

        Examples
        --------
        >>> surv = model.predict(x_test, torch.linspace(0, 10, 100), risk=1)
        >>> surv.shape
        (num_samples, 100)
        """
        t = list(time_bins.cpu().numpy())
        params = self.model.forward(x_test)
        
        shape, scale, logits = params[risk][0], params[risk][1], params[risk][2]
        k_ = shape
        b_ = scale

        squish = nn.LogSoftmax(dim=1)
        logits = squish(logits)
        
        t_horz = torch.tensor(time_bins).double().to(logits.device)
        t_horz = t_horz.repeat(shape.shape[0], 1)
        
        cdfs = []
        for j in range(len(time_bins)):

            t = t_horz[:, j]
            lcdfs = []

            for g in range(self.model.n_dists):

                k = k_[:, g]
                b = b_[:, g]
                s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
                lcdfs.append(s)

            lcdfs = torch.stack(lcdfs, dim=1)
            lcdfs = lcdfs+logits
            lcdfs = torch.logsumexp(lcdfs, dim=1)
            cdfs.append(lcdfs.detach().cpu().numpy())
        
        return np.exp(np.array(cdfs)).T
    
    def predict_cdf(
        self,
        x_test: torch.Tensor,
        time_bins: torch.Tensor,
        risk: int = 0
    ) -> np.ndarray:
        """
        Predict the cumulative distribution function F(t) = P(T ≤ t | x)
        for a specified risk using the fitted Weibull mixture model.

        Parameters
        ----------
        x_test : torch.Tensor, shape (N, D)
            Input feature matrix for test samples.
        time_bins : torch.Tensor, shape (T,)
            Discretized grid of times (in ascending order) at which cumulative
            probabilities should be estimated.
        risk : int, optional (default=0)
            Index of the risk/event for which to predict the CDF.

        Returns
        -------
        cdf : np.ndarray, shape (N, T)
            Predicted cumulative probabilities for each sample and time.
            Each entry represents P(T ≤ t | x) ∈ [0, 1].
            Increases monotonically with time.

        Examples
        --------
        >>> cdf = model.predict_cdf(x_test, torch.linspace(0, 10, 100), risk=1)
        >>> cdf.shape
        (num_samples, 100)
        """
        surv = self.predict_survival(x_test, time_bins, risk)
        cdf = 1.0 - surv
        return np.clip(cdf, 0.0, 1.0)