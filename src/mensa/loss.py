import torch

def conditional_weibull_loss(f: torch.Tensor,
                             s: torch.Tensor,
                             e: torch.Tensor,
                             n_risks: int,
                             weights: torch.Tensor = None) -> torch.Tensor:
    """
    Conditional Weibull negative log-likelihood loss for single-transition
    or competing-risks survival models.

    This loss corresponds to the likelihood over P possible states,
    where each sample transitions to at most one state p ∈ 𝒫.
    It is used for modeling mutually exclusive outcomes such as
    death causes or distinct terminal states.

    Parameters
    ----------
    f : torch.Tensor, shape (N, P)
        Log-density terms per sample and state, log f_p(t | x).
    s : torch.Tensor, shape (N, P)
        Log-survival terms per sample and state, log S_p(t | x).
    e : torch.Tensor, shape (N,)
        Integer event indicators in {0, 1, ..., P-1}, denoting
        which state transition was observed for each sample.
    n_risks : int
        Total number of modeled states (P).
    weights : torch.Tensor, optional, shape (P,)
        Optional per-state weighting factors w_p.

    Returns
    -------
    torch.Tensor
        Scalar negative log-likelihood averaged over all samples.

    Notes
    -----
    The loss implements the empirical form of the single-transition likelihood:

        L_ST = Σ_i Σ_p w_p [ 1(e_i = p) * log f_p(t_i | x_i)
                             + 1(e_i ≠ p) * log S_p(t_i | x_i) ]

    where:
        - f_p and S_p are the conditional Weibull density and survival
          functions for state p,
        - w_p are optional per-state weights.

    Conceptually, this models transitions into one of P states
    (including absorbing or terminal ones) under the competing-risks
    assumption that only one transition occurs per subject.
    """
    loss = 0.0
    for k in range(n_risks):
        observed = (e == k)
        uncensored_loss = torch.sum(observed*f[:, k].T)
        censored_loss = torch.sum(~observed*s[:, k].T)
        if weights is not None:
            loss += weights[k] * (uncensored_loss + censored_loss)
        else:
            loss += (uncensored_loss + censored_loss)
    loss = -loss / e.shape[0]
    return loss

def conditional_weibull_loss_multi(
    f: torch.Tensor,
    s: torch.Tensor,
    e: torch.Tensor,
    n_risks: int,
    weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Conditional Weibull negative log-likelihood loss for multi-state survival modeling.

    Implements the empirical form of the multi-event likelihood
    described in Eq. (11) of the paper, assuming conditional independence
    between state transitions given the covariates.

    Parameters
    ----------
    f : torch.Tensor, shape (N, P)
        Log-density terms for each sample and state p.
    s : torch.Tensor, shape (N, P)
        Log-survival terms for each sample and state p.
    e : torch.Tensor, shape (N, P)
        Binary event indicators for each state (1 if transition observed).
    n_risks : int
        Total number of modeled states (P).
    weights : torch.Tensor, optional, shape (P,)
        Optional per-state weights w_p.

    Returns
    -------
    torch.Tensor
        Scalar negative log-likelihood averaged over all samples.

    Notes
    -----
    The loss corresponds to the log-likelihood:

        L_ME = Σ_i Σ_p w_p [ δ_ip * log f_p(t_ip | x_i)
                             + (1 - δ_ip) * log S_p(t_ip | x_i) ]

    where δ_ip indicates whether state p was observed for sample i.

    Under the conditional independence assumption, transitions between
    states are independent given the covariates, and the total likelihood
    factorizes across states. The loss therefore aggregates contributions
    over all P states.
    """
    loss = 0.0
    for k in range(n_risks):
        observed = (e[:, k] == 1)
        uncensored_loss = torch.sum(observed * f[:, k].T)
        censored_loss = torch.sum((~observed) * s[:, k].T)
        if weights is not None:
            loss += weights[k] * (uncensored_loss + censored_loss)
        else:
            loss += (uncensored_loss + censored_loss)
    loss = -loss / e.shape[0]
    return loss