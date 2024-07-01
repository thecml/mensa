import torch


def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask,
                  dim=dim)) + max_val


def mtlr_nll(
        logits: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        C1: float,
        average: bool = False
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    return nll_total


def argmax_approx(
        a,
        beta
):
    if a.dim() <= 0:
        raise ValueError
    elif a.dim() == 1:
        weight = torch.arange(a.size(0)).to(a.device).float()
        nominator = (weight * torch.exp(beta * a)).sum()
        denominator = torch.exp(beta * a).T.sum()
        return nominator / denominator
    else:
        weight = torch.arange(a.size(1)).unsqueeze(dim=0).to(a.device).float()
        nominator = torch.matmul(weight, torch.exp(beta * a.T))
        denominator = torch.exp(beta * a.T).sum(dim=0)
        return nominator.squeeze() / denominator


def cox_nll(
        risk_pred: torch.Tensor,
        precision: torch.Tensor,
        log_var: torch.Tensor,
        true_times: torch.Tensor,
        true_indicator: torch.Tensor,
        model: torch.nn.Module,
        C1: float
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    risk_pred : torch.Tensor, shape (num_samples, )
        Risk prediction from Cox-based model. It means the relative hazard ratio: \beta * x.
    true_times : torch.Tensor, shape (num_samples, )
        Tensor with the censor/event time.
    true_indicator : torch.Tensor, shape (num_samples, )
        Tensor with the censor indicator.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    eps = 1e-20
    risk_pred = risk_pred.reshape(-1, 1)
    true_times = true_times.reshape(-1, 1)
    true_indicator = true_indicator.reshape(-1, 1)
    mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
    mask[(true_times.T - true_times) > 0] = 0
    max_risk = risk_pred.max()
    log_loss = torch.exp(risk_pred - max_risk) * mask
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
    # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
    # Solution: Consider increase the batch size. Afterall the nll should performed on the whole dataset.
    # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
    neg_log_loss = -torch.sum(precision * (risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator) + log_var

    # L2 regularization
    for k, v in model.named_parameters():
        if "weight" in k:
            neg_log_loss += C1/2 * torch.norm(v, p=2)

    return neg_log_loss

def cox_nll2(
        risk_pred: torch.Tensor,
        true_times: torch.Tensor,
        true_indicator: torch.Tensor,
        model: torch.nn.Module,
        C1: float
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    risk_pred : torch.Tensor, shape (num_samples, )
        Risk prediction from Cox-based model. It means the relative hazard ratio: \beta * x.
    true_times : torch.Tensor, shape (num_samples, )
        Tensor with the censor/event time.
    true_indicator : torch.Tensor, shape (num_samples, )
        Tensor with the censor indicator.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    eps = 1e-20
    risk_pred = risk_pred.reshape(-1, 1)
    true_times = true_times.reshape(-1, 1)
    true_indicator = true_indicator.reshape(-1, 1)
    mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
    mask[(true_times.T - true_times) > 0] = 0
    max_risk = risk_pred.max()
    log_loss = torch.exp(risk_pred - max_risk) * mask
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
    # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
    # Solution: Consider increase the batch size. Afterall the nll should performed on the whole dataset.
    # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
    return risk_pred, log_loss, true_indicator
    #neg_log_loss = -torch.sum((risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator)

    # L2 regularization
    for k, v in model.named_parameters():
        if "weight" in k:
            neg_log_loss += C1/2 * torch.norm(v, p=2)

    return neg_log_loss

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def single_loss(model, data, event_name='T1'):#estimates loss assuming every thing is observed/no censoring for checking 
    f = model.PDF(data[event_name], data['X'])
    return -torch.mean(LOG(f))

def double_loss(model1, model2, data, copula=None): #estimates the joint loss
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    e1 = (data['E'] == 0)*1.0
    e2 = (data['E'] == 1)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2)
    loss = -loss/data['E'].shape[0]
    return loss

def triple_loss(model1, model2, model3, data, copula=None):#estimates the joint loss
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    s3 = model3.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    f3 = model3.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2) + LOG(s3)
        p2 = LOG(f2) + LOG(s1) + LOG(s3)
        p3 = LOG(f3) + LOG(s1) + LOG(s2)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1), s3.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
        p3 = LOG(f3) + LOG(copula.conditional_cdf("w", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (data['E'] == 0)*1.0
    e2 = (data['E'] == 1)*1.0
    e3 = (data['E'] == 2)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss = -loss/data['E'].shape[0]
    return loss