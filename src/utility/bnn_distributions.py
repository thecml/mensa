import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

class ParametrizedGaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        # torch.distributions doesn't go to cuda when we call model.to('cuda').
        # We have to manually put the sampling value to cuda as we did in function self.sample,
        # the other way is to store the distribution as buffers, as described in
        # https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
        self.normal = torch.distributions.Normal(0, 1)
        self.constant = (1 + math.log(2 * math.pi)) / 2

    @property
    def sigma(self):
        # It is the standard deviation
        # torch.log1p returns a new tensor with the natural logarithm of (1 + input).
        # \sigma = ln(e^\rho + 1)
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = self.normal.sample(sample_shape=(n_samples, *self.rho.size()))
        epsilon = epsilon.to(self.mu.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, x):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in
        https://math.stackexchange.com/questions/2029707/entropy-of-the-multivariate-gaussian
        """
        part1 = torch.sum(torch.log(self.sigma))
        part2 = self.mu.numel() * self.constant
        return part1 + part2


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        if pi > 1 or pi < 0:
            raise ValueError(f"pi must be in the range of (0, 1). Got {pi} instead")

        # pi is the (hyper)params for balancing the two Gaussian Dist
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, x):
        prob1 = torch.exp(self.gaussian1.log_prob(x))
        prob2 = torch.exp(self.gaussian2.log_prob(x))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class SpikeAndSlab(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        if pi > 1 or pi < 0:
            raise ValueError(f"pi must be in the range of (0, 1). Got {pi} instead")

        # pi is the (hyper)params for balancing the two Gaussian Dist
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, x):
        prob1 = torch.exp(self.gaussian1.log_prob(x))
        prob2 = torch.exp(self.gaussian2.log_prob(x))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class InverseGamma(object):
    """ Inverse Gamma distribution """
    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters of the distribution.

        Args:
            shape: torch tensor of floats, shape parameters of the distribution
            rate: torch tensor of floats, rate parameters of the distribution
        """
        super().__init__()
        self.shape = shape
        self.rate = rate

    def exp_inverse(self):
        """
        Calculates the expectation E[1/x], where x follows
        the inverse gamma distribution
        """
        return self.shape / self.rate

    def exp_log(self):
        """
        Calculates the expectation E[log(x)], where x follows
        the inverse gamma distribution
        """
        exp_log = torch.log(self.rate) - torch.digamma(self.shape)
        return exp_log

    def entropy(self):
        """
        Calculates the entropy of the inverse gamma distribution E[-ln(p(x))]
        """
        entropy = self.shape + torch.log(self.rate) + torch.lgamma(self.shape) - \
                  (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def logprob(self, target):
        """
        Computes the value of the predictive log likelihood at the target value
        log(pdf(Inv-Gamma)) = shape * log(rate) - log(Gamma(shape)) - (shape + 1) * log(x) - rate / x

        Args:
            target: Torch tensor of floats, point(s) to evaluate the logprob

        Returns:
            loglike: float, the log likelihood
        """
        part1 = self.shape * torch.log(self.rate)
        part2 = - torch.lgamma(self.shape)
        part3 = - (self.shape + 1) * torch.log(target)
        part4 = - self.rate / target

        return part1 + part2 + part3 + part4

    def update(self, shape, rate):
        """
        Updates shape and rate of the distribution. Used for the fixed point updates.

        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        """
        self.shape = shape
        self.rate = rate
