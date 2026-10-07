"""Original implementation of standard conjugate credibility calculations."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, xlogy
from scipy.stats import nbinom


class FitError(RuntimeError):
    """A finite, interior marginal-likelihood solution was not validated."""


@dataclass(frozen=True)
class GammaPrior:
    shape: float
    rate: float

    def __post_init__(self):
        for value in (self.shape, self.rate):
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError("Gamma shape and rate must be positive finite scalars")

    @property
    def mean(self):
        return self.shape / self.rate


@dataclass(frozen=True)
class FitResult:
    prior: GammaPrior
    negative_mean_log_likelihood: float
    gradient_max: float
    iterations: int
    successful_starts: int


def validate_history(counts, exposure):
    counts, exposure = np.asarray(counts, dtype=float), np.asarray(exposure, dtype=float)
    if (counts.ndim != 1 or counts.size == 0 or counts.shape != exposure.shape
            or not np.isfinite(counts).all() or not np.isfinite(exposure).all()):
        raise ValueError("counts and exposure must be matching nonempty finite vectors")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be nonnegative integers")
    if np.any(exposure < 0) or np.any((exposure == 0) & (counts != 0)):
        raise ValueError("exposure must be nonnegative, and zero exposure requires zero counts")
    return counts, exposure


def _logpmf(counts, exposure, shape, rate):
    # log(rate/(rate+t)) is computed without subtracting nearby logarithms.
    log_ratio = np.log1p(exposure / rate)
    return (gammaln(shape + counts) - gammaln(shape) - gammaln(counts + 1)
            - shape * log_ratio + xlogy(counts, exposure / (rate + exposure)))


def marginal_logpmf(counts, exposure, prior):
    counts, exposure = validate_history(counts, exposure)
    return _logpmf(counts, exposure, prior.shape, prior.rate)


def marginal_objective(log_parameters, counts, exposure):
    """Negative mean log likelihood and its analytic log-parameter gradient.

    Inputs counts/exposure are validated by fit_prior; exported for gradient tests.
    Parameter order is log(shape), log(rate), never log(scale).
    """
    shape, rate = np.exp(log_parameters)
    value = -np.mean(_logpmf(counts, exposure, shape, rate))
    d_shape = digamma(shape + counts) - digamma(shape) - np.log1p(exposure / rate)
    d_rate = shape / rate - (shape + counts) / (rate + exposure)
    gradient = -np.array([shape * np.mean(d_shape), rate * np.mean(d_rate)])
    return float(value), gradient


def fit_prior(counts, exposure):
    """Fit hyperparameters using only an independent calibration portfolio.

    Three deterministic starts, bounded log parameters, analytic gradients.
    Refuse all-zero counts, boundary optima, or unvalidated optimizer results.
    No future counts or evaluation-contract histories enter this function.
    """
    counts, exposure = validate_history(counts, exposure)
    if counts.size < 20 or np.any(exposure <= 0):
        raise ValueError("fit requires >=20 contracts, each with positive exposure")
    if np.sum(counts) == 0:
        raise FitError("all-zero calibration counts do not identify a finite positive mean")
    empirical_rate = counts.sum() / exposure.sum()
    bounds = [(-6.0, 8.0), (-8.0, 10.0)]
    candidates = []
    for initial_shape in (0.3, 1.0, 5.0):
        initial = np.log([initial_shape, initial_shape / empirical_rate])
        initial = np.clip(initial, [v[0] + 0.01 for v in bounds], [v[1] - 0.01 for v in bounds])
        result = minimize(marginal_objective, initial, args=(counts, exposure), jac=True,
                          method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 500})
        value, gradient = marginal_objective(result.x, counts, exposure)
        interior = all(low + 1e-4 < x < high - 1e-4 for x, (low, high) in zip(result.x, bounds))
        if result.success and interior and np.isfinite(value) and np.max(np.abs(gradient)) < 1e-6:
            candidates.append((value, result, gradient))
    if not candidates:
        raise FitError("no validated interior optimum; the Poisson-limit/boundary case is unsupported")
    value, result, gradient = min(candidates, key=lambda item: item[0])
    prior = GammaPrior(*map(float, np.exp(result.x)))
    return FitResult(prior, value, float(np.max(np.abs(gradient))), int(result.nit), len(candidates))


def posterior_parameters(prior, counts, exposure):
    counts, exposure = validate_history(counts, exposure)
    return prior.shape + counts, prior.rate + exposure


def posterior_mean(prior, counts, exposure):
    shape, rate = posterior_parameters(prior, counts, exposure)
    return shape / rate


def credibility_weight(prior, exposure):
    exposure = np.asarray(exposure, dtype=float)
    if not np.isfinite(exposure).all() or np.any(exposure < 0):
        raise ValueError("exposure must be finite and nonnegative")
    return exposure / (prior.rate + exposure)


def predictive(prior, counts, exposure, horizon=3.0):
    """Distribution of future counts, integrating the contract's latent rate.

    With posterior Gamma(A,B), SciPy NB has n=A and p=B/(B+horizon).
    Empirical-Bayes hyperparameter uncertainty is NOT integrated out.
    """
    if not np.isscalar(horizon) or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be a positive finite scalar")
    shape, rate = posterior_parameters(prior, counts, exposure)
    return nbinom(n=shape, p=rate / (rate + horizon))
