"""Exposure-aware Gamma-Poisson credibility, using shape/rate parameters."""

from .model import GammaPrior, FitError, fit_prior, posterior_parameters, predictive

__all__ = ["GammaPrior", "FitError", "fit_prior", "posterior_parameters", "predictive"]
