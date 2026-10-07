"""Synthetic portfolios with independent calibration, history and future streams."""

from dataclasses import dataclass

import numpy as np

from .model import GammaPrior

HISTORY_EXPOSURES = (0.0, 0.25, 1.0, 4.0)
CALIBRATION_PRIOR = GammaPrior(0.8, 1.6)
SCENARIOS = {"stable_population": CALIBRATION_PRIOR,
             "shifted_population": GammaPrior(0.8, 0.8)}


@dataclass(frozen=True)
class Portfolio:
    latent_rate: np.ndarray
    exposure: np.ndarray
    history: np.ndarray
    future: np.ndarray


def calibration_data(seed, replication, size):
    exposure_rng, latent_rng, history_rng = [np.random.default_rng(s) for s in
        np.random.SeedSequence([seed, replication, 0]).spawn(3)]
    exposure = exposure_rng.choice([0.25, 1.0, 4.0], size=size)
    latent = latent_rng.gamma(CALIBRATION_PRIOR.shape, 1 / CALIBRATION_PRIOR.rate, size=size)
    return history_rng.poisson(latent * exposure), exposure


def evaluation_data(seed, replication, scenario_index, size, horizon):
    prior = list(SCENARIOS.values())[scenario_index]
    latent_rng, history_rng, future_rng = [np.random.default_rng(s) for s in
        np.random.SeedSequence([seed, replication, scenario_index + 1]).spawn(3)]
    exposure = np.tile(HISTORY_EXPOSURES, size // len(HISTORY_EXPOSURES))
    latent = latent_rng.gamma(prior.shape, 1 / prior.rate, size=size)
    history = history_rng.poisson(latent * exposure)
    future = future_rng.poisson(latent * horizon)
    return Portfolio(latent, exposure, history, future)
