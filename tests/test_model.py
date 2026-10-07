import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma, nbinom, poisson

from credibility_audit.model import (
    FitError, GammaPrior, credibility_weight, fit_prior, marginal_logpmf,
    marginal_objective, posterior_mean, posterior_parameters, predictive,
)
from credibility_audit.simulation import calibration_data


def test_posterior_and_predictive_hand_calculation():
    prior = GammaPrior(2, 3)
    a, b = posterior_parameters(prior, [4], [5])
    np.testing.assert_array_equal(a, [6])
    np.testing.assert_array_equal(b, [8])
    distribution = predictive(prior, [4], [5], 2)
    assert distribution.mean()[0] == pytest.approx(1.5)
    assert distribution.var()[0] == pytest.approx(1.875)


def test_credibility_weight_identity_and_monotonicity():
    prior = GammaPrior(0.8, 1.6)
    counts, exposure = np.array([0, 1, 5]), np.array([0.25, 1, 4])
    z = credibility_weight(prior, exposure)
    np.testing.assert_allclose(posterior_mean(prior, counts, exposure),
                               z * counts / exposure + (1 - z) * prior.mean)
    assert np.all(np.diff(z) > 0)
    assert credibility_weight(prior, 0) == 0


def test_zero_history_preserves_prior():
    prior = GammaPrior(0.8, 1.6)
    a, b = posterior_parameters(prior, [0], [0])
    assert a[0] == prior.shape and b[0] == prior.rate
    assert posterior_mean(prior, [0], [0])[0] == prior.mean
    assert marginal_logpmf([0], [0], prior)[0] == 0


def test_exposure_unit_conversion_invariance():
    prior = GammaPrior(0.8, 1.6)
    monthly_prior = GammaPrior(prior.shape, 12 * prior.rate)
    annual = predictive(prior, [0, 3], [0.25, 4], 3)
    monthly = predictive(monthly_prior, [0, 3], [3, 48], 36)
    np.testing.assert_allclose(annual.logpmf([2, 5]), monthly.logpmf([2, 5]), atol=1e-14)
    np.testing.assert_allclose(posterior_mean(monthly_prior, [0, 3], [3, 48]) * 12,
                               posterior_mean(prior, [0, 3], [0.25, 4]))


def test_sequential_posterior_equals_aggregated_history():
    prior = GammaPrior(1.5, 2.2)
    a, b = posterior_parameters(prior, [2], [0.5])
    sequential = posterior_parameters(GammaPrior(float(a[0]), float(b[0])), [3], [1.2])
    aggregated = posterior_parameters(prior, [5], [1.7])
    np.testing.assert_allclose(sequential, aggregated)


@pytest.mark.parametrize("future_count", [0, 1, 4, 10])
def test_predictive_matches_independent_mixture_quadrature(future_count):
    prior = GammaPrior(1.4, 2.1)
    history, exposure, horizon = 3, 0.5, 2.0
    shape, rate = prior.shape + history, prior.rate + exposure
    integrated, error = quad(lambda lam: poisson.pmf(future_count, horizon * lam)
                            * gamma.pdf(lam, a=shape, scale=1 / rate),
                            0, np.inf, epsabs=1e-11, epsrel=1e-10)
    assert error < 1e-9
    actual = predictive(prior, [history], [exposure], horizon).pmf(future_count)[0]
    assert actual == pytest.approx(integrated, rel=1e-9, abs=1e-11)


def test_marginal_matches_scipy_negative_binomial():
    prior = GammaPrior(0.8, 1.6)
    counts = np.array([0, 1, 4, 15])
    exposure = np.array([0, 0.25, 1, 4])
    expected = nbinom.logpmf(counts, n=prior.shape, p=prior.rate / (prior.rate + exposure))
    np.testing.assert_allclose(marginal_logpmf(counts, exposure, prior), expected, atol=1e-12)


@pytest.mark.parametrize("parameters", [[-0.2, 0.5], [1.2, 1.8], [-1.0, -0.4]])
def test_analytic_gradient_against_central_differences(parameters):
    parameters = np.asarray(parameters)
    counts, exposure = np.array([0, 1, 2, 5]), np.array([0.25, 1, 4, 2])
    _, analytic = marginal_objective(parameters, counts, exposure)
    finite_difference = []
    for axis in range(2):
        step = np.eye(2)[axis] * 1e-5
        finite_difference.append((marginal_objective(parameters + step, counts, exposure)[0]
                                  - marginal_objective(parameters - step, counts, exposure)[0]) / 2e-5)
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-8)


def test_hyperparameter_recovery_on_large_seeded_calibration_portfolio():
    counts, exposure = calibration_data(922, 0, 30000)
    fit = fit_prior(counts, exposure)
    assert fit.prior.shape == pytest.approx(0.8, rel=0.08)
    assert fit.prior.rate == pytest.approx(1.6, rel=0.08)
    assert fit.gradient_max < 1e-6
    assert fit.successful_starts >= 1
    best = fit.negative_mean_log_likelihood
    for perturbation in ([0.05, 0], [-0.05, 0], [0, 0.05], [0, -0.05]):
        nearby = np.log([fit.prior.shape, fit.prior.rate]) + perturbation
        assert marginal_objective(nearby, counts, exposure)[0] > best


def test_all_zero_fit_refused():
    with pytest.raises(FitError, match="all-zero"):
        fit_prior(np.zeros(30), np.ones(30))


def test_underdispersed_boundary_fit_refused():
    with pytest.raises(FitError, match="interior"):
        fit_prior(np.ones(30), np.ones(30))


@pytest.mark.parametrize("counts,exposure", [([], []), ([1, 2], [1]),
    ([-1], [1]), ([1.5], [1]), ([np.nan], [1]), ([0], [np.inf]), ([0], [-1]), ([1], [0])])
def test_invalid_history_rejected(counts, exposure):
    with pytest.raises(ValueError):
        posterior_parameters(GammaPrior(1, 2), counts, exposure)


@pytest.mark.parametrize("shape,rate", [(0, 1), (-1, 1), (1, 0), (np.inf, 1), (1, np.nan)])
def test_invalid_prior_rejected(shape, rate):
    with pytest.raises(ValueError):
        GammaPrior(shape, rate)


@pytest.mark.parametrize("horizon", [0, -1, np.nan, np.inf])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        predictive(GammaPrior(1, 2), [0], [1], horizon)


def test_invalid_calibration_design_rejected():
    with pytest.raises(ValueError):
        fit_prior([1], [1])
    with pytest.raises(ValueError):
        fit_prior(np.zeros(30), np.zeros(30))
