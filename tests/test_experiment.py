import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import poisson

from credibility_audit.experiment import build_predictions, mean_mcse, run_audit, score_predictions
from credibility_audit.model import GammaPrior, fit_prior
from credibility_audit.simulation import Portfolio, calibration_data, evaluation_data


def test_plugin_and_mixture_share_mean_not_variance():
    models = build_predictions(GammaPrior(0.8, 1.6), 0.5, GammaPrior(0.8, 1.6),
                               np.array([0, 2, 5]), np.array([0, 1, 4]), 3)
    plug, plug_rate = models["eb_mean_poisson"]
    full, full_rate = models["empirical_bayes_nb"]
    np.testing.assert_array_equal(plug_rate, full_rate)
    np.testing.assert_allclose(plug.mean(), full.mean())
    assert np.all(full.var() > plug.var())


def test_discrete_interval_endpoints_are_inclusive():
    distribution = poisson(np.full(4, 20.0))
    low, high = poisson(20).ppf([0.05, 0.95]).astype(int)
    assert low > 0
    data = Portfolio(np.full(4, 20), np.ones(4), np.zeros(4), np.array([low, high, low - 1, high + 1]))
    scores = score_predictions(distribution, np.full(4, 20), data, 0.9)
    np.testing.assert_array_equal(scores["coverage"], [1, 1, 0, 0])
    exact_mass = sum(poisson(20).pmf(np.arange(low, high + 1)))
    np.testing.assert_allclose(scores["model_interval_mass"], exact_mass, atol=1e-14)
    np.testing.assert_array_equal(scores["interval_width"], np.full(4, high - low))


def test_prediction_does_not_depend_on_future_stream():
    counts, exposure = calibration_data(177, 0, 1000)
    prior = fit_prior(counts, exposure).prior
    first = evaluation_data(177, 0, 0, 400, 3)
    second = evaluation_data(177, 0, 0, 400, 9)
    np.testing.assert_array_equal(first.history, second.history)
    np.testing.assert_array_equal(first.latent_rate, second.latent_rate)
    assert not np.array_equal(first.future, second.future)
    # Change future labels but hold the requested forecast horizon fixed.
    arguments = (prior, counts.sum() / exposure.sum(), GammaPrior(0.8, 1.6))
    predictions1 = build_predictions(*arguments, first.history, first.exposure, 3)
    predictions2 = build_predictions(*arguments, second.history, second.exposure, 3)
    for model in predictions1:
        np.testing.assert_array_equal(predictions1[model][0].mean(), predictions2[model][0].mean())


def test_repeated_audit_and_balanced_strata():
    first, details = run_audit(replications=3, calibration_size=1000, evaluation_size=400)
    second, repeated = run_audit(replications=3, calibration_size=1000, evaluation_size=400)
    assert first == second
    assert details == repeated
    assert len(details["fits"]) == 3
    assert len(details["replication_metrics"]) == 3 * 2 * 4 * 5
    assert len(first["results"]) == 2 * 4 * 5
    for row in first["results"]:
        assert row["model_interval_mass"]["mean"] >= 0.9 - 1e-12
        assert row["negative_log_score"]["mean"] >= 0
        assert 0 <= row["coverage"]["mean"] <= 1
    # Equal-size strata imply that overall means average stratum means exactly.
    for scenario in ("stable_population", "shifted_population"):
        for model in ("pooled_poisson", "eb_mean_poisson", "empirical_bayes_nb", "oracle_nb"):
            rows = [r for r in first["results"] if r["scenario"] == scenario and r["model"] == model]
            overall = next(r for r in rows if r["history_years"] == "all")
            strata = [r for r in rows if r["history_years"] != "all"]
            assert overall["negative_log_score"]["mean"] == pytest.approx(
                np.mean([r["negative_log_score"]["mean"] for r in strata]))


def test_mcse_uses_replications_and_paired_differences():
    assert mean_mcse([1, 2, 3]) == {"mean": 2.0, "mcse": float(1 / np.sqrt(3))}
    summary, details = run_audit(replications=3, calibration_size=1000, evaluation_size=400)
    selected = [r for r in details["replication_metrics"] if r["scenario"] == "stable_population"
                and r["history_years"] == "all"]
    diff = []
    for rep in range(3):
        by_model = {r["model"]: r["negative_log_score"] for r in selected if r["replication"] == rep}
        diff.append(by_model["eb_mean_poisson"] - by_model["empirical_bayes_nb"])
    stored = next(r for r in summary["paired_log_score_differences"]
                  if r["scenario"] == "stable_population" and r["history_years"] == "all"
                  and r["comparison"] == "eb_mean_poisson minus empirical_bayes_nb")
    assert stored["mean"] == mean_mcse(diff)["mean"]
    assert stored["mcse"] == mean_mcse(diff)["mcse"]


@pytest.mark.parametrize("options", [{"replications": 1}, {"replications": True},
    {"calibration_size": 19}, {"evaluation_size": 9}, {"seed": -1},
    {"horizon": 0}, {"interval_level": 1}, {"interval_level": np.nan}])
def test_invalid_experiment_options(options):
    with pytest.raises(ValueError):
        run_audit(**options)


def test_cli_creates_valid_artifacts(tmp_path):
    subprocess.run([sys.executable, "-m", "credibility_audit.experiment", "--replications", "2",
                    "--calibration-size", "1000", "--evaluation-size", "400", "--output", str(tmp_path)],
                    check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["configuration"]["replications"] == 2
    assert len(json.loads((tmp_path / "replications.json").read_text())["fits"]) == 2
    for name in ("held-out-prediction-audit.png", "cold-start-intervals.png"):
        data = (tmp_path / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 10000
