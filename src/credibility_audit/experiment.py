"""Held-out predictive scoring, exposure strata, and population-shift stress."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from .model import fit_prior, posterior_mean, predictive
from .simulation import CALIBRATION_PRIOR, HISTORY_EXPOSURES, SCENARIOS, calibration_data, evaluation_data

METRICS = ("negative_log_score", "rate_mse", "coverage", "interval_width",
           "model_interval_mass", "coverage_minus_model_mass")
MODELS = ("pooled_poisson", "eb_mean_poisson", "empirical_bayes_nb", "oracle_nb")


def build_predictions(prior, pooled_rate, oracle_prior, history, exposure, horizon):
    """No future data accepted: same fitted hyperparameters across both scenarios."""
    eb_rate = posterior_mean(prior, history, exposure)
    oracle_rate = posterior_mean(oracle_prior, history, exposure)
    pooled = np.full(history.size, pooled_rate)
    return {
        "pooled_poisson": (poisson(mu=horizon * pooled), pooled),
        "eb_mean_poisson": (poisson(mu=horizon * eb_rate), eb_rate),
        "empirical_bayes_nb": (predictive(prior, history, exposure, horizon), eb_rate),
        "oracle_nb": (predictive(oracle_prior, history, exposure, horizon), oracle_rate),
    }


def score_predictions(distribution, predicted_rate, portfolio, level):
    tail = (1 - level) / 2
    low, high = distribution.ppf(tail), distribution.ppf(1 - tail)
    mass = distribution.cdf(high) - distribution.cdf(low - 1)
    coverage = ((portfolio.future >= low) & (portfolio.future <= high)).astype(float)
    values = {"negative_log_score": -distribution.logpmf(portfolio.future),
              "rate_mse": (predicted_rate - portfolio.latent_rate) ** 2,
              "coverage": coverage, "interval_width": high - low,
              "model_interval_mass": mass, "coverage_minus_model_mass": coverage - mass}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ArithmeticError("nonfinite predictive scores or intervals")
    return values


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "mcse": float(values.std(ddof=1) / np.sqrt(values.size))}


def aggregate(rows):
    groups = sorted({(r["scenario"], r["history_years"], r["model"]) for r in rows}, key=str)
    output = []
    for scenario, exposure, model in groups:
        group = [r for r in rows if (r["scenario"], r["history_years"], r["model"]) == (scenario, exposure, model)]
        output.append({"scenario": scenario, "history_years": exposure, "model": model,
                       **{metric: mean_mcse([r[metric] for r in group]) for metric in METRICS}})
    return output


def paired_comparisons(rows):
    output = []
    for scenario in SCENARIOS:
        for exposure in ("all", *HISTORY_EXPOSURES):
            chosen = [r for r in rows if r["scenario"] == scenario and r["history_years"] == exposure]
            baseline = {r["replication"]: r["negative_log_score"] for r in chosen if r["model"] == "empirical_bayes_nb"}
            for model in ("pooled_poisson", "eb_mean_poisson", "oracle_nb"):
                difference = [r["negative_log_score"] - baseline[r["replication"]]
                              for r in chosen if r["model"] == model]
                output.append({"scenario": scenario, "history_years": exposure,
                               "comparison": f"{model} minus empirical_bayes_nb",
                               **mean_mcse(difference)})
    return output


def run_audit(replications=50, calibration_size=2000, evaluation_size=1600,
              seed=20261007, horizon=3.0, interval_level=0.9):
    for value, name, minimum in ((replications, "replications", 2),
                                 (calibration_size, "calibration_size", 20),
                                 (evaluation_size, "evaluation_size", 4)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if evaluation_size % 4:
        raise ValueError("evaluation_size must be divisible by four for balanced strata")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(horizon) or horizon <= 0 or not 0 < interval_level < 1:
        raise ValueError("invalid horizon or interval level")
    rows, fits = [], []
    for replication in range(replications):
        counts, exposure = calibration_data(seed, replication, calibration_size)
        fitted = fit_prior(counts, exposure)
        pooled_rate = float(counts.sum() / exposure.sum())
        fits.append({"replication": replication, "pooled_rate": pooled_rate, **asdict(fitted)})
        for scenario_index, (scenario, oracle_prior) in enumerate(SCENARIOS.items()):
            portfolio = evaluation_data(seed, replication, scenario_index, evaluation_size, horizon)
            predictions = build_predictions(fitted.prior, pooled_rate, oracle_prior,
                                            portfolio.history, portfolio.exposure, horizon)
            for model, (distribution, predicted_rate) in predictions.items():
                scores = score_predictions(distribution, predicted_rate, portfolio, interval_level)
                for years in ("all", *HISTORY_EXPOSURES):
                    mask = np.ones(evaluation_size, dtype=bool) if years == "all" else portfolio.exposure == years
                    rows.append({"replication": replication, "scenario": scenario,
                                 "history_years": years, "model": model,
                                 **{name: float(values[mask].mean()) for name, values in scores.items()}})
    summary = {
        "schema_version": 1,
        "data": "Seeded synthetic portfolios; no real policyholder or financial data.",
        "configuration": {"replications": replications, "calibration_size": calibration_size,
                          "evaluation_size_per_scenario": evaluation_size, "seed": seed,
                          "horizon_years": horizon, "interval_level": interval_level,
                          "history_years": list(HISTORY_EXPOSURES),
                          "calibration_prior": asdict(CALIBRATION_PRIOR),
                          "evaluation_priors": {k: asdict(v) for k, v in SCENARIOS.items()}},
        "software": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        "fit_diagnostics": {"shape": mean_mcse([f["prior"]["shape"] for f in fits]),
                            "rate": mean_mcse([f["prior"]["rate"] for f in fits]),
                            "maximum_gradient": max(f["gradient_max"] for f in fits),
                            "minimum_successful_starts": min(f["successful_starts"] for f in fits)},
        "results": aggregate(rows), "paired_log_score_differences": paired_comparisons(rows),
    }
    return summary, {"fits": fits, "replication_metrics": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--calibration-size", type=int, default=2000)
    parser.add_argument("--evaluation-size", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    summary, details = run_audit(args.replications, args.calibration_size, args.evaluation_size, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", summary), ("replications", details)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from .plotting import plot_audit
    plot_audit(summary, args.output)
    print(json.dumps(summary["fit_diagnostics"], indent=2))
    print(f"Completed {args.replications} independent calibration fits and two held-out scenarios per fit.")


if __name__ == "__main__":
    main()
