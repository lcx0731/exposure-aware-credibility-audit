"""Static figures; error bars use independent replication-level Monte Carlo SEs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulation import HISTORY_EXPOSURES, SCENARIOS

STYLES = {
    "pooled_poisson": ("Pooled Poisson", "#747b86", "o"),
    "eb_mean_poisson": ("EB mean + Poisson", "#cf6738", "s"),
    "empirical_bayes_nb": ("Empirical Bayes NB", "#007e87", "D"),
    "oracle_nb": ("Oracle NB (benchmark)", "#6b4b9a", "^"),
}


def plot_audit(summary, output: Path):
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for col, scenario in enumerate(SCENARIOS):
            for model, (label, color, marker) in STYLES.items():
                rows = [next(r for r in summary["results"] if r["scenario"] == scenario
                             and r["model"] == model and r["history_years"] == years)
                        for years in HISTORY_EXPOSURES]
                for row, metric in enumerate(("negative_log_score", "coverage_minus_model_mass")):
                    scale = 100 if row == 1 else 1
                    axes[row, col].errorbar(np.arange(4), [scale * r[metric]["mean"] for r in rows],
                        yerr=[1.96 * scale * r[metric]["mcse"] for r in rows], color=color,
                        marker=marker, label=label, linewidth=1.6, capsize=3, markersize=5)
            axes[0, col].set_title("Stable population" if col == 0 else "Shifted population: twice the prior mean")
            axes[1, col].axhline(0, color="#343a40", linestyle=":", linewidth=1)
            for ax in axes[:, col]:
                ax.set_xticks(np.arange(4), ["0 (new)", "0.25", "1", "4"])
                ax.set_xlabel("Available history exposure (years)")
                ax.grid(alpha=0.15)
        axes[0, 0].set_ylabel("Mean negative log predictive score\n(lower is better)")
        axes[1, 0].set_ylabel("Observed coverage − model interval mass\n(percentage points; ideal = 0)")
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle("Synthetic Gamma–Poisson portfolios • 3-year future counts\nBars: ±1.96 Monte Carlo SE across independent replications", fontsize=12)
        fig.savefig(output / "held-out-prediction-audit.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")
        for col, scenario in enumerate(SCENARIOS):
            rows = [next(r for r in summary["results"] if r["scenario"] == scenario
                         and r["model"] == model and r["history_years"] == 0.0) for model in STYLES]
            positions = np.arange(len(rows))
            axes[col].bar(positions, [100 * r["coverage"]["mean"] for r in rows],
                          color=[style[1] for style in STYLES.values()], alpha=0.85)
            axes[col].errorbar(positions, [100 * r["coverage"]["mean"] for r in rows],
                              yerr=[196 * r["coverage"]["mcse"] for r in rows],
                              fmt="none", color="black", capsize=3)
            axes[col].scatter(positions, [100 * r["model_interval_mass"]["mean"] for r in rows],
                              marker="_", s=220, color="black", zorder=4, label="Model-implied interval mass")
            axes[col].axhline(90, linestyle=":", color="#343a40", label="Requested 90% level")
            axes[col].set_xticks(positions, ["Pooled\nPoisson", "EB mean\nPoisson", "EB\nNB", "Oracle\nNB"])
            axes[col].set_ylim(0, 105)
            axes[col].set_title("Stable population" if col == 0 else "Shifted population")
            axes[col].set_ylabel("Coverage / interval probability (%)")
            axes[col].legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("New contracts: no history to overcome a stale population prior\nBars: observed coverage ±1.96 MCSE; black marks: model-implied mass", fontsize=12)
        fig.savefig(output / "cold-start-intervals.png", dpi=160)
        plt.close(fig)
