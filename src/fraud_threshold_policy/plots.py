"""Figures for the FD05 report. Each title names the dataset, split and model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402


def save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_curve(table, chosen_threshold: float, title: str, path: str | Path) -> Path:
    """Expected cost C(t) across the validation grid, with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    feasible = table[table["feasible"]]
    ax.plot(feasible["threshold"], feasible["expected_cost"], color="tab:blue", label="Feasible (within budget)")
    infeasible = table[~table["feasible"]]
    if not infeasible.empty:
        ax.plot(infeasible["threshold"], infeasible["expected_cost"], color="lightgrey", label="Over budget")
    ax.axvline(chosen_threshold, color="tab:red", linestyle="--", label=f"Chosen t = {chosen_threshold:.4f}")
    ax.set_xlabel("Calibrated threshold t (alert if score >= t)")
    ax.set_ylabel("Expected cost C(t) (currency units)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, path)


def plot_reliability(y_true, curves: dict[str, object], title: str, path: str | Path, bins: int = 10) -> Path:
    """Reliability diagram: one curve per probability source (for example raw and calibrated)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for label, probs in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, path)


def plot_alert_rate_by_window(table, budget: float, title: str, path: str | Path) -> Path:
    """Alert rate per 6-hour window, labeled by window start in hours from the first transaction."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{h:g}h" for h in table["window_start"] / 3600.0]
    ax.bar(labels, table["alert_rate"], color="tab:blue")
    ax.axhline(budget, color="tab:red", linestyle="--", label=f"Budget = {budget:.2%}")
    ax.set_xlabel("Window start (hours from first transaction)")
    ax.set_ylabel("Alert rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return save(fig, path)
