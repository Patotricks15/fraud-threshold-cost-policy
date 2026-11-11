"""Expected cost and alert-rate arithmetic for the threshold policy.

Alert rule: a row is alerted when score >= t. A missed fraud is a fraud with score < t.
"""

from __future__ import annotations

import numpy as np

FP_COST = 2.0
ALERT_BUDGET = 0.01


def _as_arrays(y_true, amount, scores):
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(amount, dtype=float)
    s = np.asarray(scores, dtype=float)
    if not (len(y) == len(a) == len(s)):
        raise ValueError("y_true, amount and scores must have the same length")
    if len(y) == 0:
        raise ValueError("Cannot compute costs on an empty window")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    return y, a, s


def alert_rate(scores, threshold: float) -> float:
    """Share of all transactions in the window that are alerted at `threshold`."""
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        raise ValueError("Cannot compute an alert rate on an empty window")
    return float(np.mean(s >= threshold))


def cost_breakdown(y_true, amount, scores, threshold: float, fp_cost: float = FP_COST) -> dict[str, float]:
    """FP count, missed-fraud count and amount, and total expected cost at `threshold`."""
    y, a, s = _as_arrays(y_true, amount, scores)
    alerted = s >= threshold
    fp = int(np.sum(alerted & (y == 0)))
    missed = (~alerted) & (y == 1)
    missed_amount = float(a[missed].sum())
    return {
        "threshold": float(threshold),
        "alert_rate": float(alerted.mean()),
        "fp": fp,
        "missed_frauds": int(missed.sum()),
        "missed_fraud_amount": missed_amount,
        "expected_cost": fp * fp_cost + missed_amount,
    }


def expected_cost(y_true, amount, scores, threshold: float, fp_cost: float = FP_COST) -> float:
    """C(t) = FP(t) * c_FP + sum(Amount of frauds with score < t)."""
    return cost_breakdown(y_true, amount, scores, threshold, fp_cost)["expected_cost"]


def threshold_grid(scores) -> np.ndarray:
    """Unique scores plus the edges 0 and 1, sorted ascending."""
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        raise ValueError("Cannot build a threshold grid from no scores")
    return np.unique(np.concatenate([[0.0, 1.0], s]))


def cost_curve(y_true, amount, scores, grid, fp_cost: float = FP_COST):
    """Vectorized C(t) and alert rate over `grid`. Matches `cost_breakdown` for each t.

    Uses sorted cumulative sums, so cost is O(n log n) rather than O(n * len(grid)).
    Returns a dict of arrays aligned with `grid`.
    """
    y, a, s = _as_arrays(y_true, amount, scores)
    grid = np.asarray(grid, dtype=float)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    a_sorted = a[order]
    fraud_amount_prefix = np.concatenate([[0.0], np.cumsum(np.where(y_sorted == 1, a_sorted, 0.0))])
    legit_prefix = np.concatenate([[0], np.cumsum(y_sorted == 0)])
    n_legit = int(legit_prefix[-1])
    n_below = np.searchsorted(s_sorted, grid, side="left")
    fp = n_legit - legit_prefix[n_below]
    missed_amount = fraud_amount_prefix[n_below]
    return {
        "threshold": grid,
        "alert_rate": (len(s) - n_below) / len(s),
        "fp": fp,
        "missed_fraud_amount": missed_amount,
        "expected_cost": fp * fp_cost + missed_amount,
    }


def recall_by_amount(y_true, amount, scores, threshold: float) -> float:
    """Caught fraud amount divided by total fraud amount at `threshold`."""
    y, a, s = _as_arrays(y_true, amount, scores)
    total = a[y == 1].sum()
    if total <= 0:
        raise ValueError("Recall by amount needs fraud rows with positive total Amount")
    return float(a[(y == 1) & (s >= threshold)].sum() / total)


def recall_by_count(y_true, scores, threshold: float) -> float:
    """Caught frauds divided by total frauds at `threshold`."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("Recall by count needs at least one fraud row")
    return float(np.sum((y == 1) & (s >= threshold)) / n_pos)
