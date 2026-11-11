"""Choose the alert threshold on validation and evaluate it once on test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_threshold_policy.costs import (
    ALERT_BUDGET,
    FP_COST,
    cost_breakdown,
    cost_curve,
    recall_by_amount,
    recall_by_count,
    threshold_grid,
)
from fraud_threshold_policy.metrics import confusion_at_threshold

NAIVE_THRESHOLD = 0.5
_BUDGET_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    alert_rate: float
    expected_cost: float
    grid: pd.DataFrame


def select_threshold(
    y_val,
    amount_val,
    scores_val,
    budget: float = ALERT_BUDGET,
    fp_cost: float = FP_COST,
) -> ThresholdChoice:
    """Minimize validation expected cost among thresholds whose alert rate is within `budget`."""
    if not 0 < budget <= 1:
        raise ValueError("alert budget must be in (0, 1]")
    grid = threshold_grid(scores_val)
    table = pd.DataFrame(cost_curve(y_val, amount_val, scores_val, grid, fp_cost))
    table["feasible"] = table["alert_rate"] <= budget + _BUDGET_TOLERANCE
    feasible = table[table["feasible"]]
    if feasible.empty:
        lowest = float(table["alert_rate"].min())
        raise ValueError(
            f"No threshold meets the alert budget of {budget:.2%} on validation; "
            f"the lowest achievable alert rate is {lowest:.4%}"
        )
    best = feasible.sort_values(
        ["expected_cost", "alert_rate", "threshold"], ascending=[True, True, False]
    ).iloc[0]
    return ThresholdChoice(
        threshold=float(best["threshold"]),
        alert_rate=float(best["alert_rate"]),
        expected_cost=float(best["expected_cost"]),
        grid=table,
    )


def evaluate_at_threshold(y, amount, scores, threshold: float, budget: float = ALERT_BUDGET,
                          fp_cost: float = FP_COST) -> dict:
    """Cost, budget and recall metrics at a fixed threshold on any window."""
    breakdown = cost_breakdown(y, amount, scores, threshold, fp_cost)
    return {
        "threshold": float(threshold),
        "expected_cost": breakdown["expected_cost"],
        "alert_rate": breakdown["alert_rate"],
        "within_budget": bool(breakdown["alert_rate"] <= budget + _BUDGET_TOLERANCE),
        "fp": breakdown["fp"],
        "missed_frauds": breakdown["missed_frauds"],
        "missed_fraud_amount": breakdown["missed_fraud_amount"],
        "confusion": confusion_at_threshold(y, scores, threshold),
        "recall_by_amount": recall_by_amount(y, amount, scores, threshold),
        "recall_by_count": recall_by_count(y, scores, threshold),
    }


def evaluate_test(y_test, amount_test, scores_test, policy_threshold: float,
                  budget: float = ALERT_BUDGET, fp_cost: float = FP_COST) -> dict:
    """Test-window metrics for the chosen policy threshold and for the naive 0.5 threshold."""
    scores = np.asarray(scores_test, dtype=float)
    return {
        "policy": evaluate_at_threshold(y_test, amount_test, scores, policy_threshold, budget, fp_cost),
        "naive_0_5": evaluate_at_threshold(y_test, amount_test, scores, NAIVE_THRESHOLD, budget, fp_cost),
    }
