import numpy as np
import pandas as pd
import pytest

from fraud_threshold_policy.costs import expected_cost
from fraud_threshold_policy.policy import evaluate_test, select_threshold


def _val_fixture(n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, n)
    labels = rng.binomial(1, scores**3 * 0.3)
    amounts = rng.uniform(5, 300, n)
    return labels, amounts, scores


def test_selected_threshold_respects_budget_and_is_brute_force_minimum():
    y, a, s = _val_fixture()
    budget = 0.10
    choice = select_threshold(y, a, s, budget=budget)
    assert choice.alert_rate <= budget
    feasible_costs = [
        (expected_cost(y, a, s, t), t)
        for t in choice.grid["threshold"]
        if np.mean(s >= t) <= budget
    ]
    assert choice.expected_cost == pytest.approx(min(c for c, _ in feasible_costs))


def test_selection_never_uses_test_window():
    y, a, s = _val_fixture()
    before = select_threshold(y, a, s, budget=0.05).threshold
    rng = np.random.default_rng(11)
    garbage_test_scores = rng.uniform(0, 1, 200)
    garbage_test_labels = rng.binomial(1, 0.5, 200)
    garbage_test_amounts = rng.uniform(1, 9, 200)
    evaluate_test(garbage_test_labels, garbage_test_amounts, garbage_test_scores, before, budget=0.05)
    after = select_threshold(y, a, s, budget=0.05).threshold
    assert after == before


def test_infeasible_budget_raises_value_error():
    # Every score is 1.0, so every threshold in the grid alerts 100% of rows.
    y = np.array([0, 1, 0, 1])
    a = np.array([10.0, 20.0, 30.0, 40.0])
    s = np.ones(4)
    with pytest.raises(ValueError, match="No threshold meets the alert budget"):
        select_threshold(y, a, s, budget=0.01)


def test_budget_must_be_a_valid_fraction():
    y, a, s = _val_fixture(50)
    with pytest.raises(ValueError):
        select_threshold(y, a, s, budget=0.0)


def test_grid_table_marks_feasibility_and_chooses_feasible_only():
    y, a, s = _val_fixture()
    choice = select_threshold(y, a, s, budget=0.02)
    grid = choice.grid
    assert isinstance(grid, pd.DataFrame)
    assert grid.loc[grid["feasible"], "alert_rate"].le(0.02 + 1e-12).all()
    assert choice.threshold in set(grid.loc[grid["feasible"], "threshold"])
