import numpy as np
import pytest

from fraud_threshold_policy.costs import (
    alert_rate,
    cost_breakdown,
    cost_curve,
    expected_cost,
    recall_by_amount,
    recall_by_count,
    threshold_grid,
)

# Hand-computed 6-row example at t = 0.5 (alert when score >= 0.5):
#   row  y  amount  score  alerted
#   0    0    10    0.1    no
#   1    0    20    0.6    yes  -> false alert, cost 2.0
#   2    1    30    0.2    no   -> missed fraud, amount 30
#   3    1    40    0.7    yes  -> caught
#   4    0    50    0.9    yes  -> false alert, cost 2.0
#   5    1    60    0.3    no   -> missed fraud, amount 60
# FP = 2, missed amount = 30 + 60 = 90, so C(0.5) = 2 * 2.0 + 90 = 94.
Y = [0, 0, 1, 1, 0, 1]
AMOUNT = [10, 20, 30, 40, 50, 60]
SCORES = [0.1, 0.6, 0.2, 0.7, 0.9, 0.3]


def test_expected_cost_matches_hand_computed_example():
    assert expected_cost(Y, AMOUNT, SCORES, 0.5) == pytest.approx(94.0)


def test_cost_breakdown_reports_components():
    out = cost_breakdown(Y, AMOUNT, SCORES, 0.5)
    assert out["fp"] == 2
    assert out["missed_frauds"] == 2
    assert out["missed_fraud_amount"] == pytest.approx(90.0)
    assert out["alert_rate"] == pytest.approx(0.5)
    assert out["expected_cost"] == pytest.approx(94.0)


def test_alert_rate_and_recalls_on_example():
    assert alert_rate(SCORES, 0.5) == pytest.approx(0.5)
    assert recall_by_count(Y, SCORES, 0.5) == pytest.approx(1 / 3)
    assert recall_by_amount(Y, AMOUNT, SCORES, 0.5) == pytest.approx(40 / 130)


def test_vectorized_cost_curve_matches_direct_formula_with_ties():
    rng = np.random.default_rng(7)
    scores = np.round(rng.uniform(0, 1, 300), 2)
    labels = rng.binomial(1, 0.2, 300)
    amounts = rng.uniform(1, 500, 300)
    grid = threshold_grid(scores)
    curve = cost_curve(labels, amounts, scores, grid)
    for i, t in enumerate(grid):
        assert curve["expected_cost"][i] == pytest.approx(expected_cost(labels, amounts, scores, t))
        assert curve["alert_rate"][i] == pytest.approx(alert_rate(scores, t))


def test_threshold_grid_contains_scores_and_edges():
    grid = threshold_grid([0.3, 0.3, 0.7])
    assert list(grid) == [0.0, 0.3, 0.7, 1.0]


def test_cost_rejects_mismatched_lengths_and_bad_labels():
    with pytest.raises(ValueError):
        expected_cost([0, 1], [1.0], [0.1, 0.9], 0.5)
    with pytest.raises(ValueError):
        expected_cost([0, 2], [1.0, 1.0], [0.1, 0.9], 0.5)
