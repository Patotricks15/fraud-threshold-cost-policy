import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from fraud_threshold_policy.calibration import (
    ScoreEstimator,
    calibrated_probabilities,
    fit_calibrator,
    select_calibration,
)


def _scores_and_labels(n: int = 60, seed: int = 5):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, n)
    labels = rng.binomial(1, scores)
    labels[0], labels[1] = 0, 1
    return scores, labels


def test_score_estimator_probabilities_sum_to_one_and_preserve_order():
    scores = np.array([0.05, 0.9, 0.2, 0.61, 0.0, 1.0])
    est = ScoreEstimator().fit(scores.reshape(-1, 1), None)
    probs = est.predict_proba(scores.reshape(-1, 1))
    assert probs.shape == (6, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[:, 1], scores)
    order = np.argsort(scores)
    assert np.all(np.diff(probs[order, 1]) >= 0)


def test_score_estimator_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        ScoreEstimator().fit(np.array([[1.2], [0.1]]), None)


def test_fit_calibrator_uses_only_the_validation_rows_passed_in(monkeypatch):
    fit_lengths = []
    original_fit = CalibratedClassifierCV.fit

    def spy(self, X, y, **kwargs):
        fit_lengths.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(CalibratedClassifierCV, "fit", spy)
    scores, labels = _scores_and_labels(60)
    fit_calibrator(scores, labels, "sigmoid")
    assert fit_lengths == [60]
    select_calibration(scores, labels)
    assert fit_lengths == [60, 60, 60]


def test_calibrated_probabilities_are_in_unit_interval():
    scores, labels = _scores_and_labels(80)
    calibrator, method, briers = select_calibration(scores, labels)
    assert method in ("isotonic", "sigmoid")
    assert set(briers) == {"isotonic", "sigmoid"}
    probs = calibrated_probabilities(calibrator, scores)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_fit_calibrator_rejects_single_class_validation():
    scores = np.linspace(0.1, 0.9, 10)
    with pytest.raises(ValueError, match="both fraud and legitimate"):
        fit_calibrator(scores, np.zeros(10, dtype=int), "isotonic")


def test_fit_calibrator_rejects_unknown_method():
    scores, labels = _scores_and_labels(20)
    with pytest.raises(ValueError):
        fit_calibrator(scores, labels, "platt-ish")
