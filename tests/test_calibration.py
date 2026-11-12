import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fraud_threshold_policy import calibration
from fraud_threshold_policy.calibration import (
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


def test_fit_calibrator_uses_only_the_validation_rows_passed_in(monkeypatch):
    fit_lengths = []
    original_iso_fit = IsotonicRegression.fit
    original_lr_fit = LogisticRegression.fit

    def iso_spy(self, X, y, **kwargs):
        fit_lengths.append(len(X))
        return original_iso_fit(self, X, y, **kwargs)

    def lr_spy(self, X, y, **kwargs):
        fit_lengths.append(len(X))
        return original_lr_fit(self, X, y, **kwargs)

    monkeypatch.setattr(IsotonicRegression, "fit", iso_spy)
    monkeypatch.setattr(LogisticRegression, "fit", lr_spy)
    scores, labels = _scores_and_labels(60)
    fit_calibrator(scores, labels, "sigmoid")
    assert fit_lengths == [60]
    select_calibration(scores, labels)
    assert fit_lengths == [60, 60, 60]


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_maps_higher_scores_to_higher_fraud_probability(method):
    # Regression: a CalibratedClassifierCV wrapper inverted this mapping on real FD02 scores.
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.0, 0.05, 300), rng.uniform(0.9, 1.0, 30)])
    labels = np.concatenate([np.zeros(300, dtype=int), np.ones(30, dtype=int)])
    calibrator = fit_calibrator(scores, labels, method)
    probs = calibrated_probabilities(calibrator, np.array([0.01, 0.97]))
    assert probs[1] > probs[0]
    assert probs[1] > 0.5 > probs[0]


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_brier_beats_constant_baseline_on_informative_scores(method):
    scores, labels = _scores_and_labels(400, seed=11)
    calibrator = fit_calibrator(scores, labels, method)
    probs = calibrated_probabilities(calibrator, scores)
    constant = np.full(len(labels), labels.mean())
    assert calibration.brier(labels, probs) < calibration.brier(labels, constant)


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
