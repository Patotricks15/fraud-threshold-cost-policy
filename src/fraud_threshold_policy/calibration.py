"""Calibrate the FD02 uncalibrated score on validation rows only."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from fraud_threshold_policy.metrics import brier

CALIBRATION_METHODS = ("isotonic", "sigmoid")


def _score_column(X) -> np.ndarray:
    """Accept a single score column (n, 1) or a 1-D array and return a float 1-D array in [0, 1]."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError("ScoreEstimator expects a single score column")
    if arr.size == 0:
        raise ValueError("ScoreEstimator received no scores")
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError("ScoreEstimator scores must be finite and within [0, 1]")
    return arr


class ScoreEstimator(BaseEstimator, ClassifierMixin):
    """Pass a single probability-like score through as P(fraud). Nothing is learned."""

    def fit(self, X, y=None):
        _score_column(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        s = _score_column(X)
        return np.column_stack([1.0 - s, s])

    def predict(self, X) -> np.ndarray:
        return (_score_column(X) >= 0.5).astype(int)


def _labels(y) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("calibration needs both fraud and legitimate rows in validation")
    return labels.astype(int)


def fit_calibrator(val_scores, val_labels, method: str) -> CalibratedClassifierCV:
    """Fit a calibrator on validation scores and labels. The inputs must be validation rows only."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}")
    X = _score_column(val_scores).reshape(-1, 1)
    y = _labels(val_labels)
    if len(y) != len(X):
        raise ValueError("val_scores and val_labels must have the same length")
    base = ScoreEstimator().fit(X, y)
    calibrator = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    calibrator.fit(X, y)
    return calibrator


def calibrated_probabilities(calibrator: CalibratedClassifierCV, scores) -> np.ndarray:
    """P(fraud) from a fitted calibrator for the given raw scores."""
    return calibrator.predict_proba(_score_column(scores).reshape(-1, 1))[:, 1]


def select_calibration(val_scores, val_labels) -> tuple[CalibratedClassifierCV, str, dict[str, float]]:
    """Fit isotonic and sigmoid calibrators on validation and keep the lower validation Brier score."""
    briers: dict[str, float] = {}
    fitted: dict[str, CalibratedClassifierCV] = {}
    for method in CALIBRATION_METHODS:
        calibrator = fit_calibrator(val_scores, val_labels, method)
        fitted[method] = calibrator
        briers[method] = brier(val_labels, calibrated_probabilities(calibrator, val_scores))
    best = min(CALIBRATION_METHODS, key=lambda m: briers[m])
    return fitted[best], best, briers
