"""Calibrate the FD02 uncalibrated score on validation rows only.

Isotonic and sigmoid (Platt) calibrators are fitted directly on the score. The sigmoid
method fits a logistic regression on the logit of the score, because scores span many
orders of magnitude near 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fraud_threshold_policy.metrics import brier

CALIBRATION_METHODS = ("isotonic", "sigmoid")
_EPS = 1e-9


def _score_column(X) -> np.ndarray:
    """Accept a single score column (n, 1) or a 1-D array and return a float 1-D array in [0, 1]."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError("calibration expects a single score column")
    if arr.size == 0:
        raise ValueError("calibration received no scores")
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError("calibration scores must be finite and within [0, 1]")
    return arr


@dataclass(frozen=True)
class Calibrator:
    """A fitted map from raw FD02 score to P(fraud)."""

    method: str
    model: object

    def transform(self, scores) -> np.ndarray:
        s = _score_column(scores)
        if self.method == "isotonic":
            return np.asarray(self.model.predict(s), dtype=float)
        logit = np.log(np.clip(s, _EPS, 1 - _EPS) / (1 - np.clip(s, _EPS, 1 - _EPS)))
        return self.model.predict_proba(logit.reshape(-1, 1))[:, 1]


def _labels(y) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("calibration needs both fraud and legitimate rows in validation")
    return labels.astype(int)


def fit_calibrator(val_scores, val_labels, method: str) -> Calibrator:
    """Fit a calibrator on validation scores and labels. The inputs must be validation rows only."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}")
    s = _score_column(val_scores)
    y = _labels(val_labels)
    if len(y) != len(s):
        raise ValueError("val_scores and val_labels must have the same length")
    if method == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(s, y)
    else:
        logit = np.log(np.clip(s, _EPS, 1 - _EPS) / (1 - np.clip(s, _EPS, 1 - _EPS)))
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(logit.reshape(-1, 1), y)
    return Calibrator(method=method, model=model)


def calibrated_probabilities(calibrator: Calibrator, scores) -> np.ndarray:
    """P(fraud) from a fitted calibrator for the given raw scores."""
    return calibrator.transform(scores)


def select_calibration(val_scores, val_labels) -> tuple[Calibrator, str, dict[str, float]]:
    """Fit isotonic and sigmoid calibrators on validation and keep the lower validation Brier score."""
    briers: dict[str, float] = {}
    fitted: dict[str, Calibrator] = {}
    for method in CALIBRATION_METHODS:
        calibrator = fit_calibrator(val_scores, val_labels, method)
        fitted[method] = calibrator
        briers[method] = brier(val_labels, calibrated_probabilities(calibrator, val_scores))
    best = min(CALIBRATION_METHODS, key=lambda m: briers[m])
    return fitted[best], best, briers
