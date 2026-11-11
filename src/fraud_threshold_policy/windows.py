"""Six-hour time windows, labeled by window start in seconds from the first transaction."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_SECONDS = 6 * 3600


def window_start(time_seconds, width: int = WINDOW_SECONDS):
    """Start of the window containing each time: floor(t / width) * width."""
    if width <= 0:
        raise ValueError("window width must be positive")
    t = np.asarray(time_seconds, dtype=float)
    if np.any(t < 0) or not np.isfinite(t).all():
        raise ValueError("Time must be finite and non-negative")
    return np.floor(t / width) * width


def alert_rate_by_window(time_seconds, scores, threshold: float, width: int = WINDOW_SECONDS) -> pd.DataFrame:
    """Rows, alerts and alert rate per window, labeled by window start."""
    starts = window_start(time_seconds, width)
    alerted = np.asarray(scores, dtype=float) >= threshold
    frame = pd.DataFrame({"window_start": starts, "alerted": alerted})
    table = frame.groupby("window_start", as_index=False).agg(
        transactions=("alerted", "size"), alerts=("alerted", "sum")
    )
    table["alert_rate"] = table["alerts"] / table["transactions"]
    return table
