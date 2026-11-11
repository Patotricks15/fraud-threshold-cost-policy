"""Load and validate the FD02 score export, plus the FD01 time split helper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Class"
TIME = "Time"
AMOUNT = "Amount"
SPLIT = "split"
SCORE = "score"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
VALID_SPLITS = (VAL_SPLIT, TEST_SPLIT)
REQUIRED_COLUMNS = {SPLIT, TIME, AMOUNT, TARGET, SCORE}

SCORES_PATH_ENV = "FD02_SCORES_PATH"
DEFAULT_SCORES_PATHS = (Path("data/fd02_scores.csv"), Path("../data/fd02_scores.csv"))


def resolve_scores_path() -> Path:
    """Return the FD02 export path: env override first, then the default locations."""
    env_path = os.environ.get(SCORES_PATH_ENV)
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"{SCORES_PATH_ENV} points to a missing file: {path}")
        return path
    for candidate in DEFAULT_SCORES_PATHS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "fd02_scores.csv not found. Copy the FD02 export to data/fd02_scores.csv "
        f"or set {SCORES_PATH_ENV}. See data/README.md."
    )


def validate_scores_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check the FD02 export schema and value ranges, then return the frame unchanged."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"FD02 export is missing columns: {sorted(missing)}")
    unexpected = set(df[SPLIT].dropna().unique()) - set(VALID_SPLITS)
    if unexpected or df[SPLIT].isna().any():
        raise ValueError(
            f"Unexpected split values {sorted(map(str, unexpected))}; "
            f"allowed values are {list(VALID_SPLITS)} with no blanks"
        )
    for name in VALID_SPLITS:
        if not (df[SPLIT] == name).any():
            raise ValueError(f"The '{name}' split is empty in the FD02 export")
    scores = pd.to_numeric(df[SCORE], errors="coerce")
    if scores.isna().any() or not scores.between(0, 1).all():
        raise ValueError("score must be numeric and within [0, 1] for every row")
    labels = pd.to_numeric(df[TARGET], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Class must be 0 or 1 for every row")
    for column in (TIME, AMOUNT):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite and numeric for every row")
    return df


def load_scores(path: str | Path | None = None) -> pd.DataFrame:
    """Load the FD02 export, validate it, and sort it by transaction time."""
    scores_path = Path(path) if path is not None else resolve_scores_path()
    df = pd.read_csv(scores_path)
    validate_scores_schema(df)
    df = df.astype({SCORE: float, TARGET: int, AMOUNT: float, TIME: float})
    return df.sort_values(TIME, kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class TimeSplit:
    """Boolean masks for a time-ordered train, validation and test split."""

    train: pd.Series
    val: pd.Series
    test: pd.Series
    train_end: float
    val_end: float


def time_split(
    df: pd.DataFrame,
    time_col: str = TIME,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TimeSplit:
    """Split by time quantiles: earliest 70% of time, next 15%, last 15%."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum below 1")
    times = df[time_col]
    train_end = float(times.quantile(train_frac))
    val_end = float(times.quantile(train_frac + val_frac))
    train = times <= train_end
    val = (times > train_end) & (times <= val_end)
    test = times > val_end
    for name, mask in (("train", train), ("val", val), ("test", test)):
        if not mask.any():
            raise ValueError(f"The {name} split is empty; check the time column")
    return TimeSplit(train=train, val=val, test=test, train_end=train_end, val_end=val_end)
