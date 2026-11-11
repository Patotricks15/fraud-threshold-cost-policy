import numpy as np
import pandas as pd
import pytest

from fraud_threshold_policy.data import (
    AMOUNT,
    SCORE,
    SPLIT,
    TARGET,
    TIME,
    time_split,
    validate_scores_schema,
)


def _export(n_val: int = 4, n_test: int = 3) -> pd.DataFrame:
    rows = n_val + n_test
    return pd.DataFrame(
        {
            SPLIT: ["val"] * n_val + ["test"] * n_test,
            TIME: np.arange(rows, dtype=float) * 100.0,
            AMOUNT: np.full(rows, 10.0),
            TARGET: [0, 1, 0, 0, 1, 0, 0][:rows],
            SCORE: np.linspace(0.1, 0.9, rows),
        }
    )


def test_valid_export_passes_schema_check():
    df = _export()
    assert validate_scores_schema(df) is df


@pytest.mark.parametrize("column", [SPLIT, TIME, AMOUNT, TARGET, SCORE])
def test_schema_rejects_missing_column(column):
    df = _export().drop(columns=[column])
    with pytest.raises(ValueError, match="missing columns"):
        validate_scores_schema(df)


def test_schema_rejects_unexpected_split_value():
    df = _export()
    df.loc[0, SPLIT] = "train"
    with pytest.raises(ValueError, match="Unexpected split values"):
        validate_scores_schema(df)


def test_schema_rejects_empty_test_split():
    df = _export()
    df = df[df[SPLIT] == "val"]
    with pytest.raises(ValueError, match="'test' split is empty"):
        validate_scores_schema(df)


def test_schema_rejects_score_outside_unit_interval():
    df = _export()
    df.loc[1, SCORE] = 1.5
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        validate_scores_schema(df)


def test_time_split_is_time_ordered_and_disjoint():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({TIME: np.sort(rng.uniform(0, 172_800, 500))})
    split = time_split(df)
    assert not (split.train & split.val).any()
    assert not (split.val & split.test).any()
    assert (split.train | split.val | split.test).all()
    with pytest.raises(ValueError):
        time_split(df, train_frac=0.9, val_frac=0.2)
