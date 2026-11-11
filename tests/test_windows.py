import pandas as pd
import pytest

from fraud_threshold_policy.windows import alert_rate_by_window, window_start


def test_t_zero_and_just_before_boundary_share_a_window():
    starts = window_start([0, 21599, 21600])
    assert starts[0] == starts[1] == 0
    assert starts[2] == 21600


def test_alert_rate_by_window_labels_rows_by_window_start():
    times = [0, 21599, 21600, 43200]
    scores = [0.9, 0.1, 0.8, 0.95]
    table = alert_rate_by_window(times, scores, threshold=0.5)
    assert isinstance(table, pd.DataFrame)
    assert list(table["window_start"]) == [0, 21600, 43200]
    assert list(table["transactions"]) == [2, 1, 1]
    assert list(table["alerts"]) == [1, 1, 1]
    assert list(table["alert_rate"]) == [0.5, 1.0, 1.0]


def test_window_rejects_negative_time_and_bad_width():
    with pytest.raises(ValueError):
        window_start([-1.0])
    with pytest.raises(ValueError):
        window_start([0.0], width=0)
