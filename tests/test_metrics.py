"""Metric functions are correct on a tiny known series."""

import numpy as np

from market_forecast.validate import compute_metrics, mae, mape, rmse


def test_perfect_prediction_is_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_known_values() -> None:
    y_true = np.array([2.0, 4.0])
    y_pred = np.array([1.0, 3.0])
    assert mae(y_true, y_pred) == 1.0
    assert rmse(y_true, y_pred) == 1.0
    # |(2-1)/2| = 0.5, |(4-3)/4| = 0.25 -> mean 0.375 -> 37.5%
    assert mape(y_true, y_pred) == 37.5


def test_compute_metrics_keys() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    metrics = compute_metrics(y_true, y_pred)
    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["mae"] == 1.0
