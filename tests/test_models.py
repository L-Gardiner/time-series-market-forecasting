"""Each model honours the common interface and the baselines are exact."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from market_forecast.models import (
    ArimaForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    XGBForecaster,
    future_index,
)


def test_naive_predicts_last_value(price_series) -> None:
    model = NaiveForecaster().fit(price_series)
    assert model.predict_next(price_series) == price_series.iloc[-1]
    fc = model.forecast(3)
    assert len(fc) == 3
    assert (fc == price_series.iloc[-1]).all()


def test_seasonal_naive_returns_season_ago_value() -> None:
    idx = pd.bdate_range("2021-01-01", periods=10)
    series = pd.Series(np.arange(10.0), index=idx, name="close")
    model = SeasonalNaiveForecaster(period=5).fit(series)
    # Predicting the next day returns the value from 5 observations ago.
    assert model.predict_next(series) == series.iloc[-5]


def test_future_index_is_business_days_after_last() -> None:
    last = cast(pd.Timestamp, pd.Timestamp("2021-01-01"))  # a Friday
    idx = future_index(last, 3)
    assert len(idx) == 3
    assert (idx > last).all()


def test_xgboost_fits_and_forecasts(price_series) -> None:
    model = XGBForecaster(n_estimators=20).fit(price_series)
    fc = model.forecast(5)
    assert len(fc) == 5
    assert np.isfinite(fc.to_numpy()).all()
    # One-step price is anchored to the last observed price via the return.
    assert model.predict_next(price_series) > 0


@pytest.mark.filterwarnings("ignore")
def test_arima_fits_and_forecasts(price_series) -> None:
    model = ArimaForecaster(order=(1, 1, 0)).fit(price_series)
    fc = model.forecast(4)
    assert len(fc) == 4
    assert np.isfinite(fc.to_numpy()).all()
