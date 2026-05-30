"""Forecasting models behind one common interface.

Every model implements:

* ``fit(y)``              -- learn from a training price series.
* ``predict_next(history)`` -- one-step-ahead price, using only ``history``
  (actual values strictly before the target day). This is what walk-forward
  validation calls, so all models are compared apples-to-apples on the same
  one-step task.
* ``forecast(steps)``     -- a recursive multi-step price path for serving.

Models that learn returns (XGBoost) reconstruct a price from the predicted
return so that every model is evaluated on the same price scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd

from market_forecast.config import settings
from market_forecast.features import make_feature_row, make_training_matrix


def future_index(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    """Business-day index for the ``steps`` days after ``last_date``."""
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)


class Forecaster(ABC):
    name: str = "forecaster"

    @abstractmethod
    def fit(self, y: pd.Series) -> Forecaster: ...

    @abstractmethod
    def predict_next(self, history: pd.Series) -> float:
        """One-step-ahead price given actual history ending the day before."""

    @abstractmethod
    def forecast(self, steps: int) -> pd.Series:
        """Recursive multi-step forecast from the end of the training series."""


class NaiveForecaster(Forecaster):
    """Random-walk baseline: tomorrow equals today. The bar to beat."""

    name = "naive"

    def fit(self, y: pd.Series) -> NaiveForecaster:
        self._train = y
        return self

    def predict_next(self, history: pd.Series) -> float:
        return float(history.iloc[-1])

    def forecast(self, steps: int) -> pd.Series:
        last = self._train.iloc[-1]
        idx = future_index(cast(pd.Timestamp, self._train.index[-1]), steps)
        return pd.Series(np.full(steps, last), index=idx, name="forecast")


class SeasonalNaiveForecaster(Forecaster):
    """Seasonal baseline: repeat the value from one season (week) ago."""

    name = "seasonal_naive"

    def __init__(self, period: int | None = None) -> None:
        self.period = period or settings.seasonal_period

    def fit(self, y: pd.Series) -> SeasonalNaiveForecaster:
        self._train = y
        return self

    def predict_next(self, history: pd.Series) -> float:
        if len(history) < self.period:
            return float(history.iloc[-1])
        return float(history.iloc[-self.period])

    def forecast(self, steps: int) -> pd.Series:
        season = self._train.iloc[-self.period :].to_numpy()
        values = np.array([season[i % self.period] for i in range(steps)])
        idx = future_index(cast(pd.Timestamp, self._train.index[-1]), steps)
        return pd.Series(values, index=idx, name="forecast")


class ArimaForecaster(Forecaster):
    """ARIMA via statsmodels SARIMAX. Differencing handles non-stationarity."""

    name = "arima"

    def __init__(self, order: tuple[int, int, int] | None = None) -> None:
        self.order = order or settings.arima_order

    def _build(self, y: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA

        # A plain integer index keeps SARIMAX from warning about irregular dates.
        endog = y.reset_index(drop=True)
        return ARIMA(endog, order=self.order)

    def fit(self, y: pd.Series) -> ArimaForecaster:
        self._train = y
        self._result = self._build(y).fit()
        return self

    def predict_next(self, history: pd.Series) -> float:
        # Re-apply the fitted parameters to the new history without refitting.
        result = self._result.apply(history.reset_index(drop=True), refit=False)
        return float(result.forecast(1).iloc[0])

    def forecast(self, steps: int) -> pd.Series:
        values = np.asarray(self._result.forecast(steps), dtype=float)
        idx = future_index(cast(pd.Timestamp, self._train.index[-1]), steps)
        return pd.Series(values, index=idx, name="forecast")


class XGBForecaster(Forecaster):
    """Gradient-boosted trees on lag/rolling/calendar features of returns.

    The model predicts a next-day return; the price is reconstructed as
    ``last_price * (1 + predicted_return)``. Modelling returns keeps the target
    stationary and avoids tree models' inability to extrapolate price levels.
    """

    name = "xgboost"

    def __init__(self, **params: object) -> None:
        self.params: dict[str, object] = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": settings.random_state,
            "n_jobs": -1,
            **params,
        }

    def fit(self, y: pd.Series) -> XGBForecaster:
        from xgboost import XGBRegressor

        self._train = y
        x, target = make_training_matrix(y)
        self._columns = list(x.columns)
        self._model = XGBRegressor(**self.params)
        self._model.fit(x, target)
        return self

    def _predict_return(self, history: pd.Series) -> float:
        row = make_feature_row(history)[self._columns]
        return float(self._model.predict(row)[0])

    def predict_next(self, history: pd.Series) -> float:
        ret = self._predict_return(history)
        return float(history.iloc[-1] * (1.0 + ret))

    def forecast(self, steps: int) -> pd.Series:
        working = self._train.copy()
        idx = future_index(cast(pd.Timestamp, self._train.index[-1]), steps)
        preds: list[float] = []
        for date in idx:
            next_price = self.predict_next(working)
            preds.append(next_price)
            working = pd.concat([working, pd.Series([next_price], index=[date])])
        return pd.Series(preds, index=idx, name="forecast")


# Registry used by training, validation and serving.
def build_models() -> list[Forecaster]:
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(),
        ArimaForecaster(),
        XGBForecaster(),
    ]


MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "arima": ArimaForecaster,
    "xgboost": XGBForecaster,
}
