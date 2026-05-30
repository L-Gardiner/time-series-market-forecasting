"""Leakage-safe feature engineering.

The ML model forecasts the *next-day return* (a roughly stationary target) and
the predicted return is later turned back into a price. Every feature on row
``t`` is a function of information available at the close of day ``t`` only;
the target is the return realised on day ``t + 1``. Nothing in here ever reads
a future value.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from market_forecast.config import settings


def compute_returns(close: pd.Series) -> pd.Series:
    """Simple daily return. ``ret_t`` is known at the close of day ``t``."""
    return cast(pd.Series, close.pct_change()).rename("ret")


def _build_feature_frame(close: pd.Series) -> pd.DataFrame:
    """Feature matrix indexed by date. Row ``t`` uses only data up to day ``t``.

    Warm-up rows (insufficient history for a lag/window) are left as NaN and
    dropped by the caller; they are never imputed with future information.
    """
    ret = compute_returns(close)
    feats: dict[str, pd.Series] = {}

    # Lagged returns: lag L is the L-th most recent known return.
    for lag in settings.lags:
        feats[f"ret_lag_{lag}"] = cast(pd.Series, ret.shift(lag - 1))

    # Rolling statistics over known returns and price-vs-trend distance.
    for window in settings.rolling_windows:
        feats[f"ret_mean_{window}"] = cast(pd.Series, ret.rolling(window).mean())
        feats[f"ret_std_{window}"] = cast(pd.Series, ret.rolling(window).std())
        feats[f"px_vs_ma_{window}"] = cast(pd.Series, close / close.rolling(window).mean() - 1.0)

    frame = pd.DataFrame(feats, index=close.index)

    # Calendar features (the row's own date is always known).
    dates = pd.Series(pd.DatetimeIndex(close.index), index=close.index)
    frame["dayofweek"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month

    return frame


def make_training_matrix(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` aligned for supervised learning.

    ``y`` is the next-day return placed on row ``t``; rows with any NaN feature
    or a missing (final-row) target are dropped.
    """
    features = _build_feature_frame(close)
    target = cast(pd.Series, compute_returns(close).shift(-1)).rename("target_ret")

    data = features.join(target)
    data = data.dropna()
    y = data.pop("target_ret")
    return data, y


def make_feature_row(close: pd.Series) -> pd.DataFrame:
    """Build the single feature row used to forecast the day after ``close`` ends."""
    features = _build_feature_frame(close)
    last = features.iloc[[-1]]
    if last.isna().to_numpy().any():
        raise ValueError(
            "Not enough history to build features for the latest point; supply a longer series."
        )
    return last


def warmup_period() -> int:
    """Rows of history consumed before the first complete feature row exists."""
    max_lag = max(settings.lags)
    max_window = max(settings.rolling_windows)
    return max(max_lag, max_window)
