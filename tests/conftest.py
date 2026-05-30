"""Shared fixtures: deterministic synthetic price series and a trained artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_series() -> pd.Series:
    """A deterministic random-walk price series on business days."""
    rng = np.random.default_rng(0)
    n = 500
    steps = rng.normal(0.0005, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(steps))
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.Series(prices, index=idx, name="close")
