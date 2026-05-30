"""Data loading and validation.

Fetches a daily series from FRED's keyless CSV endpoint
(``fredgraph.csv``) -- no API key, no licensing friction -- and returns a clean,
validated price series indexed by date. FRED marks non-trading days with ``"."``;
those rows are dropped (never forward-filled). Raw pulls are cached under
``data/raw/`` (gitignored) and never committed.
"""

from __future__ import annotations

import datetime as dt
from io import StringIO
from pathlib import Path
from typing import cast

import pandas as pd

from market_forecast.config import settings

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _raw_path(series_id: str) -> Path:
    return settings.data_dir / "raw" / f"{series_id}.csv"


def fetch_prices(
    series_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Fetch a daily FRED series as an ascending, dated price Series."""
    import requests

    series_id = series_id or settings.series_id
    start = start or settings.start_date
    end = end or settings.end_date or dt.date.today().isoformat()

    params = {"id": series_id, "cosd": start, "coed": end}
    response = requests.get(FRED_CSV_URL, params=params, timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(
        StringIO(response.text),
        parse_dates=[0],
        na_values=["."],
    )
    frame.columns = ["date", "close"]
    series = frame.dropna().set_index("date")["close"]
    series = cast(pd.Series, series)
    if series.empty:
        raise ValueError(f"No data returned for series {series_id!r} ({start} to {end}).")
    series.name = "close"
    series.index.name = "date"
    return validate_series(series)


def load_series(
    series_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """Load the price series, using a local cache when available."""
    series_id = series_id or settings.series_id
    path = _raw_path(series_id)

    if use_cache and path.exists():
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        return validate_series(cast(pd.Series, frame["close"]))

    series = fetch_prices(series_id, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return series


def validate_series(series: pd.Series) -> pd.Series:
    """Enforce the contract the rest of the pipeline relies on.

    A clean series has a sorted ``DatetimeIndex`` with no duplicate dates, no
    missing values, and at least a couple of observations.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    series = series.astype(float).sort_index()

    if series.index.has_duplicates:
        raise ValueError("Series index contains duplicate dates.")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Series index is not sorted ascending.")
    if series.isna().any():
        raise ValueError("Series contains missing values.")
    if len(series) < 2:
        raise ValueError("Series is too short to forecast.")

    return series
