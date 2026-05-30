"""Inference logic shared by ``api.py`` and ``app.py``.

Loads the pre-fitted model artifact (no refitting on request) and turns it into
a horizon forecast with a clear, non-advice disclaimer attached.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
from pydantic import BaseModel, Field

from market_forecast.config import settings

DISCLAIMER = (
    "Educational demonstration only. This is not financial advice and must not "
    "be used for trading or investment decisions. Markets are near-random-walk; "
    "past performance does not predict future results."
)


class ForecastRequest(BaseModel):
    horizon: int = Field(
        default=settings.horizon, ge=1, le=60, description="Number of business days to forecast."
    )


class ForecastPoint(BaseModel):
    date: str
    forecast: float


class ForecastResponse(BaseModel):
    series_id: str
    train_end: str
    last_price: float
    horizon: int
    points: list[ForecastPoint]
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    """Load and cache the fitted model artifact saved by ``train.py``."""
    model_path = Path(path) if path else settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model artifact at {model_path}. Train one first: "
            "`make train` (or `python -m market_forecast.train`)."
        )
    return joblib.load(model_path)


def predict(request: ForecastRequest) -> ForecastResponse:
    artifact = load_artifact()
    model = artifact["model"]
    forecast = model.forecast(request.horizon)

    points = [
        ForecastPoint(date=ts.date().isoformat(), forecast=float(value))
        for ts, value in forecast.items()
    ]
    return ForecastResponse(
        series_id=artifact["series_id"],
        train_end=artifact["train_end"],
        last_price=artifact["last_price"],
        horizon=request.horizon,
        points=points,
    )
