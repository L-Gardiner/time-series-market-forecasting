"""FastAPI inference service (optional demo).

``POST /forecast`` takes a horizon and returns predicted prices with timestamps.
The fitted model is loaded once at startup; requests never refit.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from market_forecast import __version__
from market_forecast.predict import (
    DISCLAIMER,
    ForecastRequest,
    ForecastResponse,
    predict,
)

app = FastAPI(
    title="market-forecast",
    version=__version__,
    description=f"Time-series price forecasting. {DISCLAIMER}",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
    try:
        return predict(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
