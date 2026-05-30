"""API returns a valid schema for a sample horizon (TestClient, no network)."""

from datetime import UTC, datetime

import joblib
import pytest
from fastapi.testclient import TestClient

from market_forecast import predict as predict_mod
from market_forecast.api import app
from market_forecast.config import settings
from market_forecast.models import XGBForecaster


@pytest.fixture
def client(price_series, tmp_path, monkeypatch):
    # Train a tiny model and persist an artifact to a temporary model dir.
    monkeypatch.setattr(settings, "model_dir", tmp_path)
    model = XGBForecaster(n_estimators=20).fit(price_series)
    artifact = {
        "model": model,
        "series_id": "TEST",
        "trained_at": datetime.now(UTC).isoformat(),
        "train_end": price_series.index[-1].isoformat(),
        "last_price": float(price_series.iloc[-1]),
        "summary_metrics": {"mae": 1.0, "rmse": 1.0, "mape": 1.0},
    }
    joblib.dump(artifact, settings.model_path)
    predict_mod.load_artifact.cache_clear()
    yield TestClient(app)
    predict_mod.load_artifact.cache_clear()


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_schema(client) -> None:
    response = client.post("/forecast", json={"horizon": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["series_id"] == "TEST"
    assert body["horizon"] == 3
    assert len(body["points"]) == 3
    assert {"date", "forecast"} == set(body["points"][0])
    assert "not financial advice" in body["disclaimer"].lower()


def test_forecast_rejects_out_of_range_horizon(client) -> None:
    response = client.post("/forecast", json={"horizon": 999})
    assert response.status_code == 422
