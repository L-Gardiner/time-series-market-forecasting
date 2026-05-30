"""Typed configuration via pydantic-settings.

Single source of truth for runtime settings. Reads from environment
variables (prefix ``MF_``) and an optional ``.env`` file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MF_",
        extra="ignore",
    )

    app_name: str = "market-forecast"
    log_level: str = "INFO"

    # --- Data -------------------------------------------------------------
    # FRED series id (keyless CSV download). "SP500" is the daily S&P 500 index
    # (~10 years of history); any daily FRED series works, e.g. "DGS10".
    series_id: str = "SP500"
    start_date: str = "2015-01-01"
    end_date: str | None = None  # None -> today
    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")

    # --- Forecasting ------------------------------------------------------
    # Number of business days produced by the serving forecast.
    horizon: int = 5

    # --- Walk-forward validation -----------------------------------------
    # Expanding-window folds and the size of each held-out test block.
    n_splits: int = 5
    test_size: int = 40  # business days per fold's test block
    min_train_size: int = 250  # ~1 trading year before the first fold

    # --- Features (leakage-safe; all built from strictly past data) -------
    lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    seasonal_period: int = 5  # one trading week, for the seasonal-naive baseline

    # --- Models -----------------------------------------------------------
    arima_order: tuple[int, int, int] = (1, 1, 1)
    random_state: int = 42

    @property
    def model_path(self) -> Path:
        return self.model_dir / "forecaster.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.json"

    @property
    def plot_path(self) -> Path:
        return self.model_dir / "forecast_vs_actual.png"


settings = Settings()
