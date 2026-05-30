"""Smoke tests: the package imports and core wiring exists."""

import importlib


def test_package_imports() -> None:
    mod = importlib.import_module("market_forecast")
    assert hasattr(mod, "__version__")


def test_config_loads() -> None:
    from market_forecast.config import settings

    assert settings.app_name
    assert settings.horizon >= 1
