"""Walk-forward (expanding-window) validation -- the correctness centerpiece.

Time-series must never be split randomly: that leaks the future into training
and produces fantasy metrics. Here every fold trains only on data that comes
*before* its held-out block, and within a fold each one-step-ahead forecast
sees actual history up to (but not including) the day being predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_forecast.config import settings
from market_forecast.models import Forecaster, build_models

Split = tuple[np.ndarray, np.ndarray]


# --- Metrics -------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


# --- Splits --------------------------------------------------------------
def walk_forward_splits(
    n_obs: int,
    n_splits: int | None = None,
    test_size: int | None = None,
    min_train_size: int | None = None,
) -> list[Split]:
    """Expanding-window folds ordered earliest -> latest.

    Fold ``k`` trains on ``[0, test_start)`` and tests on a contiguous block of
    ``test_size`` observations. Guarantees ``max(train) < min(test)`` for every
    fold (no forward leakage).
    """
    n_splits = n_splits or settings.n_splits
    test_size = test_size or settings.test_size
    min_train_size = min_train_size or settings.min_train_size

    first_test_start = n_obs - n_splits * test_size
    if first_test_start < min_train_size:
        raise ValueError(
            f"Not enough data for {n_splits} folds of {test_size} "
            f"(need >= {min_train_size + n_splits * test_size} obs, have {n_obs})."
        )

    splits: list[Split] = []
    for k in range(n_splits):
        test_start = first_test_start + k * test_size
        test_end = test_start + test_size
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


# --- Evaluation loop -----------------------------------------------------
def _evaluate_model_on_fold(
    model: Forecaster, series: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the train block, then one-step-ahead predict across the test block."""
    model.fit(series.iloc[train_idx])
    preds = np.array([model.predict_next(series.iloc[:t]) for t in test_idx])
    actual = series.iloc[test_idx].to_numpy()
    return actual, preds


def run_walk_forward(series: pd.Series, models: list[Forecaster] | None = None) -> dict:
    """Run walk-forward validation for every model and summarise results."""
    models = models if models is not None else build_models()
    splits = walk_forward_splits(len(series))

    per_fold: dict[str, list[dict[str, float]]] = {m.name: [] for m in models}
    final_fold: dict[str, object] = {}
    _, last_test_idx = splits[-1]
    final_dates = pd.DatetimeIndex(series.index[last_test_idx])
    final_fold["dates"] = [d.isoformat() for d in final_dates]
    final_fold["actual"] = series.iloc[last_test_idx].to_numpy().tolist()

    for model in models:
        for fold, (train_idx, test_idx) in enumerate(splits):
            actual, preds = _evaluate_model_on_fold(model, series, train_idx, test_idx)
            metrics = compute_metrics(actual, preds)
            metrics["fold"] = fold
            per_fold[model.name].append(metrics)
            if fold == len(splits) - 1:
                final_fold[model.name] = preds.tolist()

    summary = {
        name: {
            "mae": float(np.mean([f["mae"] for f in folds])),
            "rmse": float(np.mean([f["rmse"] for f in folds])),
            "mape": float(np.mean([f["mape"] for f in folds])),
        }
        for name, folds in per_fold.items()
    }

    # Skill relative to the naive baseline: positive means lower error than naive.
    baseline = summary.get("naive", {})
    skill_vs_naive = {
        name: {
            "mae_skill": 1.0 - s["mae"] / baseline["mae"] if baseline.get("mae") else None,
            "rmse_skill": 1.0 - s["rmse"] / baseline["rmse"] if baseline.get("rmse") else None,
        }
        for name, s in summary.items()
    }

    return {
        "n_splits": len(splits),
        "test_size": settings.test_size,
        "per_fold": per_fold,
        "summary": summary,
        "skill_vs_naive": skill_vs_naive,
        "final_fold": final_fold,
    }
