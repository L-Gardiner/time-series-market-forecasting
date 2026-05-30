"""Training entrypoint.

Runs walk-forward validation across all models, saves a metrics report and a
forecast-vs-actual plot of the final fold, then fits the chosen model on the
full history and persists it for serving.

Run with: ``python -m market_forecast.train``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: render to file, never to a window
import matplotlib.pyplot as plt  # noqa: E402

from market_forecast.config import settings  # noqa: E402
from market_forecast.data import load_series  # noqa: E402
from market_forecast.models import XGBForecaster  # noqa: E402
from market_forecast.validate import run_walk_forward  # noqa: E402

# The XGBoost model is the main contribution; this is what we ship for serving.
FINAL_MODEL = XGBForecaster


def _print_summary(results: dict) -> None:
    summary = results["summary"]
    skill = results["skill_vs_naive"]
    print(
        f"\nWalk-forward results ({results['n_splits']} folds x "
        f"{results['test_size']} days), price scale:\n"
    )
    header = f"{'model':<16}{'MAE':>10}{'RMSE':>10}{'MAPE%':>10}{'RMSE skill':>12}"
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        sk = skill[name]["rmse_skill"]
        sk_str = "baseline" if name == "naive" else f"{sk:+.3f}"
        print(f"{name:<16}{m['mae']:>10.2f}{m['rmse']:>10.2f}{m['mape']:>10.3f}{sk_str:>12}")
    print()


def plot_final_fold(results: dict, path) -> None:
    final = results["final_fold"]
    dates = pd.to_datetime(final["dates"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, final["actual"], label="actual", color="black", linewidth=2)
    for name in results["summary"]:
        if name in final:
            ax.plot(dates, final[name], label=name, alpha=0.8)
    ax.set_title(f"Final-fold one-step-ahead forecast vs actual ({settings.series_id})")
    ax.set_ylabel("price")
    ax.legend()
    ax.text(
        0.01,
        0.01,
        "Educational only - not financial advice",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading series {settings.series_id!r} ...")
    series = load_series()
    start = cast(pd.Timestamp, series.index[0])
    end = cast(pd.Timestamp, series.index[-1])
    print(f"Loaded {len(series)} observations ({start.date()} -> {end.date()}).")

    print("Running walk-forward validation ...")
    results = run_walk_forward(series)
    _print_summary(results)

    settings.metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics -> {settings.metrics_path}")

    plot_final_fold(results, settings.plot_path)
    print(f"Saved plot    -> {settings.plot_path}")

    print(f"Fitting final {FINAL_MODEL.name} model on full history ...")
    model = FINAL_MODEL().fit(series)
    artifact = {
        "model": model,
        "series_id": settings.series_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "train_end": end.isoformat(),
        "last_price": float(series.iloc[-1]),
        "summary_metrics": results["summary"][FINAL_MODEL.name],
    }
    joblib.dump(artifact, settings.model_path)
    print(f"Saved model   -> {settings.model_path}")


if __name__ == "__main__":
    main()
