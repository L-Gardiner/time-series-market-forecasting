# Model Card: time-series-market-forecasting

## Overview
- **Task:** univariate daily time-series forecasting (next-day price; recursive
  multi-step for serving).
- **Model type:** XGBoost regressor on engineered return features (shipped),
  benchmarked against naive, seasonal-naive, and ARIMA.
- **Version:** 0.1.0
- **Date:** 2026-05-30

## Intended use
- **Primary use case:** an educational demonstration of leakage-safe time-series
  modelling and walk-forward evaluation for a portfolio.
- **Out-of-scope / not intended for:** trading, investment, or any real financial
  decision. **This is not financial advice.** No uncertainty quantification, no
  exogenous information, no regime-change handling.

## Data
- **Training data:** FRED `SP500` daily index, ~2,500 trading days (~10 years),
  pulled via the keyless `fredgraph.csv` endpoint. Non-trading days dropped.
- **Evaluation data:** the most recent folds of the same series, held out via
  expanding-window walk-forward (no overlap with training within a fold).
- **Known biases / gaps:** single liquid index; survivorship/structural-break
  effects not modelled; daily frequency only.

## Metrics
Walk-forward, 5 expanding folds × 40 trading days, one-step-ahead, price scale.

| Metric | Value (XGBoost) | Notes |
|---|---|---|
| MAE | 39.85 | vs naive 40.42 |
| RMSE | 51.10 | vs naive 51.46 |
| MAPE | 0.586% | vs naive 0.594% |
| RMSE skill vs naive | +0.007 | `1 − model/naive` |

ARIMA ≈ naive (RMSE skill −0.005); seasonal-naive is much worse (−1.123).

## Evaluation method
Expanding-window **walk-forward** validation (`src/market_forecast/validate.py`).
Each fold trains only on data preceding its held-out block; within a fold,
parameters are frozen while one-step-ahead forecasts consume the actual expanding
history. Every model is compared to the **naive random-walk baseline**, which is
the benchmark to beat. A random train/test split is deliberately **not** used.

## Limitations & ethical considerations
- Daily index returns are near-random-walk; measured skill over naive is a
  fraction of a percent and is **not** a trading edge.
- Point forecasts only; no prediction intervals.
- The multi-step serving forecast is recursive and degrades quickly — illustrative.
- Misuse risk (treating output as investment guidance) is mitigated by a prominent
  disclaimer in the README, the UI, the API description, and the plot.

## Caveats
**Educational only — not financial advice.** Do not use these forecasts for
trading or investment. Past performance does not predict future results.
