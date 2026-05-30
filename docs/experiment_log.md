# Experiment Log: time-series-market-forecasting

A lightweight running log. One row per meaningful experiment. All metrics are
walk-forward (5 expanding folds × 40 trading days), one-step-ahead, price scale,
on FRED `SP500`.

| Date | Exp ID | Change / hypothesis | Key params | RMSE | MAPE % | Result vs baseline | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-30 | exp-001 | Naive random walk | — | 51.46 | 0.594 | — | Reference point (the bar to beat) |
| 2026-05-30 | exp-002 | Seasonal naive (1 week) | period=5 | 109.26 | 1.336 | −1.123 RMSE skill | Far worse; weekly seasonality on price is the wrong prior |
| 2026-05-30 | exp-003 | ARIMA on level | order=(1,1,1) | 51.71 | 0.599 | −0.005 RMSE skill | Essentially ties naive; differencing ≈ random walk |
| 2026-05-30 | exp-004 | XGBoost on returns + lag/rolling/calendar | depth=3, n=300, lr=0.05 | 51.10 | 0.586 | +0.007 RMSE skill | Marginally beats naive; chosen for serving |

## Decisions
- **Chosen approach:** XGBoost on **returns** (not price level). It marginally and
  consistently beats the naive baseline while keeping the target stationary, and
  reconstructing price from a predicted return avoids tree models' inability to
  extrapolate trends.
- **Discarded:** seasonal-naive (wrong prior for index prices); modelling the
  **price level** directly with trees (cannot extrapolate beyond the training
  range). ARIMA retained as a statistical reference but not shipped.
- **Honest takeaway:** on a liquid daily index the achievable skill over naive is
  tiny. The value of this project is the leakage-safe, walk-forward methodology
  and baseline-relative reporting, not the headline accuracy.
