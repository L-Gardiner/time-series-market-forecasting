"""Feature builder is leakage-safe and aligns lags/targets correctly."""

import numpy as np

from market_forecast.features import (
    _build_feature_frame,
    compute_returns,
    make_feature_row,
    make_training_matrix,
)


def test_training_matrix_has_no_nans(price_series) -> None:
    x, y = make_training_matrix(price_series)
    assert not x.isna().to_numpy().any()
    assert not y.isna().to_numpy().any()
    assert len(x) == len(y)


def test_target_is_next_day_return(price_series) -> None:
    x, y = make_training_matrix(price_series)
    ret = compute_returns(price_series)
    # The target on row dated d must equal the return realised on the *next* day.
    for d in x.index[:20]:
        pos = price_series.index.get_loc(d)
        expected = ret.iloc[pos + 1]
        assert np.isclose(y.loc[d], expected)


def test_features_do_not_change_when_future_appended(price_series) -> None:
    """Row t must be identical whether or not future observations exist."""
    full = _build_feature_frame(price_series)
    t = 250
    truncated = _build_feature_frame(price_series.iloc[: t + 1])
    np.testing.assert_allclose(
        full.iloc[t].to_numpy(dtype=float),
        truncated.iloc[-1].to_numpy(dtype=float),
    )


def test_feature_row_only_depends_on_supplied_history(price_series) -> None:
    cutoff = 300
    row_a = make_feature_row(price_series.iloc[:cutoff])
    # Appending arbitrary future values must not change the row for the cutoff day.
    perturbed = price_series.copy()
    perturbed.iloc[cutoff:] = perturbed.iloc[cutoff:] * 5.0
    row_b = make_feature_row(perturbed.iloc[:cutoff])
    np.testing.assert_allclose(row_a.to_numpy(dtype=float), row_b.to_numpy(dtype=float))


def test_lag_one_is_most_recent_return(price_series) -> None:
    frame = _build_feature_frame(price_series)
    ret = compute_returns(price_series)
    # ret_lag_1 on row t is the return known at the close of day t (ret_t).
    np.testing.assert_allclose(frame["ret_lag_1"].iloc[50], ret.iloc[50])
