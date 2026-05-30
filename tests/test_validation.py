"""Walk-forward splits never leak the future into training."""

import numpy as np
import pytest

from market_forecast.validate import run_walk_forward, walk_forward_splits


def test_no_forward_leakage() -> None:
    splits = walk_forward_splits(100, n_splits=3, test_size=10, min_train_size=20)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        # The defining property: all training indices precede all test indices.
        assert train_idx.max() < test_idx.min()
        assert len(test_idx) == 10
        # Train and test never overlap.
        assert not set(train_idx.tolist()) & set(test_idx.tolist())


def test_windows_expand_and_reach_end() -> None:
    splits = walk_forward_splits(100, n_splits=3, test_size=10, min_train_size=20)
    train_sizes = [len(tr) for tr, _ in splits]
    assert train_sizes == sorted(train_sizes)  # expanding
    assert train_sizes[0] < train_sizes[-1]
    # Final fold's test block ends at the last observation.
    assert splits[-1][1].max() == 99


def test_test_blocks_are_contiguous_and_ordered() -> None:
    splits = walk_forward_splits(100, n_splits=3, test_size=10, min_train_size=20)
    starts = [test_idx.min() for _, test_idx in splits]
    assert starts == [70, 80, 90]
    for _, test_idx in splits:
        assert np.array_equal(test_idx, np.arange(test_idx.min(), test_idx.min() + 10))


def test_raises_when_insufficient_history() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(25, n_splits=3, test_size=10, min_train_size=20)


def test_run_walk_forward_reports_baseline_and_skill(price_series) -> None:
    from market_forecast.models import NaiveForecaster, XGBForecaster

    results = run_walk_forward(price_series, models=[NaiveForecaster(), XGBForecaster()])
    assert results["n_splits"] >= 1
    assert "naive" in results["summary"]
    # Naive is its own baseline -> zero skill against itself.
    assert results["skill_vs_naive"]["naive"]["rmse_skill"] == 0.0
    # Final-fold predictions are recorded for plotting, one per actual.
    final = results["final_fold"]
    assert len(final["xgboost"]) == len(final["actual"])
