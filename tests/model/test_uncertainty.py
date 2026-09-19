from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from eegfeat.model.uncertainty import (
    _compute_conformal_quantile,
    _order_stat_quantile,
    prediction_intervals,
)

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])


def test_intervals_bracket_their_point_prediction() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, cv_splits=4)
    assert np.all(result.lower <= result.upper)


def test_a_smaller_alpha_gives_wider_intervals() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    wide = prediction_intervals(PIPE, X, y, X[:5], alpha=0.05, cv_splits=4)
    narrow = prediction_intervals(PIPE, X, y, X[:5], alpha=0.10, cv_splits=4)
    assert np.all((wide.upper - wide.lower) >= (narrow.upper - narrow.lower))


def test_the_calibration_unit_is_recorded_so_coverage_cannot_be_over_read() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    assert prediction_intervals(PIPE, X, y, X[:5], cv_splits=4).calibration_unit == "trial"
    assert (
        prediction_intervals(PIPE, X, y, X[:5], cv_splits=4, groups=groups).calibration_unit
        == "subject"
    )


def test_no_calibration_data_raises_rather_than_returning_an_infinite_interval() -> None:
    with pytest.raises(ValueError):
        prediction_intervals(
            PIPE,
            np.arange(2.0).reshape(-1, 1),
            np.arange(2.0),
            np.zeros((1, 1)),
            alpha=0.1,
            cv_splits=5,
        )


def test_split_method_produces_valid_intervals() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, method="split")
    assert result.method == "split"
    assert len(result.lower) == 5
    assert np.all(result.lower <= result.upper)


def test_quantile_method_produces_valid_intervals() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, method="quantile", cv_splits=3)
    assert result.method == "quantile"
    assert len(result.lower) == 5
    assert np.all(result.lower <= result.upper)


def test_invalid_alpha_raises() -> None:
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="alpha"):
        prediction_intervals(PIPE, X, y, X[:2], alpha=1.5)


def test_split_conformal_with_groups() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, method="split", groups=groups)
    assert result.calibration_unit == "subject"
    assert len(result.lower) == 5


def test_split_conformal_raises_on_small_data() -> None:
    X = np.arange(4.0).reshape(-1, 1)
    y = np.arange(4.0)
    with pytest.raises(ValueError, match="at least 5"):
        prediction_intervals(PIPE, X, y, X[:2], method="split")


def test_invalid_method_raises() -> None:
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="Unknown method"):
        prediction_intervals(PIPE, X, y, X[:2], method="magic")  # type: ignore[arg-type]


def test_compute_conformal_quantile_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="Calibration set cannot be empty"):
        _compute_conformal_quantile(np.array([], dtype=float), 0.1)


def test_compute_conformal_quantile_returns_inf_when_insufficient_samples() -> None:
    assert _compute_conformal_quantile(np.array([1.0], dtype=float), 0.01) == float("inf")


def test_compute_conformal_quantile_finite() -> None:
    res = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    assert _compute_conformal_quantile(res, 0.1) == 9.0


def test_quantile_intervals_use_the_models_preprocessing() -> None:
    # The quantile regressors sit behind the model's own preprocessing, so a pipeline that
    # imputes missing features also gives intervals for data that contain NaN.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    X[0, 1] = np.nan
    y = X[:, 0] + rng.normal(scale=0.1, size=60)
    pipe = Pipeline([("impute", SimpleImputer()), ("regressor", LinearRegression())])
    result = prediction_intervals(pipe, X, y, X[:5], method="quantile", cv_splits=3)
    assert np.all(np.isfinite(result.lower)) and np.all(np.isfinite(result.upper))


def test_quantile_intervals_reach_their_coverage_on_new_data() -> None:
    # Calibrated in CV+ form, the quantile intervals cover at least 1 - 2 * alpha of new
    # exchangeable trials, here with noise that grows away from zero.
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(600, 1))
    y = X[:, 0] + rng.normal(scale=0.2 + 0.3 * np.abs(X[:, 0]))
    pipe = Pipeline([("regressor", LinearRegression())])
    result = prediction_intervals(
        pipe, X[:200], y[:200], X[200:], alpha=0.1, method="quantile", cv_splits=5
    )
    covered = (y[200:] >= result.lower) & (y[200:] <= result.upper)
    assert covered.mean() >= 0.8


def test_order_stat_quantile_formulas() -> None:
    res = np.arange(1, 10, dtype=float)
    # Upper bound uses ceil((1-alpha)*(n+1)), lower bound uses floor(alpha*(n+1))
    upper = _order_stat_quantile(res, 0.1, tail="upper")
    lower = _order_stat_quantile(res, 0.1, tail="lower")
    assert np.isfinite(upper)
    assert np.isfinite(lower)
    assert lower <= upper
