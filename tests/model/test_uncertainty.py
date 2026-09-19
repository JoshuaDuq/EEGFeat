from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.uncertainty import prediction_intervals

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

