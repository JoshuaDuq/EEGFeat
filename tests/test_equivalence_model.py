from __future__ import annotations

import pathlib

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_classification, cross_fit_regression
from eegfeat.model.scoring import scoring_dict
from eegfeat.model.splits import InnerSplit, loso_folds, within_subject_folds

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "model_reference.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="model fixtures not generated")


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    return dict(np.load(FIXTURE, allow_pickle=False))


def _unflatten(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def test_loso_folds_match_the_reference_pipeline(reference: dict[str, np.ndarray]) -> None:
    train = _unflatten(reference["loso_train"], reference["loso_train_offsets"])
    test = _unflatten(reference["loso_test"], reference["loso_test_offsets"])
    folds = loso_folds(reference["groups"].astype(object))
    assert len(folds) == len(train)
    for fold, expected_train, expected_test in zip(folds, train, test, strict=True):
        np.testing.assert_array_equal(fold.train, expected_train)
        np.testing.assert_array_equal(fold.test, expected_test)


# The reference grid is narrower than this data needs; the pin is on the engine, not the grid.
@pytest.mark.filterwarnings("ignore:regressor__alpha was chosen at the lower end")
def test_nested_loso_predictions_match_the_reference_pipeline(
    reference: dict[str, np.ndarray],
) -> None:
    X = reference["X"]
    y = reference["y"]
    groups = reference["groups"].astype(object)
    pipe = Pipeline([("regressor", Ridge())])
    param_grid = {"regressor__alpha": [0.1, 1.0, 10.0]}
    folds = loso_folds(groups)

    results = cross_fit_regression(
        folds,
        X,
        y,
        groups,
        pipe,
        param_grid,
        inner=InnerSplit(grouping="subject", n_splits=3),
        seed=42,
        scoring=scoring_dict(),
        refit="neg_mse",
    )
    y_pred = np.concatenate([r.y_pred for r in results])
    np.testing.assert_allclose(y_pred, reference["loso_y_pred"], rtol=1e-5, atol=1e-7)


@pytest.mark.filterwarnings("ignore:regressor__alpha was chosen at the lower end")
def test_within_subject_predictions_match_the_reference_pipeline(
    reference: dict[str, np.ndarray],
) -> None:
    ws_X = reference["ws_X"]
    ws_y = reference["ws_y"]
    ws_groups = reference["ws_groups"].astype(object)
    ws_runs = reference["ws_runs"].astype(object)
    pipe = Pipeline([("regressor", Ridge())])
    param_grid = {"regressor__alpha": [0.1, 1.0, 10.0]}
    folds = within_subject_folds(ws_groups, ws_runs, inner_splits=2, outer_splits=2)

    results = cross_fit_regression(
        folds,
        ws_X,
        ws_y,
        ws_groups,
        pipe,
        param_grid,
        inner=InnerSplit(grouping="run", n_splits=2),
        seed=42,
        runs=ws_runs,
        scoring=scoring_dict(),
        refit="neg_mse",
    )
    y_pred = np.concatenate([r.y_pred for r in results])
    np.testing.assert_allclose(y_pred, reference["ws_y_pred"], rtol=1e-5, atol=1e-7)


def test_nested_loso_classification_matches_reference(
    reference: dict[str, np.ndarray],
) -> None:
    from eegfeat.model.estimators import logistic_grid, logistic_pipeline
    from eegfeat.model.transformers import PreprocessingConfig

    clf_X = reference["clf_X"]
    clf_y = reference["clf_y"]
    clf_groups = reference["clf_groups"].astype(object)
    pipe = logistic_pipeline(PreprocessingConfig(), seed=42)
    param_grid = logistic_grid()
    folds = loso_folds(clf_groups)

    results = cross_fit_classification(
        folds,
        clf_X,
        clf_y,
        clf_groups,
        pipe,
        param_grid,
        inner=InnerSplit(grouping="subject", stratified=True, n_splits=2),
        seed=42,
        scoring="average_precision",
    )
    y_pred = np.concatenate([r.y_pred for r in results])
    y_prob = np.concatenate([r.y_prob for r in results if r.y_prob is not None])
    np.testing.assert_array_equal(y_pred, reference["clf_y_pred"])
    np.testing.assert_allclose(y_prob[:, 1], reference["clf_y_prob"], rtol=1e-5, atol=1e-7)
