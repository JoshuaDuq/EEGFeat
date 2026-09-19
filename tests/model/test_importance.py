from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eegfeat.model.importance import (
    Importance,
    aggregate_by,
    permutation_importance,
    permutation_importance_over_folds,
    shap_importance,
    shap_importance_over_folds,
)
from eegfeat.model.splits import Fold, InnerSplit
from eegfeat.table import FeatureMeta

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
needs_shap = pytest.mark.skipif(
    importlib.util.find_spec("shap") is None, reason="shap not installed"
)


def test_permutation_importance_names_every_feature_it_scores() -> None:
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = X[:, 0] * 2.0
    result = permutation_importance(PIPE.fit(X, y), X, y, n_repeats=3, seed=0)
    assert len(result.feature_names) == result.values.size == 3


def test_importance_aggregates_by_a_metadata_field_not_a_name_fragment(
    alpha_beta_meta: tuple[FeatureMeta, FeatureMeta],
) -> None:
    importance = Importance(
        feature_names=("a", "b"), values=np.array([1.0, 3.0]), per_fold=np.empty((0, 2))
    )
    assert aggregate_by(importance, alpha_beta_meta, "band") == {"alpha": 1.0, "beta": 3.0}


def test_shap_importance_raises_when_shap_missing() -> None:
    if importlib.util.find_spec("shap") is None:
        X = np.ones((10, 2))
        with pytest.raises(ModuleNotFoundError, match="SHAP"):
            shap_importance(PIPE, X, ["f1", "f2"])


@needs_shap
def test_shap_importance_computes_with_shap_installed() -> None:
    X = np.random.default_rng(0).normal(size=(20, 2))
    y = X[:, 0] * 1.5
    pipe = Pipeline([("regressor", DummyRegressor())])
    pipe.fit(X, y)
    imp = shap_importance(pipe, X, ["f0", "f1"])
    assert len(imp.feature_names) == 2
    assert imp.values.shape == (2,)


def test_shap_kernel_uses_estimator_predict_fn_for_transformed_features() -> None:
    captured: dict[str, object] = {}

    class FakeKernelExplainer:
        def __init__(self, predict_fn: object, background: object) -> None:
            captured["predict_fn"] = predict_fn
            captured["background"] = background

        def shap_values(self, X_input: object, nsamples: int = 100) -> np.ndarray:
            arr = np.asarray(X_input)
            return np.zeros_like(arr)

    fake_shap = MagicMock()
    fake_shap.KernelExplainer = FakeKernelExplainer

    regressor = DummyRegressor()
    scaler = StandardScaler()
    pipe = Pipeline([("scaler", scaler), ("regressor", regressor)])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    pipe.fit(X, y)

    with (
        patch("eegfeat.model.importance.require_shap", return_value=None),
        patch.dict("sys.modules", {"shap": fake_shap}),
    ):
        shap_importance(pipe, X, ["f1", "f2"])

    assert captured["predict_fn"] == regressor.predict


def test_shap_stage_requires_min_valid_fold_fraction() -> None:
    folds = [
        Fold(index=0, train=np.array([0, 1]), test=np.array([2])),
        Fold(index=1, train=np.array([0, 2]), test=np.array([1])),
    ]
    X = np.ones((3, 2))
    y = np.ones(3)
    groups = np.array(["s1", "s2", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    with (
        patch("eegfeat.model.importance.require_shap", return_value=None),
        patch(
            "eegfeat.model.importance.shap_importance",
            side_effect=[
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([1.0, 2.0]),
                    per_fold=np.empty((0, 2)),
                ),
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([np.nan, np.nan]),
                    per_fold=np.empty((0, 2)),
                ),
            ],
        ),
        pytest.raises(ValueError, match="Insufficient successful folds for SHAP"),
    ):
        shap_importance_over_folds(
            folds,
            X,
            y,
            groups,
            pipe,
            {},
            ["f1", "f2"],
            inner=inner,
            min_complete_fraction=0.8,
        )


def test_permutation_importance_stage_requires_min_valid_fold_fraction() -> None:
    folds = [
        Fold(index=0, train=np.array([0, 1]), test=np.array([2])),
        Fold(index=1, train=np.array([0, 2]), test=np.array([1])),
    ]
    X = np.ones((3, 2))
    y = np.ones(3)
    groups = np.array(["s1", "s2", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    with (
        patch(
            "eegfeat.model.importance.permutation_importance",
            side_effect=[
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([1.0, 2.0]),
                    per_fold=np.empty((0, 2)),
                ),
                Importance(
                    feature_names=("f1", "f2"),
                    values=np.array([np.nan, np.nan]),
                    per_fold=np.empty((0, 2)),
                ),
            ],
        ),
        pytest.raises(
            ValueError, match="Insufficient successful folds for permutation importance"
        ),
    ):
        permutation_importance_over_folds(
            folds,
            X,
            y,
            groups,
            pipe,
            {},
            inner=inner,
            min_complete_fraction=0.8,
        )


def test_permutation_importance_over_folds_runs_and_aggregates() -> None:
    folds = [
        Fold(index=0, train=np.array([0, 1, 2, 3]), test=np.array([4, 5])),
        Fold(index=1, train=np.array([2, 3, 4, 5]), test=np.array([0, 1])),
    ]
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([1.0, 1.2, 2.0, 2.1, 3.0, 3.2])
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
    pipe = Pipeline([("regressor", DummyRegressor())])
    inner = InnerSplit(grouping="subject", n_splits=2)

    res = permutation_importance_over_folds(
        folds, X, y, groups, pipe, {}, inner=inner, feature_names=["f0", "f1"], n_repeats=2
    )
    assert res.per_fold.shape == (2, 2)
    assert res.values.shape == (2,)
    assert res.feature_names == ("f0", "f1")
