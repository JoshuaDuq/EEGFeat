from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_regression
from eegfeat.model.estimators import (
    elasticnet_grid,
    elasticnet_pipeline,
    ensemble_pipeline,
    logistic_grid,
    logistic_pipeline,
    random_forest_classifier_grid,
    random_forest_classifier_pipeline,
    random_forest_grid,
    random_forest_pipeline,
    ridge_grid,
    ridge_pipeline,
    svm_grid,
    svm_pipeline,
)
from eegfeat.model.splits import InnerSplit, loso_folds
from eegfeat.model.transformers import PreprocessingConfig

_CONFIG = PreprocessingConfig()


def test_the_pipeline_scales_before_it_regularizes() -> None:
    # ElasticNet penalizes coefficients on their own scale, so an unscaled feature in
    # different units is regularized differently from an identical one in volts.
    names = [name for name, _ in elasticnet_pipeline(_CONFIG, seed=0).steps]
    assert names.index("scaler") < names.index("regressor")


def test_the_seed_reaches_the_estimator() -> None:
    pipe = elasticnet_pipeline(_CONFIG, seed=17)
    assert pipe.named_steps["regressor"].random_state == 17


def test_pipelines_are_built_fresh_not_shared() -> None:
    assert elasticnet_pipeline(_CONFIG, seed=0) is not elasticnet_pipeline(_CONFIG, seed=0)


@pytest.mark.parametrize(
    ("pipeline_factory", "grid_factory"),
    [
        (elasticnet_pipeline, elasticnet_grid),
        (ridge_pipeline, ridge_grid),
        (random_forest_pipeline, random_forest_grid),
        (svm_pipeline, svm_grid),
        (logistic_pipeline, logistic_grid),
        (random_forest_classifier_pipeline, random_forest_classifier_grid),
    ],
)
@pytest.mark.parametrize("n_covariates", [0, 2])
def test_every_grid_key_names_a_step_that_exists_in_its_pipeline(
    pipeline_factory: Callable[..., Pipeline],
    grid_factory: Callable[..., dict[str, list[object]]],
    n_covariates: int,
) -> None:
    # A grid key that does not resolve is not an error in GridSearchCV until fit time,
    # and then it reports a parameter name rather than the typo that caused it.
    pipe = pipeline_factory(_CONFIG, seed=42, n_covariates=n_covariates)
    grid = grid_factory()
    valid_params = set(pipe.get_params())
    assert set(grid.keys()) <= valid_params


def test_default_grids_give_the_same_predictions_whatever_the_feature_units() -> None:
    # A FeatureTable mixes units (V^2, log power, ratios, PLV). A variance threshold on
    # unscaled values selected features by their unit, and PLV-like columns, with variance
    # near 0.003, were removed entirely, which crashed the fold.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(6)], 20).astype(object)
    X = rng.uniform(0.3, 0.5, size=(120, 8))
    y = 10.0 * X[:, 0] + rng.normal(size=120)
    folds = loso_folds(groups)
    inner = InnerSplit(grouping="subject", n_splits=3)
    pipe = ridge_pipeline(_CONFIG, seed=0)
    as_given, rescaled = (
        cross_fit_regression(folds, values, y, groups, pipe, ridge_grid(), inner=inner, seed=0)
        for values in (X, X * 1e-6)
    )
    np.testing.assert_allclose(
        np.concatenate([p.y_pred for p in as_given]),
        np.concatenate([p.y_pred for p in rescaled]),
        rtol=1e-6,
    )


def test_ensemble_pipeline_contains_all_base_classifiers() -> None:
    pipe = ensemble_pipeline(_CONFIG, seed=42)
    ensemble = pipe.named_steps["ensemble"]
    named_estimators = dict(ensemble.estimators)
    assert "svm" in named_estimators
    assert "lr" in named_estimators
    assert "rf" in named_estimators


def test_logistic_elasticnet_grid_includes_l1_ratio() -> None:
    pipe = logistic_pipeline(_CONFIG, seed=42, penalty="elasticnet")
    grid = logistic_grid(penalty="elasticnet")
    assert "lr__l1_ratio" in grid
    assert set(grid.keys()) <= set(pipe.get_params())


@pytest.mark.parametrize("penalty", ["typo", "L2", "ridge"])
def test_logistic_pipeline_refuses_an_unknown_penalty(penalty: str) -> None:
    # On scikit-learn >= 1.8 the penalty is translated rather than passed through, so an
    # unrecognised value used to fall past every branch and fit a default L2 model in silence.
    with pytest.raises(ValueError, match="penalty must be one of"):
        logistic_pipeline(_CONFIG, seed=0, penalty=penalty)
