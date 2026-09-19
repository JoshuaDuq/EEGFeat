from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_regression
from eegfeat.model.estimators import ridge_pipeline
from eegfeat.model.splits import InnerSplit, loso_folds, within_subject_folds
from eegfeat.model.transformers import PreprocessingConfig

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean", "median"]}
GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 8).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 4).astype(object)
X = np.arange(GROUPS.size, dtype=float).reshape(-1, 1)
Y = np.arange(GROUPS.size, dtype=float)
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)
BY_RUN = InnerSplit(grouping="run", n_splits=2)


def test_every_trial_is_predicted_exactly_once_across_folds() -> None:
    predictions = cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0
    )
    rows = np.concatenate([p.rows for p in predictions])
    np.testing.assert_array_equal(np.sort(rows), np.arange(GROUPS.size))


def test_no_fold_is_fitted_on_the_subject_it_predicts() -> None:
    for prediction in cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0
    ):
        assert len(set(GROUPS[prediction.rows])) == 1


def test_the_same_loop_serves_within_subject_folds_grouped_on_runs() -> None:
    # The only things that change between designs are the folds and the inner grouping.
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2, seed=0)
    predictions = cross_fit_regression(
        folds, X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0, runs=RUNS
    )
    assert all(p.subject is not None for p in predictions)


def test_within_subject_folds_cannot_be_grouped_by_subject() -> None:
    # Caught at the entry point, where the message can name the real problem, rather than
    # surfacing as "at least 2 groups" from inside tuning.
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2, seed=0)
    with pytest.raises(ValueError, match="within-subject"):
        cross_fit_regression(folds, X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0)


def test_run_grouping_without_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="runs"):
        cross_fit_regression(loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0)


def test_a_failing_fold_raises_rather_than_being_dropped() -> None:
    # Safeguard 6: a fold that cannot be fitted is a fault, not a measurement. Dropping it
    # would silently compute the cohort result on a subset nobody chose.
    with pytest.raises(ValueError):
        cross_fit_regression(
            loso_folds(GROUPS),
            X,
            Y,
            GROUPS,
            PIPE,
            {"regressor__nonexistent": [1]},
            inner=BY_SUBJECT,
            seed=0,
        )


def test_results_are_ordered_by_fold_not_by_completion() -> None:
    predictions = cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0, outer_n_jobs=2
    )
    assert [p.fold for p in predictions] == sorted(p.fold for p in predictions)


def test_cross_fit_applies_feature_harmonization() -> None:
    X_mod = np.column_stack([X, np.zeros(GROUPS.size)])
    X_mod[GROUPS != "s1", -1] = 1.0
    predictions = cross_fit_regression(
        loso_folds(GROUPS),
        X_mod,
        Y,
        GROUPS,
        PIPE,
        GRID,
        inner=BY_SUBJECT,
        seed=0,
        harmonization="intersection",
    )
    assert len(predictions) == 4


def test_cross_fit_applies_target_residualization() -> None:
    covariates = Y.reshape(-1, 1)
    predictions = cross_fit_regression(
        loso_folds(GROUPS),
        X,
        Y,
        GROUPS,
        PIPE,
        GRID,
        inner=BY_SUBJECT,
        seed=0,
        covariates=covariates,
        residualize_on=["c1"],
    )
    assert len(predictions) == 4
    # Residualized target has nuisance subtracted, so it must not equal raw Y
    raw_y_fold0 = Y[predictions[0].rows]
    assert not np.allclose(predictions[0].y_true, raw_y_fold0)
    assert np.allclose(predictions[0].y_true, 0.0, atol=1e-10)


def test_cross_fit_regression_rejects_residualize_on_without_covariates() -> None:
    with pytest.raises(ValueError, match="residualize_on, but covariates is None"):
        cross_fit_regression(
            loso_folds(GROUPS),
            X,
            Y,
            GROUPS,
            PIPE,
            GRID,
            inner=BY_SUBJECT,
            seed=0,
            covariates=None,
            residualize_on=["c1"],
        )


@pytest.mark.parametrize("n_covariates", [0, 1])
def test_a_training_subject_above_the_missingness_limit_fails_the_fold(
    n_covariates: int,
) -> None:
    # max_subject_missingness belongs to the pipeline's missingness step, but pipelines never
    # route groups to their steps, so the limit has to be checked on the fitted model.
    rng = np.random.default_rng(0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    values = rng.normal(size=(40, 10 + n_covariates))
    # Where s1 trains, each of these features is 20% missing (kept), but 36% of s1's own
    # feature values are missing.
    values[:6, :6] = np.nan
    pipe = ridge_pipeline(
        PreprocessingConfig(max_subject_missingness=0.3), seed=0, n_covariates=n_covariates
    )
    with pytest.raises(ValueError, match="s1"):
        cross_fit_regression(
            loso_folds(groups),
            values,
            rng.normal(size=40),
            groups,
            pipe,
            {},
            inner=BY_SUBJECT,
            seed=0,
        )


def test_custom_fold_with_overlap_is_rejected() -> None:
    from eegfeat.model.splits import Fold

    bad = Fold(
        index=1,
        train=np.array([0, 1, 2]),
        test=np.array([2, 3]),
    )

    with pytest.raises(ValueError, match="train/test overlap"):
        cross_fit_regression(
            [bad],
            X,
            Y,
            GROUPS,
            PIPE,
            {},
            inner=BY_SUBJECT,
            seed=0,
        )

