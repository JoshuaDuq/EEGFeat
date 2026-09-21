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
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2)
    predictions = cross_fit_regression(
        folds, X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0, runs=RUNS
    )
    assert all(p.subject is not None for p in predictions)


def test_within_subject_folds_cannot_be_grouped_by_subject() -> None:
    # Caught at the entry point, where the message can name the real problem, rather than
    # surfacing as "at least 2 groups" from inside tuning.
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2)
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


def test_target_residualization_is_refitted_inside_inner_cv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eegfeat.model.crossfit as module

    rng = np.random.default_rng(42)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 8).astype(object)
    x = rng.normal(size=(32, 3))
    y = rng.normal(size=32)
    covariates = rng.normal(size=(32, 1))

    folds = loso_folds(groups)
    training_sizes: list[int] = []

    original = module.residualize_targets

    def record_fit(y_all, cov_all, train, test, *, columns):
        training_sizes.append(len(train))
        return original(
            y_all,
            cov_all,
            train,
            test,
            columns=columns,
        )

    monkeypatch.setattr(module, "residualize_targets", record_fit)

    module.cross_fit_regression(
        folds,
        x,
        y,
        groups,
        PIPE,
        GRID,
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=42,
        covariates=covariates,
        residualize_on=("nuisance",),
        scoring="neg_mean_squared_error",
    )

    outer_training_size = len(folds[0].train)

    assert training_sizes
    assert any(
        size < outer_training_size for size in training_sizes
    ), "Nuisance fitting never occurred inside the inner CV folds."


def test_inner_validation_targets_do_not_leak_into_nuisance_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eegfeat.model.crossfit as module

    rng = np.random.default_rng(42)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 8).astype(object)
    x = rng.normal(size=(32, 3))
    y1 = rng.normal(size=32)
    y2 = y1.copy()
    y2[groups == "s2"] += 500.0
    covariates = rng.normal(size=(32, 1))

    folds = [loso_folds(groups)[0]]  # fold where s1 is test, s2/s3/s4 are train

    recorded_inner_fits_y1: dict[tuple[int, ...], np.ndarray] = {}
    recorded_inner_fits_y2: dict[tuple[int, ...], np.ndarray] = {}

    original = module.residualize_targets

    def record_fit(store: dict[tuple[int, ...], np.ndarray]):
        def fit_wrapper(y_all, cov_all, train, test, *, columns):
            res_tr, res_te = original(y_all, cov_all, train, test, columns=columns)
            store[tuple(int(i) for i in train)] = res_tr.copy()
            return res_tr, res_te

        return fit_wrapper

    monkeypatch.setattr(module, "residualize_targets", record_fit(recorded_inner_fits_y1))
    module.cross_fit_regression(
        folds,
        x,
        y1,
        groups,
        PIPE,
        GRID,
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=42,
        covariates=covariates,
        residualize_on=("nuisance",),
        scoring="neg_mean_squared_error",
    )

    monkeypatch.setattr(module, "residualize_targets", record_fit(recorded_inner_fits_y2))
    module.cross_fit_regression(
        folds,
        x,
        y2,
        groups,
        PIPE,
        GRID,
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=42,
        covariates=covariates,
        residualize_on=("nuisance",),
        scoring="neg_mean_squared_error",
    )

    s2_indices = set(np.flatnonzero(groups == "s2"))
    # In inner CV splits where s2 was NOT in train (i.e. s2 was in validation),
    # the training residuals must be bit-for-bit identical between y1 and y2
    matches = [
        train_idx for train_idx in recorded_inner_fits_y1 if not s2_indices.intersection(train_idx)
    ]
    assert matches, "Expected at least one inner split where s2 was held out in validation"
    for train_idx in matches:
        np.testing.assert_array_equal(
            recorded_inner_fits_y1[train_idx],
            recorded_inner_fits_y2[train_idx],
        )


def test_covariates_that_still_carry_rejected_epochs_are_refused() -> None:
    # Covariates are indexed positionally, so a frame that kept the rejected epochs would
    # residualize each trial against another trial's nuisance values without any error.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(4)], 10).astype(object)
    values = rng.normal(size=(40, 3))
    target = rng.normal(size=40)

    with pytest.raises(ValueError, match="covariates has 60 rows and X has 40"):
        cross_fit_regression(
            loso_folds(groups),
            values,
            target,
            groups,
            PIPE,
            {},
            inner=BY_SUBJECT,
            seed=0,
            covariates=rng.normal(size=(60, 1)),
            residualize_on=["age"],
        )


def test_run_grouping_under_cross_subject_folds_is_refused() -> None:
    # Run labels are shared across subjects, so a run-grouped inner split of a LOSO
    # training set keeps every subject on both sides and tunes for the wrong task.
    with pytest.raises(ValueError, match="cross-subject folds cannot be grouped by run"):
        cross_fit_regression(
            loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0, runs=RUNS
        )
