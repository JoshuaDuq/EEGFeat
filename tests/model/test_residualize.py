from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegfeat.model.residualize import (
    FoldNuisanceFit,
    fit_nuisance_model,
    fit_staged_residual_preprocessor,
    reconstruct_staged_permutation_target_for_fold,
    residualize_targets,
)
from eegfeat.model.transformers import PreprocessingConfig

TRAIN = np.arange(8, dtype=np.intp)
TEST = np.arange(8, 12, dtype=np.intp)


def test_the_nuisance_model_is_fitted_on_training_rows_only() -> None:
    # If the test rows entered the fit, the test residuals would be centred by
    # construction and the held-out score would be optimistic.
    y = np.concatenate([np.arange(8, dtype=float), np.full(4, 100.0)])
    covariates = np.concatenate([np.arange(8, dtype=float), np.zeros(4)]).reshape(-1, 1)
    _, y_test = residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])
    assert np.all(y_test > 50.0)


def test_a_covariate_that_explains_the_target_leaves_near_zero_residuals() -> None:
    y = np.arange(12, dtype=float)
    covariates = np.arange(12, dtype=float).reshape(-1, 1)
    y_train, _ = residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])
    assert np.allclose(y_train, 0.0, atol=1e-9)


def test_a_rank_deficient_training_design_is_refused() -> None:
    # A constant covariate adds nothing to the intercept; silently inverting a singular
    # design would return residuals that depend on the pseudo-inverse's tie-breaking.
    y = np.arange(12, dtype=float)
    covariates = np.ones((12, 1))
    with pytest.raises(ValueError, match="rank deficient"):
        residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])


def test_train_and_test_indices_may_not_overlap() -> None:
    y = np.arange(12, dtype=float)
    covariates = np.arange(12, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="overlap"):
        residualize_targets(y, covariates, TRAIN, TRAIN, columns=["c"])


def test_fit_nuisance_model_accepts_dataframe_and_populates_fit() -> None:
    y = np.asarray([100.0, 110.0, 0.0, 10.0, 0.0, 10.0], dtype=float)
    meta = pd.DataFrame(
        {
            "nuisance": [0, 1, 0, 1, 0, 1],
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
        }
    )
    train_idx = np.asarray([2, 3, 4, 5], dtype=np.intp)
    test_idx = np.asarray([0, 1], dtype=np.intp)
    fit = fit_nuisance_model(y, meta, train_idx, test_idx, columns=["nuisance"])
    assert isinstance(fit, FoldNuisanceFit)
    assert np.allclose(fit.train_residual, 0.0)
    assert np.allclose(fit.test_residual, 100.0)
    assert fit.details["columns"] == ["nuisance"]
    assert fit.details["n_train"] == 4
    assert fit.details["n_test"] == 2


def test_residualize_targets_checks_rank_only_on_training_design() -> None:
    y = np.asarray([0.0, 1.0, 2.0, 3.0, 10.0], dtype=float)
    meta = pd.DataFrame({"nuisance": [0.0, 1.0, 2.0, 3.0, 1.0]})
    y_train, y_test = residualize_targets(
        y,
        meta,
        np.asarray([0, 1, 2, 3], dtype=np.intp),
        np.asarray([4], dtype=np.intp),
        columns=["nuisance"],
    )
    assert np.allclose(y_train, 0.0)
    assert y_test.shape == (1,)


def test_missing_column_raises_error() -> None:
    meta = pd.DataFrame({"nuisance": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing"):
        residualize_targets(
            np.arange(3.0),
            meta,
            np.asarray([0, 1], dtype=np.intp),
            np.asarray([2], dtype=np.intp),
            columns=["unknown"],
        )


def test_reconstruct_staged_permutation_target_shifts_residuals_within_fold() -> None:
    y = np.asarray([10.0, 25.0, 30.0, 40.0], dtype=float)
    covariates = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
    train_idx = np.asarray([0, 1, 2], dtype=np.intp)
    test_idx = np.asarray([3], dtype=np.intp)
    perm_indices = np.asarray([1, 2, 0, 3], dtype=np.intp)
    y_perm = reconstruct_staged_permutation_target_for_fold(
        y=y,
        covariates=covariates,
        train=train_idx,
        test=test_idx,
        columns=["c"],
        permutation_indices=perm_indices,
    )
    assert y_perm.shape == y.shape
    assert not np.allclose(y_perm, y)
    np.testing.assert_allclose(y_perm, [15.0, 20.0, 30.0, 40.0])


def test_staged_preprocessor_transforms_features_and_target() -> None:
    rng = np.random.default_rng(42)
    n = 30
    nuisance = rng.normal(size=n)
    X = np.column_stack([nuisance * 3.0, rng.normal(size=n)])
    y = 5.0 + 2.0 * nuisance + rng.normal(scale=0.1, size=n)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 10)
    meta = pd.DataFrame({"nuisance": nuisance})
    train_rows = np.arange(20, dtype=np.intp)
    test_rows = np.arange(20, 30, dtype=np.intp)

    preprocessor = fit_staged_residual_preprocessor(
        X=X,
        y=y,
        covariates=meta,
        groups=groups,
        rows=train_rows,
        columns=["nuisance"],
    )
    X_train_res = preprocessor.transform_features(X, meta, train_rows, groups)
    assert X_train_res.shape == (20, 2)
    assert not np.allclose(X_train_res, X[train_rows])
    np.testing.assert_allclose(X_train_res[:, 0], 0.0, atol=1e-10)

    y_test_trans = preprocessor.transform_target(y, meta, test_rows)
    assert y_test_trans.shape == (10,)
    y_test_inv = preprocessor.inverse_transform_target(y_test_trans)
    assert y_test_inv.shape == (10,)


def test_staged_imputation_rejects_excessive_subject_missingness() -> None:
    n_subjects = 30
    n_trials = 10
    n = n_subjects * n_trials
    X = np.ones((n, 2))
    groups = np.repeat([f"sub-{i:02d}" for i in range(n_subjects)], n_trials)
    # One subject with missing EEG is 10/300 = 3.3% feature missingness (under 5% limit),
    # but 100% subject missingness (exceeds 10% limit).
    X[groups == "sub-00"] = np.nan
    y = np.ones(n)
    meta = pd.DataFrame({"nuisance": np.arange(n, dtype=float)})
    train_rows = np.arange(n, dtype=np.intp)

    with pytest.raises(ValueError, match="Subject sub-00 has missingness"):
        fit_staged_residual_preprocessor(
            X=X,
            y=y,
            covariates=meta,
            groups=groups,
            rows=train_rows,
            columns=["nuisance"],
            config=PreprocessingConfig(max_subject_missingness=0.10),
        )


def test_a_constant_covariate_is_refused_even_when_its_value_is_inexact_in_binary() -> None:
    # 0.1 has no exact binary form, so centring twelve copies leaves rounding residue
    # rather than zeros. The rank check must still see a constant column instead of
    # rescaling the residue into one that looks informative.
    y = np.arange(16, dtype=float)
    covariates = np.full((16, 1), 0.1)
    with pytest.raises(ValueError, match="rank deficient"):
        residualize_targets(
            y,
            covariates,
            np.arange(12, dtype=np.intp),
            np.arange(12, 16, dtype=np.intp),
            columns=["c"],
        )
