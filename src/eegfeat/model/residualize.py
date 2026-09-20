from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegfeat.model import _deps as _deps
from eegfeat.model.transformers import PreprocessingConfig, validate_subject_missingness

__all__ = [
    "FoldNuisanceFit",
    "StagedResidualPreprocessor",
    "fit_nuisance_model",
    "fit_staged_residual_preprocessor",
    "reconstruct_staged_permutation_target_for_fold",
    "residualize_targets",
]


@dataclass(frozen=True)
class FoldNuisanceFit:
    train_target: npt.NDArray[np.float64]
    test_target: npt.NDArray[np.float64]
    train_prediction: npt.NDArray[np.float64]
    test_prediction: npt.NDArray[np.float64]
    train_residual: npt.NDArray[np.float64]
    test_residual: npt.NDArray[np.float64]
    details: Mapping[str, object]


def _validate_indices(
    y: npt.NDArray[np.float64],
    train_idx: npt.NDArray[np.intp],
    test_idx: npt.NDArray[np.intp],
) -> None:
    if y.ndim != 1:
        msg = f"Target residualization expects a 1D target vector, got shape {y.shape}."
        raise ValueError(msg)
    if train_idx.size == 0:
        msg = "Target residualization requires non-empty train indices."
        raise ValueError(msg)
    max_index = len(y) - 1
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        msg = "Target residualization indices must be non-negative."
        raise ValueError(msg)
    if np.any(train_idx > max_index) or (test_idx.size > 0 and np.any(test_idx > max_index)):
        msg = "Target residualization indices exceed target length."
        raise ValueError(msg)
    if test_idx.size > 0 and np.intersect1d(train_idx, test_idx).size > 0:
        msg = "Train and test indices must not overlap."
        raise ValueError(msg)


def _validate_training_nuisance_rank(
    nuisance_design: npt.NDArray[np.float64],
    columns: Sequence[str],
    *,
    tolerance: float = 1e-10,
) -> None:
    centered = nuisance_design - np.mean(nuisance_design, axis=0, keepdims=True)
    column_norms = np.linalg.norm(centered, axis=0)
    # Centring a constant with no exact binary form leaves rounding residue, not zeros, so
    # a column is judged constant relative to its own magnitude.
    column_scales = np.max(np.abs(nuisance_design), axis=0) * np.sqrt(nuisance_design.shape[0])
    if np.any(column_norms <= float(tolerance) * column_scales):
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)} contain constant training-fold nuisance terms."
        )
        raise ValueError(msg)

    scaled = centered / column_norms
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    if singular_values.size < len(columns):
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={singular_values.size}, parameters={len(columns)}."
        )
        raise ValueError(msg)

    max_singular_value = float(singular_values[0])
    if max_singular_value <= 0.0:
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)} have zero training-fold variance."
        )
        raise ValueError(msg)
    singular_ratios = singular_values / max_singular_value
    if np.any(singular_ratios < float(tolerance)):
        rank = int(np.sum(singular_ratios >= float(tolerance)))
        msg = (
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={rank}, parameters={len(columns)}, "
            f"tolerance={float(tolerance)}."
        )
        raise ValueError(msg)


def _design_matrix(
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp],
    columns: Sequence[str],
    *,
    check_rank: bool,
) -> npt.NDArray[np.float64]:
    if isinstance(covariates, pd.DataFrame):
        missing = [c for c in columns if c not in covariates.columns]
        if missing:
            msg = f"Target residualization nuisance columns are missing: {missing}."
            raise ValueError(msg)
        sub = covariates.iloc[rows]
        design_columns: list[npt.NDArray[np.float64]] = [np.ones(len(rows), dtype=np.float64)]
        for col in columns:
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(values)):
                msg = f"Target residualization column '{col}' contains non-finite values."
                raise ValueError(msg)
            design_columns.append(values)
        design = np.column_stack(design_columns)
    else:
        cov_arr = np.asarray(covariates, dtype=np.float64)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr.reshape(-1, 1)
        if cov_arr.shape[1] != len(columns):
            msg = (
                f"Covariates width ({cov_arr.shape[1]}) does not match "
                f"columns count ({len(columns)})."
            )
            raise ValueError(msg)
        sub_arr = cov_arr[rows]
        if not np.all(np.isfinite(sub_arr)):
            msg = "Target residualization covariates contain non-finite values."
            raise ValueError(msg)
        design = np.column_stack([np.ones(len(rows), dtype=np.float64), sub_arr])

    if check_rank:
        _validate_training_nuisance_rank(design[:, 1:], columns)
    return design


def _fit_coefficients(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # Solve in comparable units so lstsq's relative rank cutoff cannot erase
    # a nuisance term solely because its measurement units are small.
    scales = np.linalg.norm(design, axis=0)
    coefficients, _, rank, _ = np.linalg.lstsq(design / scales, target, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Target residualization design is rank deficient after scaling.")
    return np.asarray(coefficients / (scales if target.ndim == 1 else scales[:, None]))


def fit_nuisance_model(
    y: npt.NDArray[np.float64] | Sequence[float],
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    train: npt.NDArray[np.intp] | Sequence[int],
    test: npt.NDArray[np.intp] | Sequence[int] | None = None,
    *,
    columns: Sequence[str],
) -> FoldNuisanceFit:
    column_names = tuple(str(c).strip() for c in columns if str(c).strip())
    if not column_names:
        msg = "Target residualization requires at least one nuisance column."
        raise ValueError(msg)

    y_values = np.asarray(y, dtype=np.float64)
    train_indices = np.asarray(train, dtype=np.intp)
    test_indices = (
        np.asarray(test, dtype=np.intp) if test is not None else np.empty(0, dtype=np.intp)
    )
    _validate_indices(y_values, train_indices, test_indices)

    design_train = _design_matrix(covariates, train_indices, column_names, check_rank=True)
    y_train = y_values[train_indices]
    if not np.all(np.isfinite(y_train)):
        msg = "Target residualization requires finite train target values."
        raise ValueError(msg)
    if len(y_train) <= design_train.shape[1]:
        msg = (
            "Target residualization requires more training rows than nuisance parameters: "
            f"rows={len(y_train)}, parameters={design_train.shape[1]}."
        )
        raise ValueError(msg)

    coefficients = _fit_coefficients(design_train, y_train)
    train_prediction = design_train @ coefficients
    train_residual = y_train - train_prediction

    if test_indices.size > 0:
        design_test = _design_matrix(covariates, test_indices, column_names, check_rank=False)
        y_test = y_values[test_indices]
        if not np.all(np.isfinite(y_test)):
            msg = "Target residualization requires finite test target values."
            raise ValueError(msg)
        test_prediction = design_test @ coefficients
        test_residual = y_test - test_prediction
    else:
        y_test = np.empty(0, dtype=np.float64)
        test_prediction = np.empty(0, dtype=np.float64)
        test_residual = np.empty(0, dtype=np.float64)

    details: dict[str, object] = {
        "columns": list(column_names),
        "n_parameters": int(design_train.shape[1]),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
    }
    return FoldNuisanceFit(
        train_target=y_train,
        test_target=y_test,
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        train_residual=train_residual,
        test_residual=test_residual,
        details=details,
    )


def residualize_targets(
    y: npt.NDArray[np.float64] | Sequence[float],
    covariates: pd.DataFrame | npt.NDArray[np.float64],
    train: npt.NDArray[np.intp] | Sequence[int],
    test: npt.NDArray[np.intp] | Sequence[int],
    *,
    columns: Sequence[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    fit = fit_nuisance_model(y, covariates, train, test, columns=columns)
    return fit.train_residual, fit.test_residual


def _finite_feature_block(
    X: npt.NDArray[np.float64], rows: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    values = np.asarray(X, dtype=np.float64)[rows].copy()
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass(frozen=True)
class StagedResidualPreprocessor:
    columns: tuple[str, ...]
    feature_support: npt.NDArray[np.bool_]
    feature_medians: npt.NDArray[np.float64]
    feature_coefficients: npt.NDArray[np.float64]
    target_coefficients: npt.NDArray[np.float64]
    power_transform: object
    n_fit_rows: int
    max_subject_missingness: float

    def _design(
        self,
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
        *,
        check_rank: bool,
    ) -> npt.NDArray[np.float64]:
        return _design_matrix(covariates, rows, self.columns, check_rank=check_rank)

    def transform_features(
        self,
        X: npt.NDArray[np.float64],
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
        groups: Sequence[object] | npt.NDArray[Any],
    ) -> npt.NDArray[np.float64]:
        values = _finite_feature_block(X, rows)[:, self.feature_support]
        validate_subject_missingness(
            values,
            cast(npt.NDArray[np.object_], np.asarray(groups)[rows]),
            maximum=self.max_subject_missingness,
        )
        filled = np.where(np.isnan(values), self.feature_medians[None, :], values)
        design = self._design(covariates, rows, check_rank=False)
        return filled - design @ self.feature_coefficients

    def nuisance_prediction(
        self,
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        design = self._design(covariates, rows, check_rank=False)
        return design @ self.target_coefficients

    def transform_target(
        self,
        y: npt.NDArray[np.float64],
        covariates: pd.DataFrame | npt.NDArray[np.float64],
        rows: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        pred = self.nuisance_prediction(covariates, rows)
        residual = np.asarray(y, dtype=np.float64)[rows] - pred
        from sklearn.preprocessing import PowerTransformer

        pt = cast(PowerTransformer, self.power_transform)
        transformed = pt.transform(residual.reshape(-1, 1)).flatten()
        return cast(npt.NDArray[np.float64], transformed)

    def inverse_transform_target(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        from sklearn.preprocessing import PowerTransformer

        pt = cast(PowerTransformer, self.power_transform)
        inversed = pt.inverse_transform(
            np.asarray(values, dtype=np.float64).reshape(-1, 1)
        ).flatten()
        return cast(npt.NDArray[np.float64], inversed)


def fit_staged_residual_preprocessor(
    *,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    covariates: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    meta: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    groups: Sequence[object] | npt.NDArray[Any],
    rows: npt.NDArray[np.intp],
    columns: Sequence[str],
    config: PreprocessingConfig | None = None,
) -> StagedResidualPreprocessor:
    from sklearn.preprocessing import PowerTransformer

    cov = covariates if covariates is not None else meta
    if cov is None:
        msg = "fit_staged_residual_preprocessor requires 'covariates' or 'meta'."
        raise ValueError(msg)

    cfg = config or PreprocessingConfig()
    fit_rows = np.asarray(rows, dtype=np.intp)
    values = _finite_feature_block(X, fit_rows)

    max_missing = cfg.max_feature_missingness
    missing_rate = (
        np.isnan(values).sum(axis=0) / values.shape[0]
        if values.shape[0]
        else np.ones(values.shape[1])
    )
    support = missing_rate <= max_missing
    if not np.any(support):
        msg = (
            f"Every feature exceeds the {max_missing:.1%} missingness limit on this "
            "training split; nothing is left to residualize."
        )
        raise ValueError(msg)

    kept = values[:, support]
    max_subject_missingness = cfg.max_subject_missingness
    validate_subject_missingness(
        kept,
        cast(npt.NDArray[np.object_], np.asarray(groups)[fit_rows]),
        maximum=max_subject_missingness,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        medians = np.nanmedian(kept, axis=0)
    if not np.all(np.isfinite(medians)):
        msg = (
            "A feature retained by the missingness policy has no finite training value "
            "to impute from."
        )
        raise ValueError(msg)
    imputed = np.where(np.isnan(kept), medians[None, :], kept)

    design = _design_matrix(cov, fit_rows, tuple(columns), check_rank=True)
    feature_coefficients = _fit_coefficients(design, imputed)

    y_fit = np.asarray(y, dtype=np.float64)[fit_rows]
    if not np.all(np.isfinite(y_fit)):
        msg = "Staged residual learning requires finite training target values."
        raise ValueError(msg)
    if y_fit.size <= design.shape[1]:
        msg = (
            "Staged residual learning requires more training rows than nuisance parameters: "
            f"rows={y_fit.size}, parameters={design.shape[1]}."
        )
        raise ValueError(msg)
    target_coefficients = _fit_coefficients(design, y_fit)

    power_transform = PowerTransformer(method="yeo-johnson", standardize=True)
    power_transform.fit((y_fit - design @ target_coefficients).reshape(-1, 1))

    return StagedResidualPreprocessor(
        columns=tuple(columns),
        feature_support=support,
        feature_medians=medians,
        feature_coefficients=feature_coefficients,
        target_coefficients=target_coefficients,
        power_transform=power_transform,
        n_fit_rows=int(fit_rows.size),
        max_subject_missingness=max_subject_missingness,
    )


def reconstruct_staged_permutation_target_for_fold(
    *,
    y: npt.NDArray[np.float64],
    covariates: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    meta: pd.DataFrame | npt.NDArray[np.float64] | None = None,
    train: npt.NDArray[np.intp] | None = None,
    test: npt.NDArray[np.intp] | None = None,
    train_idx: npt.NDArray[np.intp] | None = None,
    test_idx: npt.NDArray[np.intp] | None = None,
    columns: Sequence[str],
    permutation_indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    cov = covariates if covariates is not None else meta
    if cov is None:
        msg = "reconstruct_staged_permutation_target_for_fold requires 'covariates' or 'meta'."
        raise ValueError(msg)
    trn = train if train is not None else train_idx
    tst = test if test is not None else test_idx
    if trn is None or tst is None:
        msg = "reconstruct_staged_permutation_target_for_fold requires train and test indices."
        raise ValueError(msg)

    y_arr = np.asarray(y, dtype=np.float64)
    train_indices = np.asarray(trn, dtype=np.intp)
    test_indices = np.asarray(tst, dtype=np.intp)
    fold_mask = np.zeros(len(y_arr), dtype=np.bool_)
    fold_mask[train_indices] = True
    if np.any(fold_mask[test_indices]):
        msg = "Outer fold train and test indices must be disjoint."
        raise ValueError(msg)
    fold_mask[test_indices] = True
    fold_indices = np.flatnonzero(fold_mask)

    nuisance_fit = fit_nuisance_model(
        y=y_arr,
        covariates=cov,
        train=train_indices,
        test=test_indices,
        columns=columns,
    )
    nuisance_prediction = np.full(len(y_arr), np.nan, dtype=np.float64)
    residual = np.full(len(y_arr), np.nan, dtype=np.float64)
    nuisance_prediction[train_indices] = nuisance_fit.train_prediction
    nuisance_prediction[test_indices] = nuisance_fit.test_prediction
    residual[train_indices] = nuisance_fit.train_residual
    residual[test_indices] = nuisance_fit.test_residual

    residual_fold = residual[fold_indices]
    nuisance_prediction_fold = nuisance_prediction[fold_indices]
    if not np.all(np.isfinite(residual_fold)):
        msg = "Staged residual permutation requires finite fold residuals."
        raise ValueError(msg)
    if not np.all(np.isfinite(nuisance_prediction_fold)):
        msg = "Staged residual permutation requires finite nuisance predictions."
        raise ValueError(msg)

    source_indices = np.asarray(permutation_indices, dtype=np.intp)
    if source_indices.shape != (len(y_arr),):
        msg = "Permutation indices must contain one source row per target row."
        raise ValueError(msg)
    if not np.array_equal(np.sort(source_indices), np.arange(len(y_arr), dtype=np.intp)):
        msg = "Permutation indices must be a permutation of all row indices."
        raise ValueError(msg)
    fold_source_indices = source_indices[fold_indices]
    if not np.all(fold_mask[fold_source_indices]):
        msg = "Permutation indices map an outer-fold row outside that fold."
        raise ValueError(msg)
    shifted_residual_fold = residual[fold_source_indices]

    y_perm = y_arr.copy()
    y_perm[fold_indices] = nuisance_prediction_fold + shifted_residual_fold
    return y_perm
