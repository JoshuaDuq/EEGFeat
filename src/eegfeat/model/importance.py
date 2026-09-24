from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import is_classifier
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.pipeline import Pipeline

from eegfeat._validation import blank_non_finite
from eegfeat.model._deps import require_shap
from eegfeat.model.aggregate import _SubjectRScorer
from eegfeat.model.crossfit import (
    _default_scoring,
    _fit_fold,
    _FittedFold,
    _validate_and_resolve_inner_groups,
    _validate_outer_folds,
    _validate_residualize_within,
    _validate_row_aligned,
)
from eegfeat.model.execution import run_folds
from eegfeat.model.splits import Fold, InnerSplit
from eegfeat.model.transformers import transform_feature_names
from eegfeat.table import FeatureMeta

__all__ = [
    "Importance",
    "aggregate_by",
    "permutation_importance",
    "permutation_importance_over_folds",
    "shap_importance",
    "shap_importance_over_folds",
]


@dataclass(frozen=True)
class Importance:
    feature_names: tuple[str, ...]
    values: npt.NDArray[np.float64]
    per_fold: npt.NDArray[np.float64]


def aggregate_by(
    importance: Importance,
    meta: Sequence[FeatureMeta],
    field: str,
) -> dict[str, float]:
    valid_fields = {f.name for f in fields(FeatureMeta)}
    if field not in valid_fields:
        raise ValueError(f"Unknown FeatureMeta field: {field!r}")

    names = tuple(importance.feature_names)
    if len(names) != len(set(names)):
        raise ValueError("Importance contains duplicate feature names.")

    metadata_by_name = {m.name: m for m in meta}
    if len(metadata_by_name) != len(meta):
        raise ValueError("Metadata contains duplicate feature names.")

    if set(names) != set(metadata_by_name):
        raise ValueError("Importance and metadata must contain exactly the same feature names.")

    # A feature no fold could score carries NaN, which would silently turn its whole
    # category's total into NaN. Name the features instead of returning that.
    unscored = [
        name for name, value in zip(names, importance.values, strict=True) if np.isnan(value)
    ]
    if unscored:
        msg = (
            f"No fold scored {unscored[:3]}, so their importance is undefined and would make "
            "the totals NaN; drop them or lower the harmonization threshold that removed them."
        )
        raise ValueError(msg)

    totals: dict[str, float] = {}
    for name, value in zip(names, importance.values, strict=True):
        m = metadata_by_name[name]
        raw = getattr(m, field)
        key = "unknown" if raw is None else str(raw.name) if hasattr(raw, "name") else str(raw)
        totals[key] = totals.get(key, 0.0) + float(value)

    return totals


def permutation_importance(
    model: Pipeline,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    *,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: object = None,
) -> Importance:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    res = sklearn_perm_importance(
        model, X_arr, y_arr, scoring=scoring, n_repeats=n_repeats, random_state=seed
    )
    values = np.asarray(res.importances_mean, dtype=np.float64)
    if feature_names is None:
        names = tuple(f"feature_{i}" for i in range(values.size))
    else:
        names = tuple(feature_names)
    per_fold = np.empty((0, len(names)), dtype=np.float64)
    return Importance(feature_names=names, values=values, per_fold=per_fold)


def shap_importance(
    model: Pipeline,
    X: npt.NDArray[np.float64],
    feature_names: Sequence[str],
    *,
    seed: int = 42,
) -> Importance:
    require_shap()
    import shap

    X_arr = np.asarray(X, dtype=np.float64)
    steps = list(model.steps)
    _, final_estimator = steps[-1]
    X_trans = X_arr
    for _, step in steps[:-1]:
        if hasattr(step, "transform"):
            X_trans = step.transform(X_trans)
    # SHAP explains the columns the final estimator sees, which the preprocessing steps may
    # have dropped; each explained column is mapped back to the input feature it came from.
    explained = transform_feature_names(steps[:-1], feature_names)
    if len(explained) != X_trans.shape[1]:
        msg = (
            f"The pipeline outputs {X_trans.shape[1]} columns but reports {len(explained)} "
            "names, so SHAP values cannot be matched to features."
        )
        raise ValueError(msg)
    positions = _input_positions(explained, feature_names)

    rng = np.random.default_rng(seed)
    if hasattr(final_estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(X_trans)
    elif hasattr(final_estimator, "coef_"):
        explainer = shap.LinearExplainer(final_estimator, X_trans)
        shap_values = explainer.shap_values(X_trans)
    else:
        predict_fn = (
            final_estimator.predict_proba
            if hasattr(final_estimator, "predict_proba")
            else final_estimator.predict
        )
        bg_size = min(100, len(X_trans))
        bg_indices = rng.choice(len(X_trans), bg_size, replace=False)
        background = X_trans[bg_indices]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_trans, nsamples=100)

    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        # SHAP >= 0.45 places the output/class axis last.
        shap_values = (
            shap_values[:, :, 1]
            if is_classifier(final_estimator) and shap_values.shape[2] == 2
            else np.mean(np.abs(shap_values), axis=2)
        )
    if shap_values.shape != X_trans.shape:
        raise ValueError("SHAP values must align with the explained samples and features.")

    # A feature the pipeline dropped has no effect on the prediction, so its value is 0.
    names = tuple(feature_names)
    values = np.zeros(len(names), dtype=np.float64)
    values[positions] = np.mean(np.abs(shap_values), axis=0)
    per_fold = np.empty((0, len(names)), dtype=np.float64)
    return Importance(feature_names=names, values=values, per_fold=per_fold)


def _input_positions(
    explained: Sequence[str], feature_names: Sequence[str]
) -> npt.NDArray[np.intp]:
    index = {name: i for i, name in enumerate(feature_names)}
    if len(index) != len(feature_names):
        raise ValueError("feature_names must be unique to attribute SHAP values to them.")
    unknown = [name for name in explained if name not in index]
    if unknown:
        msg = (
            f"The pipeline outputs {unknown[:3]}, which are not input features, so their SHAP "
            "values belong to no single feature; remove steps that mix features, such as PCA."
        )
        raise ValueError(msg)
    return np.array([index[name] for name in explained], dtype=np.intp)


def _fold_fitter(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None,
    harmonization: str | None,
    covariates: npt.NDArray[np.float64] | None,
    residualize_on: Sequence[str],
    residualize_within: Literal["subject"] | None,
    scoring: object,
    refit: str | bool | None,
) -> Callable[[Fold], _FittedFold]:
    # Importance fits each fold exactly as cross-fitting does, so it explains the model that
    # was evaluated rather than one tuned or trained differently. That includes the fold
    # checks: importance is an entry point of its own, and folds reaching it never passed
    # through cross_fit_*.
    _validate_outer_folds(folds, len(X), groups, runs)
    _validate_row_aligned(len(X), y, covariates)
    _validate_residualize_within(residualize_within, residualize_on)
    inner_groups_all = _validate_and_resolve_inner_groups(folds, inner, groups, runs)
    task = _task(pipeline)
    scoring = _default_scoring(task, scoring)

    def fit(f: Fold) -> _FittedFold:
        return _fit_fold(
            task,
            f,
            X,
            y,
            groups,
            inner_groups_all,
            pipeline,
            grid,
            inner=inner,
            seed=seed,
            harmonization=harmonization,
            covariates=covariates,
            residualize_on=residualize_on,
            residualize_within=residualize_within,
            scoring=scoring,
            refit=refit,
        )

    return fit


def _task(pipeline: Pipeline) -> str:
    return "classification" if is_classifier(pipeline) else "regression"


def _importance_scoring(scoring: object, refit: str | bool | None) -> object:
    # Importance is the drop in the metric the model was selected on.
    if isinstance(scoring, Mapping):
        if isinstance(refit, str) and refit in scoring:
            return scoring[refit]
        msg = "With multi-metric scoring, refit must name the metric importance is measured on."
        raise ValueError(msg)
    return scoring


def _fold_values(
    values: npt.NDArray[np.float64], features: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    # A feature harmonized out of a fold was never available to that fold's model.
    fold_imp = np.full(features.size, np.nan, dtype=np.float64)
    fold_imp[features] = values
    return fold_imp


def _combine_folds(
    results: Sequence[npt.NDArray[np.float64]],
    n_folds: int,
    min_complete_fraction: float,
    label: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # run_folds propagates a failed fold, so a result here is always an array; a fold counts
    # as unsuccessful only when no feature survived it.
    successful = [r for r in results if np.any(np.isfinite(r))]
    rate = len(successful) / n_folds if n_folds else 0.0
    if rate < min_complete_fraction:
        msg = f"Insufficient successful folds for {label} ({len(successful)}/{n_folds})."
        raise ValueError(msg)

    per_fold = np.vstack(successful)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"Mean of empty slice")
        # The same reading of "missing" the fold filter above used: one feature's
        # non-finite value drops out of the average rather than becoming it.
        values = np.asarray(np.nanmean(blank_non_finite(per_fold), axis=0), dtype=np.float64)
    return per_fold, values


def shap_importance_over_folds(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    feature_names: Sequence[str],
    *,
    inner: InnerSplit,
    seed: int = 42,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    residualize_within: Literal["subject"] | None = None,
    scoring: object = None,
    refit: str | bool | None = None,
    min_complete_fraction: float = 0.5,
) -> Importance:
    require_shap()
    fit = _fold_fitter(
        folds,
        X,
        y,
        groups,
        pipeline,
        grid,
        inner=inner,
        seed=seed,
        runs=runs,
        harmonization=harmonization,
        covariates=covariates,
        residualize_on=residualize_on,
        residualize_within=residualize_within,
        scoring=scoring,
        refit=refit,
    )
    all_names = list(feature_names)

    def _fold_importance(f: Fold) -> npt.NDArray[np.float64]:
        fitted = fit(f)
        retained = [name for name, keep in zip(all_names, fitted.features, strict=True) if keep]
        imp = shap_importance(fitted.model, fitted.X_test, retained, seed=seed + f.index)
        return _fold_values(imp.values, fitted.features)

    results = run_folds(folds, _fold_importance, outer_n_jobs=outer_n_jobs)
    per_fold, values = _combine_folds(results, len(folds), min_complete_fraction, "SHAP")
    return Importance(feature_names=tuple(all_names), values=values, per_fold=per_fold)


def permutation_importance_over_folds(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    residualize_within: Literal["subject"] | None = None,
    scoring: object = None,
    refit: str | bool | None = None,
    min_complete_fraction: float = 0.5,
) -> Importance:
    importance_scoring = _importance_scoring(_default_scoring(_task(pipeline), scoring), refit)
    fit = _fold_fitter(
        folds,
        X,
        y,
        groups,
        pipeline,
        grid,
        inner=inner,
        seed=seed,
        runs=runs,
        harmonization=harmonization,
        covariates=covariates,
        residualize_on=residualize_on,
        residualize_within=residualize_within,
        scoring=scoring,
        refit=refit,
    )
    all_names = (
        list(feature_names)
        if feature_names is not None
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    def _fold_importance(f: Fold) -> npt.NDArray[np.float64]:
        fitted = fit(f)
        retained = [name for name, keep in zip(all_names, fitted.features, strict=True) if keep]
        fold_scoring = importance_scoring
        if isinstance(importance_scoring, _SubjectRScorer):
            # The permuted rows stay in test order, so each keeps its subject.
            test_groups = groups[f.test]

            def fold_scoring(
                estimator: object, X_: npt.NDArray[np.float64], y_: npt.NDArray[np.float64]
            ) -> float:
                return importance_scoring(estimator, X_, y_, test_groups)

        imp = permutation_importance(
            fitted.model,
            fitted.X_test,
            fitted.y_test,
            feature_names=retained,
            n_repeats=n_repeats,
            seed=seed + f.index,
            scoring=fold_scoring,
        )
        return _fold_values(imp.values, fitted.features)

    results = run_folds(folds, _fold_importance, outer_n_jobs=outer_n_jobs)
    per_fold, values = _combine_folds(
        results, len(folds), min_complete_fraction, "permutation importance"
    )
    return Importance(feature_names=tuple(all_names), values=values, per_fold=per_fold)
