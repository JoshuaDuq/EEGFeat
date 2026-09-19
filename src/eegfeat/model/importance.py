from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.pipeline import Pipeline

from eegfeat.model._deps import require_shap
from eegfeat.model.design import harmonize_fold
from eegfeat.model.execution import run_folds
from eegfeat.model.splits import Fold, InnerSplit
from eegfeat.model.tuning import fit_untuned, tune
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
    if len(importance.values) != len(meta):
        msg = (
            f"Length mismatch: {len(importance.values)} importance values "
            f"vs {len(meta)} metadata records."
        )
        raise ValueError(msg)
    totals: dict[str, float] = {}
    for val, m in zip(importance.values, meta, strict=True):
        raw_key = getattr(m, field, None)
        if raw_key is None:
            key = "unknown"
        elif hasattr(raw_key, "name"):
            key = str(raw_key.name)
        else:
            key = str(raw_key)
        totals[key] = totals.get(key, 0.0) + float(val)
    return totals


def permutation_importance(
    model: Pipeline,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    *,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
) -> Importance:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    res = sklearn_perm_importance(model, X_arr, y_arr, n_repeats=n_repeats, random_state=seed)
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
    final_name, final_estimator = steps[-1]
    X_trans = X_arr
    for _, step in steps[:-1]:
        if hasattr(step, "transform"):
            X_trans = step.transform(X_trans)

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

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif isinstance(shap_values, list):
        shap_values = np.mean(np.abs(np.stack(shap_values)), axis=0)

    values = np.mean(np.abs(shap_values), axis=0)
    names = tuple(feature_names)
    per_fold = np.empty((0, len(names)), dtype=np.float64)
    return Importance(
        feature_names=names, values=np.asarray(values, dtype=np.float64), per_fold=per_fold
    )


def _validate_inner_groups(
    inner: InnerSplit,
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None,
) -> npt.NDArray[np.object_]:
    if inner.grouping == "run":
        if runs is None:
            raise ValueError("Within-subject grouping on runs requires 'runs' array.")
        return runs
    return groups


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
    min_complete_fraction: float = 0.5,
) -> Importance:
    require_shap()
    inner_groups_all = _validate_inner_groups(inner, groups, runs)

    def _fold_importance(f: Fold) -> npt.NDArray[np.float64]:
        train_idx = f.train
        test_idx = f.test
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        if harmonization is not None:
            X_tr, X_te, _ = harmonize_fold(
                X_tr,
                X_te,
                groups[train_idx],
                mode=harmonization,
                n_covariates=0,
            )

        if grid:
            tuned = tune(
                pipeline,
                grid,
                X_tr,
                y_tr,
                inner_groups_all[train_idx],
                split=inner,
                seed=seed,
                fold=f.index,
            )
            model = tuned.estimator
        else:
            model = fit_untuned(pipeline, X_tr, y_tr, seed=seed, fold=f.index)

        imp = shap_importance(model, X_te, feature_names, seed=seed + f.index)
        return imp.values

    results: list[npt.NDArray[np.float64]] = run_folds(
        folds, _fold_importance, outer_n_jobs=outer_n_jobs
    )
    successful = [r for r in results if r is not None and np.all(np.isfinite(r))]
    rate = len(successful) / len(folds) if folds else 0.0
    if rate < min_complete_fraction:
        msg = f"Insufficient successful folds for SHAP ({len(successful)}/{len(folds)})."
        raise ValueError(msg)

    per_fold = np.vstack(successful)
    values = np.asarray(np.mean(per_fold, axis=0), dtype=np.float64)
    return Importance(feature_names=tuple(feature_names), values=values, per_fold=per_fold)


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
    min_complete_fraction: float = 0.5,
) -> Importance:
    inner_groups_all = _validate_inner_groups(inner, groups, runs)

    def _fold_importance(f: Fold) -> npt.NDArray[np.float64]:
        train_idx = f.train
        test_idx = f.test
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if harmonization is not None:
            X_tr, X_te, _ = harmonize_fold(
                X_tr,
                X_te,
                groups[train_idx],
                mode=harmonization,
                n_covariates=0,
            )

        if grid:
            tuned = tune(
                pipeline,
                grid,
                X_tr,
                y_tr,
                inner_groups_all[train_idx],
                split=inner,
                seed=seed,
                fold=f.index,
            )
            model = tuned.estimator
        else:
            model = fit_untuned(pipeline, X_tr, y_tr, seed=seed, fold=f.index)

        imp = permutation_importance(
            model,
            X_te,
            y_te,
            feature_names=feature_names,
            n_repeats=n_repeats,
            seed=seed + f.index,
        )
        return imp.values

    results: list[npt.NDArray[np.float64]] = run_folds(
        folds, _fold_importance, outer_n_jobs=outer_n_jobs
    )
    successful = [r for r in results if r is not None and np.all(np.isfinite(r))]
    rate = len(successful) / len(folds) if folds else 0.0
    if rate < min_complete_fraction:
        msg = (
            f"Insufficient successful folds for permutation importance "
            f"({len(successful)}/{len(folds)})."
        )
        raise ValueError(msg)

    per_fold = np.vstack(successful)
    values = np.asarray(np.mean(per_fold, axis=0), dtype=np.float64)
    names = (
        tuple(feature_names)
        if feature_names is not None
        else tuple(f"feature_{i}" for i in range(values.size))
    )
    return Importance(feature_names=names, values=values, per_fold=per_fold)
