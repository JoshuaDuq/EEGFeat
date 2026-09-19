from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from eegfeat.model import _deps as _deps
from eegfeat.model.design import harmonize_fold
from eegfeat.model.execution import run_folds
from eegfeat.model.residualize import residualize_targets
from eegfeat.model.splits import Fold, InnerSplit
from eegfeat.model.transformers import _check_subject_missingness
from eegfeat.model.tuning import fit_untuned, tune

__all__ = [
    "FoldClassification",
    "FoldPrediction",
    "cross_fit_classification",
    "cross_fit_regression",
]


@dataclass(frozen=True)
class FoldPrediction:
    fold: int
    subject: str | None
    rows: npt.NDArray[np.intp]
    y_true: npt.NDArray[np.float64]
    y_pred: npt.NDArray[np.float64]
    best_params: dict[str, object]


@dataclass(frozen=True)
class FoldClassification:
    fold: int
    subject: str | None
    rows: npt.NDArray[np.intp]
    y_true: npt.NDArray[np.intp]
    y_pred: npt.NDArray[np.intp]
    y_prob: npt.NDArray[np.float64] | None
    classes: tuple[int, ...]
    best_params: dict[str, object]


def _validate_and_resolve_inner_groups(
    folds: Sequence[Fold],
    inner: InnerSplit,
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None,
) -> npt.NDArray[np.object_]:
    if inner.grouping == "run" and runs is None:
        msg = "within-subject grouping on runs requires 'runs' array."
        raise ValueError(msg)
    if inner.grouping == "subject" and any(f.subject is not None for f in folds):
        msg = "within-subject folds cannot be grouped by subject (inner.grouping == 'subject')."
        raise ValueError(msg)
    inner_groups = runs if inner.grouping == "run" else groups
    if inner_groups is None:
        msg = "Inner grouping array is missing."
        raise ValueError(msg)
    return inner_groups


@dataclass(frozen=True)
class _FittedFold:
    model: Pipeline
    X_test: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64] | npt.NDArray[np.intp]
    y_test: npt.NDArray[np.float64] | npt.NDArray[np.intp]
    features: npt.NDArray[np.bool_]
    best_params: dict[str, object]


def _fit_fold(
    task: str,
    f: Fold,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    inner_groups_all: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    harmonization: str | None,
    covariates: npt.NDArray[np.float64] | None,
    residualize_on: Sequence[str],
    scoring: object,
    refit: str | bool | None,
) -> _FittedFold:
    # The single place a fold's model is fitted, so predictions and feature importance
    # always describe the same model.
    train_idx = f.train
    test_idx = f.test
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    if task == "classification" and len(np.unique(y_tr)) < 2:
        sub = f.subject or (str(groups[test_idx[0]]) if len(test_idx) > 0 else f"fold_{f.index}")
        msg = f"Fold {f.index}: only one class in training for subject {sub}."
        raise ValueError(msg)

    features = np.ones(X.shape[1], dtype=np.bool_)
    if harmonization is not None:
        X_tr, X_te, features = harmonize_fold(
            X_tr,
            X_te,
            groups[train_idx],
            mode=harmonization,
            n_covariates=0,
        )

    if residualize_on:
        if covariates is None:
            msg = "Target residualization requested via residualize_on, but covariates is None."
            raise ValueError(msg)
        y_tr, y_te = residualize_targets(
            np.asarray(y, dtype=np.float64),
            covariates,
            train_idx,
            test_idx,
            columns=residualize_on,
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
            scoring=scoring,
            refit=refit,
        )
        model = tuned.estimator
        best_params = tuned.best_params
    else:
        model = fit_untuned(pipeline, X_tr, y_tr, seed=seed, fold=f.index)
        best_params = {}

    _check_subject_missingness(model, X_tr, groups[train_idx])
    return _FittedFold(
        model=model,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        features=features,
        best_params=best_params,
    )


def _cross_fit_engine(
    task: str,
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    scoring: object = None,
    refit: str | bool | None = None,
) -> list[FoldPrediction] | list[FoldClassification]:
    inner_groups_all = _validate_and_resolve_inner_groups(folds, inner, groups, runs)

    def _execute_fold(f: Fold) -> FoldPrediction | FoldClassification:
        fitted = _fit_fold(
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
            scoring=scoring,
            refit=refit,
        )
        model = fitted.model

        if task == "regression":
            y_pred = model.predict(fitted.X_test)
            return FoldPrediction(
                fold=f.index,
                subject=f.subject,
                rows=f.test,
                y_true=np.asarray(fitted.y_test, dtype=np.float64),
                y_pred=np.asarray(y_pred, dtype=np.float64),
                best_params=fitted.best_params,
            )

        y_pred = np.asarray(model.predict(fitted.X_test), dtype=np.intp)
        raw_classes = getattr(model, "classes_", None)
        if raw_classes is None and hasattr(model, "steps") and len(model.steps) > 0:
            raw_classes = getattr(model.steps[-1][1], "classes_", None)
        classes = (
            tuple(int(c) for c in raw_classes)
            if raw_classes is not None
            else tuple(int(c) for c in np.unique(fitted.y_train))
        )
        y_prob: npt.NDArray[np.float64] | None = None
        if hasattr(model, "predict_proba"):
            y_prob = np.asarray(model.predict_proba(fitted.X_test), dtype=np.float64)

        return FoldClassification(
            fold=f.index,
            subject=f.subject,
            rows=f.test,
            y_true=np.asarray(fitted.y_test, dtype=np.intp),
            y_pred=y_pred,
            y_prob=y_prob,
            classes=classes,
            best_params=fitted.best_params,
        )

    results = run_folds(folds, _execute_fold, outer_n_jobs=outer_n_jobs)
    return cast(list[FoldPrediction] | list[FoldClassification], results)


def cross_fit_regression(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    scoring: object = None,
    refit: str | bool | None = None,
) -> tuple[FoldPrediction, ...]:
    results = _cross_fit_engine(
        "regression",
        folds,
        X,
        y,
        groups,
        pipeline,
        grid,
        inner=inner,
        seed=seed,
        runs=runs,
        outer_n_jobs=outer_n_jobs,
        harmonization=harmonization,
        covariates=covariates,
        residualize_on=residualize_on,
        scoring=scoring,
        refit=refit,
    )
    return tuple(cast(list[FoldPrediction], results))


def cross_fit_classification(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    *,
    inner: InnerSplit,
    seed: int,
    runs: npt.NDArray[np.object_] | None = None,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    scoring: object = None,
    refit: str | bool | None = None,
) -> tuple[FoldClassification, ...]:
    results = _cross_fit_engine(
        "classification",
        folds,
        X,
        y,
        groups,
        pipeline,
        grid,
        inner=inner,
        seed=seed,
        runs=runs,
        outer_n_jobs=outer_n_jobs,
        harmonization=harmonization,
        scoring=scoring,
        refit=refit,
    )
    return tuple(cast(list[FoldClassification], results))
