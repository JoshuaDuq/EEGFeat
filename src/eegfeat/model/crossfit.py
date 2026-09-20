from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from eegfeat.model import _deps as _deps
from eegfeat.model.design import harmonize_fold
from eegfeat.model.execution import run_folds
from eegfeat.model.residualize import residualize_targets
from eegfeat.model.splits import Fold, InnerSplit, inner_cv
from eegfeat.model.transformers import _check_subject_missingness
from eegfeat.model.tuning import FoldFitError, fit_untuned, tune

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


def _select_fold_local_params(
    f: Fold,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    groups: npt.NDArray[np.object_],
    inner_groups: npt.NDArray[np.object_],
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
) -> dict[str, object]:
    """Select hyperparameters with every external transform fitted per inner fold."""
    if refit is False:
        raise ValueError("refit=False cannot return a fitted outer-fold model.")

    chosen_scoring = scoring
    if isinstance(scoring, Mapping):
        if not isinstance(refit, str) or refit not in scoring:
            raise ValueError("Multi-metric scoring requires refit to name a scoring metric.")
        chosen_scoring = scoring[refit]

    outer_train = np.asarray(f.train, dtype=np.intp)
    train_groups = inner_groups[outer_train]
    strat_labels = np.asarray(y[outer_train], dtype=np.intp) if inner.stratified else None

    splitter = inner_cv(
        train_groups,
        inner,
        y_train=strat_labels,
        random_state=seed + max(f.index - 1, 0),
    )

    local_splits = list(
        splitter.split(
            X[outer_train],
            y[outer_train],
            groups=train_groups,
        )
    )
    if not local_splits:
        raise ValueError(f"Fold {f.index}: no inner CV splits.")

    best_score = -np.inf
    best_params: dict[str, object] | None = None

    for parameters in ParameterGrid(dict(grid)):
        fold_scores: list[float] = []

        for local_train, local_valid in local_splits:
            train_idx = outer_train[local_train]
            valid_idx = outer_train[local_valid]

            X_train = np.asarray(X[train_idx], dtype=np.float64)
            X_valid = np.asarray(X[valid_idx], dtype=np.float64)
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            if harmonization is not None:
                X_train, X_valid, _ = harmonize_fold(
                    X_train,
                    X_valid,
                    groups[train_idx],
                    mode=harmonization,
                    n_covariates=0,
                )

            if residualize_on:
                if covariates is None:
                    raise ValueError(
                        "Target residualization requested via residualize_on, "
                        "but covariates is None."
                    )

                y_train, y_valid = residualize_targets(
                    np.asarray(y, dtype=np.float64),
                    covariates,
                    train_idx,
                    valid_idx,
                    columns=residualize_on,
                )

            candidate = clone(pipeline)
            candidate.set_params(**parameters)

            try:
                fitted = fit_untuned(
                    candidate,
                    X_train,
                    y_train,
                    seed=seed,
                    fold=f.index,
                )
                _check_subject_missingness(fitted, X_train, groups[train_idx])

                scorer = check_scoring(fitted, scoring=chosen_scoring)
                score = float(scorer(fitted, X_valid, y_valid))
            except Exception as exc:
                raise FoldFitError(
                    f"Outer fold {f.index}: inner fold failed for "
                    f"parameters {parameters}: {exc}"
                ) from exc

            if not np.isfinite(score):
                raise FoldFitError(
                    f"Outer fold {f.index}: non-finite inner CV "
                    f"score for parameters {parameters}."
                )

            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(parameters)

    if best_params is None:
        raise FoldFitError(f"Outer fold {f.index}: no valid hyperparameter candidate.")

    return best_params


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
        if harmonization is not None or residualize_on:
            best_params = _select_fold_local_params(
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

            selected_pipeline = clone(pipeline)
            selected_pipeline.set_params(**best_params)

            model = fit_untuned(
                selected_pipeline,
                X_tr,
                y_tr,
                seed=seed,
                fold=f.index,
            )
        else:
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


def _validate_binary_labels(y: npt.NDArray[np.intp]) -> None:
    # Everything downstream, from the positive-class probability to the confusion matrix,
    # reads 0 and 1. A third label is a silent multiclass fit that only the metrics catch,
    # by which point every fold has been tuned.
    labels = np.unique(np.ravel(y))
    if not set(labels.tolist()) <= {0, 1}:
        raise ValueError(
            f"cross_fit_classification expects labels coded 0/1, got {labels.tolist()}."
        )


def _validate_row_aligned(
    n_rows: int,
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    covariates: npt.NDArray[np.float64] | None,
) -> None:
    # Every input indexed by fold row has to be as long as X. A covariates frame that still
    # carries the rejected epochs is longer, and indexing it positionally then residualizes
    # each trial against another trial's nuisance values without any error.
    if len(y) != n_rows:
        raise ValueError(f"y has {len(y)} rows and X has {n_rows}.")

    if covariates is not None and len(covariates) != n_rows:
        raise ValueError(f"covariates has {len(covariates)} rows and X has {n_rows}.")


def _validate_outer_folds(
    folds: Sequence[Fold],
    n_rows: int,
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None,
) -> None:
    if not folds:
        raise ValueError("At least one outer fold is required.")

    if len(groups) != n_rows:
        raise ValueError("groups length does not match X.")

    if runs is not None and len(runs) != n_rows:
        raise ValueError("runs length does not match X.")

    tested: set[int] = set()

    for fold in folds:
        tr = np.asarray(fold.train)
        te = np.asarray(fold.test)

        for label, idx in (("train", tr), ("test", te)):
            if idx.ndim != 1 or idx.size == 0:
                raise ValueError(f"Fold {fold.index}: {label} must be a nonempty 1-D array.")
            if not np.issubdtype(idx.dtype, np.integer):
                raise ValueError(f"Fold {fold.index}: {label} indices must be integers.")
            if np.any(idx < 0) or np.any(idx >= n_rows):
                raise ValueError(f"Fold {fold.index}: {label} contains out-of-range indices.")
            if np.unique(idx).size != idx.size:
                raise ValueError(f"Fold {fold.index}: duplicate {label} indices.")

        if np.intersect1d(tr, te).size:
            raise ValueError(f"Fold {fold.index}: train/test overlap.")

        duplicated_tests = tested.intersection(map(int, te))
        if duplicated_tests:
            raise ValueError(f"Fold {fold.index}: observations tested in multiple folds.")
        tested.update(map(int, te))

        train_subjects = set(groups[tr])
        test_subjects = set(groups[te])

        if fold.subject is None:
            # Convention used by loso_folds().
            if train_subjects & test_subjects:
                raise ValueError(f"Fold {fold.index}: subject overlap in a LOSO fold.")
        else:
            expected = {fold.subject}
            if {str(v) for v in train_subjects} != expected:
                raise ValueError(
                    f"Fold {fold.index}: training subjects do not match " f"{fold.subject!r}."
                )
            if {str(v) for v in test_subjects} != expected:
                raise ValueError(
                    f"Fold {fold.index}: test subjects do not match " f"{fold.subject!r}."
                )
            if runs is None:
                raise ValueError(f"Fold {fold.index}: within-subject folds require runs.")
            if set(runs[tr]) & set(runs[te]):
                raise ValueError(f"Fold {fold.index}: train/test run overlap.")


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
    _validate_outer_folds(folds, len(X), groups, runs)
    _validate_row_aligned(len(X), y, covariates)

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
    _validate_binary_labels(y)
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
