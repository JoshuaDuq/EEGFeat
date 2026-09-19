from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from eegfeat.model import _deps as _deps
from eegfeat.model.design import harmonize_fold
from eegfeat.model.execution import run_folds
from eegfeat.model.residualize import residualize_targets
from eegfeat.model.splits import Fold, InnerSplit
from eegfeat.model.tuning import fit_untuned, tune

__all__ = [
    "FoldPrediction",
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
    if inner.grouping == "run" and runs is None:
        msg = "within-subject grouping on runs requires 'runs' array."
        raise ValueError(msg)
    if inner.grouping == "subject" and any(f.subject is not None for f in folds):
        msg = "within-subject folds cannot be grouped by subject (inner.grouping == 'subject')."
        raise ValueError(msg)

    inner_groups_all = runs if inner.grouping == "run" else groups
    if inner_groups_all is None:
        msg = "Inner grouping array is missing."
        raise ValueError(msg)

    n_cov = covariates.shape[1] if covariates is not None else 0

    def _execute_fold(f: Fold) -> FoldPrediction:
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
                n_covariates=n_cov,
            )

        if residualize_on and covariates is not None:
            y_tr, y_te = residualize_targets(
                y,
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

        y_pred = model.predict(X_te)
        return FoldPrediction(
            fold=f.index,
            subject=f.subject,
            rows=test_idx,
            y_true=y_te,
            y_pred=np.asarray(y_pred, dtype=np.float64),
            best_params=best_params,
        )

    results = run_folds(folds, _execute_fold, outer_n_jobs=outer_n_jobs)
    return tuple(results)
