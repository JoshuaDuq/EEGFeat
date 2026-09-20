from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from eegfeat.model import _deps as _deps
from eegfeat.model.execution import seeded
from eegfeat.model.splits import InnerSplit, inner_cv

__all__ = [
    "FoldFitError",
    "TunedFit",
    "fit_untuned",
    "tune",
]


class FoldFitError(ValueError, RuntimeError):
    """A fold's estimator could not be fitted: a fault, not a measurement.

    Subclasses ``ValueError`` so existing handlers keep working, and ``RuntimeError`` so a
    caller can count fit failures separately from designs that cannot be fitted at all.
    """


@dataclass(frozen=True)
class TunedFit:
    estimator: Pipeline
    best_params: dict[str, object]


def _raise_for_nonfinite_grid_search_scores(grid: GridSearchCV, fold: int) -> None:
    cv_results = getattr(grid, "cv_results_", None)
    if not isinstance(cv_results, dict):
        return

    bad_score_keys: list[str] = []
    for key, values in cv_results.items():
        is_test_score = key.startswith("mean_test") or (key.startswith("split") and "_test" in key)
        if not is_test_score:
            continue
        scores: npt.NDArray[np.float64] = np.ma.asarray(values, dtype=float).filled(np.nan)
        if scores.size and not np.all(np.isfinite(scores)):
            bad_score_keys.append(str(key))

    if bad_score_keys:
        msg = f"Fold {fold}: non-finite inner CV test scores: {', '.join(bad_score_keys)}."
        raise FoldFitError(msg)


def _assign_random_state(estimator: object, seed: int) -> None:
    if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
        random_state_keys = [
            key
            for key in estimator.get_params(deep=True)
            if key == "random_state" or key.endswith("__random_state")
        ]
        if random_state_keys:
            estimator.set_params(**{key: seed for key in random_state_keys})


def tune(
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    inner_groups_train: npt.NDArray[np.object_],
    *,
    split: InnerSplit,
    seed: int,
    fold: int,
    n_jobs: int = 1,
    scoring: object = None,
    refit: str | bool | None = None,
) -> TunedFit:
    if refit is False:
        raise ValueError("refit=False cannot return a fitted outer-fold model.")

    groups_arr = np.asarray(inner_groups_train)
    n_unique = len(np.unique(groups_arr))
    if n_unique < 2:
        msg = f"Fold {fold}: inner CV requires at least 2 unique groups, got {n_unique}."
        raise ValueError(msg)

    with seeded(seed, fold):
        y_strat = np.asarray(y_train, dtype=np.intp) if split.stratified else None
        inner_seed = seed + (fold - 1 if fold > 0 else 0)
        cv = inner_cv(groups_arr, split, y_train=y_strat, random_state=inner_seed)

        pipe_clone = cast(Pipeline, clone(pipeline))
        _assign_random_state(pipe_clone, seed)

        refit_param: str | bool = refit if refit is not None else True
        gs = GridSearchCV(
            estimator=pipe_clone,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            refit=refit_param,
            error_score="raise",
            n_jobs=n_jobs,
        )

        try:
            gs.fit(X_train, y_train, groups=groups_arr)
        except Exception as exc:
            msg = f"Fold {fold}: inner CV failed: {exc}"
            raise FoldFitError(msg) from exc

    _raise_for_nonfinite_grid_search_scores(gs, fold)
    return TunedFit(
        estimator=cast(Pipeline, gs.best_estimator_),
        best_params=dict(gs.best_params_),
    )


def fit_untuned(
    pipeline: Pipeline,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    *,
    seed: int,
    fold: int = 0,
) -> Pipeline:
    with seeded(seed, fold):
        pipe_clone = cast(Pipeline, clone(pipeline))
        _assign_random_state(pipe_clone, seed)
        try:
            pipe_clone.fit(X, y)
        except Exception as exc:
            msg = f"Fold {fold}: fit failed: {exc}"
            raise FoldFitError(msg) from exc
    return pipe_clone
