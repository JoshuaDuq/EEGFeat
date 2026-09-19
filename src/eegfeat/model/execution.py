from __future__ import annotations

import random as pyrandom
import warnings
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from eegfeat.model import _deps as _deps
from eegfeat.model.splits import Fold

__all__ = [
    "determine_inner_n_jobs",
    "inner_n_jobs",
    "run_folds",
    "set_random_seeds",
    "should_parallelize",
    "should_parallelize_folds",
]

_T = TypeVar("_T")


def set_random_seeds(seed: int, fold: int) -> None:
    combined = seed + fold
    np.random.seed(combined)
    pyrandom.seed(combined)


def inner_n_jobs(outer_n_jobs: int, n_jobs: int) -> int:
    return 1 if (outer_n_jobs and outer_n_jobs != 1) else n_jobs


def should_parallelize(outer_n_jobs: int, n_folds: int) -> bool:
    return bool(outer_n_jobs and outer_n_jobs != 1 and n_folds > 1)


def run_folds(
    folds: Sequence[Fold],
    work: Callable[[Fold], _T],
    *,
    outer_n_jobs: int,
) -> list[_T]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"sklearn\.utils\.parallel",
        )
        if should_parallelize(outer_n_jobs, len(folds)):
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(
                delayed(work)(fold) for fold in folds
            )
            return list(results)
        return [work(fold) for fold in folds]


determine_inner_n_jobs = inner_n_jobs
should_parallelize_folds = should_parallelize
