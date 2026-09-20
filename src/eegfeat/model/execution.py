from __future__ import annotations

import random as pyrandom
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

import numpy as np

from eegfeat.model import _deps as _deps
from eegfeat.model.splits import Fold

__all__ = [
    "inner_n_jobs",
    "run_folds",
    "seeded",
    "set_random_seeds",
    "should_parallelize",
]

_T = TypeVar("_T")


def set_random_seeds(seed: int, fold: int) -> None:
    combined = seed + fold
    np.random.seed(combined)
    pyrandom.seed(combined)


@contextmanager
def seeded(seed: int, fold: int) -> Iterator[None]:
    # Seeding the global generators is what makes an estimator that consults them reproducible,
    # but leaving them seeded would reset the caller's own random stream as a side effect of
    # fitting a model, so anything they drew afterwards would depend on how many folds ran.
    np_state = np.random.get_state()
    py_state = pyrandom.getstate()
    try:
        set_random_seeds(seed, fold)
        yield
    finally:
        np.random.set_state(np_state)
        pyrandom.setstate(py_state)


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
            module=r".*sklearn[/.]utils[/.]parallel",
        )
        if should_parallelize(outer_n_jobs, len(folds)):
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=outer_n_jobs, prefer="processes")(
                delayed(work)(fold) for fold in folds
            )
            return list(results)
        return [work(fold) for fold in folds]
