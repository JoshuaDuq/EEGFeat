from __future__ import annotations

import numpy as np

from eegfeat.model.execution import (
    inner_n_jobs,
    run_folds,
    set_random_seeds,
    should_parallelize,
)
from eegfeat.model.splits import Fold

FOLDS = tuple(
    Fold(index=i, train=np.array([0], dtype=np.intp), test=np.array([1], dtype=np.intp))
    for i in range(3)
)


def test_fold_results_come_back_in_fold_order_not_completion_order() -> None:
    # Downstream aggregation indexes fold records positionally, so a parallel backend
    # returning them as they finish would silently misattribute per-fold results.
    results = run_folds(FOLDS, lambda fold: {"index": fold.index}, outer_n_jobs=2)
    assert [r["index"] for r in results] == [0, 1, 2]


def test_the_same_seed_and_fold_reproduce_the_same_draw() -> None:
    set_random_seeds(42, 1)
    first = np.random.random()
    set_random_seeds(42, 1)
    assert np.random.random() == first


def test_different_folds_do_not_share_a_draw() -> None:
    set_random_seeds(42, 1)
    first = np.random.random()
    set_random_seeds(42, 2)
    assert np.random.random() != first


def test_a_single_fold_is_not_parallelized() -> None:
    assert not should_parallelize(outer_n_jobs=4, n_folds=1)


def test_inner_n_jobs_drops_to_one_when_outer_is_parallel() -> None:
    assert inner_n_jobs(outer_n_jobs=4, n_jobs=4) == 1
    assert inner_n_jobs(outer_n_jobs=1, n_jobs=4) == 4
    assert inner_n_jobs(outer_n_jobs=0, n_jobs=4) == 4


def test_run_folds_sequential_when_outer_is_one() -> None:
    results = run_folds(FOLDS, lambda fold: {"index": fold.index}, outer_n_jobs=1)
    assert [r["index"] for r in results] == [0, 1, 2]
