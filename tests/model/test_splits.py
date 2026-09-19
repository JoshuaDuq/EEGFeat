from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegfeat.model.splits import (
    Fold,
    InnerSplit,
    find_run_column,
    inner_cv,
    inner_cv_splits,
    loso_folds,
    within_subject_folds,
)

GROUPS = np.array(["s1", "s1", "s2", "s2", "s3"], dtype=object)


def test_loso_folds_never_share_a_subject_between_train_and_test() -> None:
    for fold in loso_folds(GROUPS):
        assert not set(GROUPS[fold.train]) & set(GROUPS[fold.test])


def test_loso_folds_hold_out_exactly_one_subject_each() -> None:
    folds = loso_folds(GROUPS)
    assert len(folds) == 3
    assert [set(GROUPS[f.test]) for f in folds] == [{"s1"}, {"s2"}, {"s3"}]


def test_loso_folds_cover_every_trial_exactly_once_across_test_sets() -> None:
    tested = np.concatenate([f.test for f in loso_folds(GROUPS)])
    np.testing.assert_array_equal(np.sort(tested), np.arange(GROUPS.size))


def test_loso_folds_carry_no_subject_label() -> None:
    assert all(f.subject is None for f in loso_folds(GROUPS))


def test_inner_cv_refuses_a_single_training_group() -> None:
    # Tuning inside a fold needs at least two groups to split on; with one, the inner
    # split is not group-disjoint and the tuned hyperparameters leak the held-out subject.
    with pytest.raises(ValueError, match="at least 2"):
        inner_cv(
            np.array(["s1", "s1", "s1"], dtype=object),
            InnerSplit(grouping="run", n_splits=3),
        )


def test_inner_cv_splits_are_capped_by_available_groups() -> None:
    assert inner_cv_splits(3, default=5) == 3
    assert inner_cv_splits(10, default=5) == 5


def test_a_within_subject_fold_cannot_be_grouped_by_subject() -> None:
    # Its training rows are one subject, so a subject-grouped inner split has one group.
    # Refusing the combination here beats a puzzling "at least 2 groups" from inside tuning.
    with pytest.raises(ValueError, match="within-subject"):
        inner_cv(
            np.array(["s1"] * 6, dtype=object),
            InnerSplit(grouping="subject", n_splits=3),
        )


def test_a_stratified_inner_split_requires_labels() -> None:
    with pytest.raises(ValueError, match="y_train"):
        inner_cv(
            np.array(["r1", "r2", "r3"], dtype=object),
            InnerSplit(grouping="run", stratified=True, n_splits=2),
        )


def test_fold_is_frozen() -> None:
    fold = Fold(index=1, train=np.array([0], dtype=np.intp), test=np.array([1], dtype=np.intp))
    with pytest.raises(AttributeError):
        fold.index = 2  # type: ignore[misc]


def test_create_within_subject_folds_respects_outer_cv_splits() -> None:
    groups = np.array(["sub-0001"] * 6, dtype=object)
    blocks = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    folds = within_subject_folds(
        groups=groups,
        blocks=blocks,
        inner_splits=5,
        outer_splits=2,
        seed=42,
    )
    assert len(folds) == 2


def test_create_within_subject_folds_requires_runs() -> None:
    groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
    with pytest.raises(ValueError, match="run labels"):
        within_subject_folds(
            groups=groups,
            blocks=None,
            inner_splits=2,
            outer_splits=2,
            seed=42,
        )


def test_create_within_subject_folds_rejects_insufficient_subject_blocks() -> None:
    groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
    blocks = np.array([1, 1, 1, 2], dtype=int)
    with pytest.raises(ValueError, match="insufficient runs"):
        within_subject_folds(
            groups=groups,
            blocks=blocks,
            inner_splits=2,
            outer_splits=2,
            seed=42,
        )


def test_create_within_subject_folds_supports_forward_ordering() -> None:
    groups = np.array(["sub-0001"] * 6, dtype=object)
    blocks = np.array([1, 1, 2, 2, 3, 3], dtype=float)
    folds = within_subject_folds(
        groups=groups,
        blocks=blocks,
        inner_splits=3,
        outer_splits=2,
        seed=42,
        ordered_runs=True,
    )
    assert len(folds) == 2
    for fold in folds:
        train_blocks = blocks[fold.train]
        test_blocks = blocks[fold.test]
        assert np.max(train_blocks) < np.min(test_blocks)


def test_create_within_subject_folds_raises_when_ordered_runs_cannot_be_formed() -> None:
    groups = np.array(["sub-0001"] * 4, dtype=object)
    blocks = np.array(["run-a", "run-a", "run-b", "run-b"], dtype=object)
    with pytest.raises(ValueError, match="ordered within-subject CV requested"):
        within_subject_folds(
            groups=groups,
            blocks=blocks,
            inner_splits=2,
            outer_splits=2,
            seed=42,
            ordered_runs=True,
        )


def test_find_run_column_parses_run_prefixed_labels() -> None:
    events = pd.DataFrame({"run": ["run-01", "run-01", "run-02", "run-02"]})
    runs = find_run_column(events)
    assert runs is not None
    vals = pd.to_numeric(runs, errors="coerce").to_numpy(dtype=float)
    np.testing.assert_allclose(vals, np.array([1.0, 1.0, 2.0, 2.0], dtype=float), atol=1e-12)
