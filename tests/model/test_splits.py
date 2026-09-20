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
    parse_run_label_to_int,
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


def test_inner_cv_splits_rejects_fewer_than_two_groups() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        inner_cv_splits(0)
    with pytest.raises(ValueError, match="at least 2"):
        inner_cv_splits(1)


def test_stratified_inner_cv_enforces_minority_class_precondition() -> None:
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    y_train = np.array([0, 0, 0, 1], dtype=np.intp)
    with pytest.raises(ValueError, match="requires each class to have at least 2"):
        inner_cv(
            groups,
            InnerSplit(grouping="subject", stratified=True, n_splits=2),
            y_train=y_train,
        )


def test_the_stratification_error_names_the_rare_class_and_every_count() -> None:
    # A "binary" target with a stray -1 code, such as a missed response, has three classes.
    # A bare minority count hides that; naming the class and the counts reveals it.
    groups = np.array(["s1"] * 4 + ["s2"] * 4, dtype=object)
    y_train = np.array([0, 1, 0, 1, 0, 1, -1, 0], dtype=np.intp)
    with pytest.raises(ValueError, match=r"class -1 has 1 \(class counts: -1: 1, 0: 4, 1: 3\)"):
        inner_cv(
            groups,
            InnerSplit(grouping="subject", stratified=True, n_splits=2),
            y_train=y_train,
        )


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
        )


def test_create_within_subject_folds_supports_forward_ordering() -> None:
    groups = np.array(["sub-0001"] * 6, dtype=object)
    blocks = np.array([1, 1, 2, 2, 3, 3], dtype=float)
    folds = within_subject_folds(
        groups=groups,
        blocks=blocks,
        inner_splits=3,
        outer_splits=2,
        ordered_runs=True,
    )
    assert len(folds) == 1
    assert set(blocks[folds[0].train]) == {1.0, 2.0}
    assert set(blocks[folds[0].test]) == {3.0}
    for fold in folds:
        train_blocks = blocks[fold.train]
        test_blocks = blocks[fold.test]
        assert np.max(train_blocks) < np.min(test_blocks)


def test_create_within_subject_folds_raises_when_ordered_runs_cannot_be_formed() -> None:
    groups = np.array(["sub-0001"] * 4, dtype=object)
    blocks = np.array(["run-a", "run-a", "run-b", "run-b"], dtype=object)
    with pytest.raises(ValueError, match="carry no run number"):
        within_subject_folds(
            groups=groups,
            blocks=blocks,
            inner_splits=2,
            outer_splits=2,
            ordered_runs=True,
        )


def test_forward_cv_needs_three_ordered_runs() -> None:
    groups = np.array(["sub-0001"] * 4, dtype=object)
    blocks = np.array([1, 1, 2, 2], dtype=object)
    with pytest.raises(ValueError, match="at least three ordered runs"):
        within_subject_folds(groups=groups, blocks=blocks, inner_splits=2, ordered_runs=True)


def test_forward_cv_refuses_a_run_it_cannot_place_in_order() -> None:
    # An unorderable label fell out of both the train and the test mask, so its trials were
    # silently absent from every fold while ordered_runs=False used all of them.
    groups = np.array(["sub-0001"] * 8, dtype=object)
    blocks = np.array(["1", "1", "2", "2", "3", "3", "rest", "rest"], dtype=object)

    with pytest.raises(ValueError, match="carry no run number"):
        within_subject_folds(groups=groups, blocks=blocks, inner_splits=2, ordered_runs=True)

    folds = within_subject_folds(groups, blocks, inner_splits=2, ordered_runs=False)
    used = {int(i) for fold in folds for i in [*fold.train.tolist(), *fold.test.tolist()]}
    assert used == set(range(8))


def test_within_subject_folds_accept_integer_subject_ids() -> None:
    # Subject ids read from a targets table are often integers; they must match themselves.
    groups = np.array([1] * 6 + [2] * 6, dtype=object)
    blocks = np.array([1, 1, 2, 2, 3, 3] * 2, dtype=object)
    folds = within_subject_folds(groups, blocks, inner_splits=3)
    assert sorted({fold.subject for fold in folds if fold.subject is not None}) == ["1", "2"]
    for fold in folds:
        assert set(groups[fold.train]) == set(groups[fold.test]) == {int(str(fold.subject))}


def test_find_run_column_parses_run_prefixed_labels() -> None:
    events = pd.DataFrame({"run": ["run-01", "run-01", "run-02", "run-02"]})
    runs = find_run_column(events)
    assert runs is not None
    vals = pd.to_numeric(runs, errors="coerce").to_numpy(dtype=float)
    np.testing.assert_allclose(vals, np.array([1.0, 1.0, 2.0, 2.0], dtype=float), atol=1e-12)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        # One subject writing run 3 as a bare integer used to blank every other label, because
        # a single coercible value was enough to accept the all-NaN numeric view.
        (["run-1", "run-2", 3, "run-4"], [1.0, 2.0, 3.0, 4.0]),
        (["run-1", "run-2", "run-3"], [1.0, 2.0, 3.0]),
        (["1", "2", "3"], [1.0, 2.0, 3.0]),
        # A value that is already missing is not a failure to coerce, so the numeric view stands
        # and the float 1.0 stays run 1 rather than becoming run 0 via its trailing digit.
        ([1.0, 2.0, float("nan")], [1.0, 2.0, float("nan")]),
    ],
)
def test_run_labels_survive_a_mix_of_numbers_and_strings(
    labels: list[object], expected: list[float]
) -> None:
    resolved = find_run_column(pd.DataFrame({"run": labels}))
    assert resolved is not None
    np.testing.assert_array_equal(resolved.to_numpy(dtype=float), np.array(expected))


def test_a_float_run_label_parses_as_its_value_not_its_last_digit() -> None:
    assert parse_run_label_to_int(1.0) == 1
    assert parse_run_label_to_int("sub-02_run-4") == 4
