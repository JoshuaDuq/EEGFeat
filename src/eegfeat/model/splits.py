from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold


@dataclass(frozen=True)
class Fold:
    index: int
    train: npt.NDArray[np.intp]
    test: npt.NDArray[np.intp]
    subject: str | None = None


@dataclass(frozen=True)
class InnerSplit:
    grouping: Literal["subject", "run"]
    stratified: bool = False
    n_splits: int = 5

    def __post_init__(self) -> None:
        if self.grouping not in ("subject", "run"):
            raise ValueError(f"grouping must be 'subject' or 'run', got {self.grouping!r}")
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")


def parse_run_label_to_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if match := re.match(r"^run-(\d+)$", s):
        return int(match.group(1))
    # A plain number is read as a number before trailing digits are scanned for, so that the
    # float 1.0 is run 1 and not run 0 from the digit after the decimal point.
    try:
        return int(float(s))
    except (ValueError, TypeError):
        pass
    if match := re.search(r"(\d+)$", s):
        return int(match.group(1))
    return None


def find_run_column(events: pd.DataFrame) -> pd.Series | None:
    if "run" not in events.columns:
        return None
    series = events["run"]
    numeric = pd.to_numeric(series, errors="coerce")
    # Every value that is actually present has to coerce, not just one of them: a column of
    # "run-1", "run-2", 3 would otherwise return the numeric view, in which the two labelled
    # runs have silently become NaN. An already-missing entry is not a failure to coerce.
    present = ~pd.isna(series).to_numpy()
    coerced = np.isfinite(numeric.to_numpy(dtype=float))
    if np.all(coerced[present]):
        return numeric
    parsed = pd.Series(
        [parse_run_label_to_int(v) for v in series],
        index=series.index,
        dtype="float64",
    )
    if np.any(np.isfinite(parsed.to_numpy(dtype=float))):
        return parsed
    return numeric


def inner_cv_splits(n_unique_groups: int, *, default: int = 5) -> int:
    if n_unique_groups < 2:
        msg = f"Inner CV requires at least 2 training groups, got {n_unique_groups}."
        raise ValueError(msg)
    return max(2, min(default, n_unique_groups))


def run_aware_cv(
    blocks: npt.NDArray[np.object_],
    *,
    n_splits: int | None = None,
    default_splits: int = 5,
) -> tuple[GroupKFold | None, int]:
    target_splits = default_splits if n_splits is None else n_splits
    # Runs are paradigm-specific, so a trial without one has no run to be held out with.
    n_unlabelled = int(np.sum(pd.isna(blocks)))
    if n_unlabelled:
        raise ValueError(
            f"Run-aware CV needs a run label for every trial; {n_unlabelled} of "
            f"{len(blocks)} have none."
        )
    n_unique = len(np.unique(blocks))
    if n_unique < 2:
        return None, 0
    effective_splits = min(target_splits, n_unique)
    return GroupKFold(n_splits=effective_splits), effective_splits


def run_aware_inner_cv(
    blocks_train: npt.NDArray[np.object_],
    n_splits: int,
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] | None:
    """Run-disjoint inner splits for one training fold.

    Takes no seed: :class:`~sklearn.model_selection.GroupKFold` partitions groups
    deterministically, so these splits are reproducible without one. It gained a
    ``shuffle`` option in scikit-learn 1.6, which this package does not require.
    """
    block_cv, _ = run_aware_cv(blocks_train, n_splits=n_splits)
    if block_cv is None:
        return None
    splits = block_cv.split(np.arange(len(blocks_train)), groups=blocks_train)
    return [(train_idx.astype(np.intp), test_idx.astype(np.intp)) for train_idx, test_idx in splits]


def loso_folds(groups: npt.NDArray[np.object_]) -> tuple[Fold, ...]:
    groups_arr = np.asarray(groups)
    logo = LeaveOneGroupOut()
    dummy_x = np.zeros(len(groups_arr))
    return tuple(
        Fold(
            index=fold,
            train=train_idx.astype(np.intp),
            test=test_idx.astype(np.intp),
            subject=None,
        )
        for fold, (train_idx, test_idx) in enumerate(
            logo.split(dummy_x, groups=groups_arr), start=1
        )
    )


def within_subject_folds(
    groups: npt.NDArray[np.object_],
    blocks: npt.NDArray[np.object_] | None,
    *,
    inner_splits: int,
    outer_splits: int | None = None,
    ordered_runs: bool = False,
) -> tuple[Fold, ...]:
    """Per-subject, run-disjoint outer folds.

    Takes no seed: both the forward-ordered and the GroupKFold path partition runs
    deterministically, so the folds are reproducible without one.
    """
    if blocks is None:
        raise ValueError("Within-subject CV requires run labels for every subject.")
    blocks_arr = np.asarray(blocks)
    groups_arr = np.asarray(groups)
    if blocks_arr.shape[0] != groups_arr.shape[0]:
        raise ValueError("Within-subject CV run labels must be aligned to the sample axis.")

    folds: list[Fold] = []
    fold_counter = 0

    for subject_id in np.unique(groups_arr):
        # Match on the id as stored; only the label is a string, so integer ids still match.
        subject = str(subject_id)
        subject_indices = np.where(groups_arr == subject_id)[0]
        n_samples = len(subject_indices)
        requested_splits = outer_splits if outer_splits is not None else inner_splits
        n_splits = min(max(2, requested_splits), n_samples)

        subject_blocks = blocks_arr[subject_indices]
        if np.any(pd.isna(subject_blocks)):
            raise ValueError(f"Subject {subject}: missing run labels.")

        n_unique_blocks = len(np.unique(subject_blocks))
        if n_unique_blocks < 2:
            raise ValueError(
                f"Subject {subject}: insufficient runs for within-subject CV "
                f"({n_unique_blocks}); at least 2 are required."
            )

        if ordered_runs:
            num_blocks = np.asarray(pd.to_numeric(subject_blocks, errors="coerce"), dtype=float)
            if not np.all(np.isfinite(num_blocks)):
                parsed = [parse_run_label_to_int(b) for b in subject_blocks]
                if all(p is not None for p in parsed):
                    num_blocks = np.array(parsed, dtype=float)

            # Forward CV places runs in order, so a label with no order has no position in the
            # sequence. It would otherwise fall out of both the train and the test mask below
            # and vanish from every fold without a word.
            if not np.all(np.isfinite(num_blocks)):
                unorderable = sorted(
                    {
                        str(b)
                        for b, ok in zip(subject_blocks, np.isfinite(num_blocks), strict=True)
                        if not ok
                    }
                )
                raise ValueError(
                    f"Subject {subject}: forward CV orders runs, but {unorderable[:3]} carry no "
                    "run number; rename them or pass ordered_runs=False."
                )

            # Parsing reads only trailing digits, so ses1-run1 and ses2-run1 are both run 1;
            # merging them would put a later session's run before an earlier one's.
            labels_by_number: dict[float, set[str]] = {}
            for b, num in zip(subject_blocks, num_blocks, strict=True):
                labels_by_number.setdefault(float(num), set()).add(str(b))
            shared = sorted(
                sorted(labels) for labels in labels_by_number.values() if len(labels) > 1
            )
            if shared:
                raise ValueError(
                    f"Subject {subject}: forward CV orders runs, but {shared[0]} share run "
                    "number and cannot be ordered; relabel them with distinct run numbers."
                )

            unique_nums = sorted(np.unique(num_blocks))
            if len(unique_nums) < 3:
                raise ValueError(
                    f"Subject {subject}: at least three ordered runs are required for "
                    f"nested forward CV, got {len(unique_nums)}."
                )
            ordered_splits: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = []
            # The first evaluated fold needs at least two training runs
            # so that run-disjoint inner tuning is possible.
            for block_idx in range(2, len(unique_nums)):
                train_mask = np.isin(num_blocks, [float(u) for u in unique_nums[:block_idx]])
                test_mask = num_blocks == float(unique_nums[block_idx])
                if np.any(train_mask) and np.any(test_mask):
                    ordered_splits.append(
                        (
                            np.flatnonzero(train_mask).astype(np.intp),
                            np.flatnonzero(test_mask).astype(np.intp),
                        )
                    )
            if not ordered_splits:
                raise ValueError(
                    f"Subject {subject}: ordered within-subject CV requested but no valid "
                    "ordered run folds were found."
                )
            if len(ordered_splits) > n_splits:
                ordered_splits = ordered_splits[-n_splits:]
            for train_local, test_local in ordered_splits:
                fold_counter += 1
                folds.append(
                    Fold(
                        index=fold_counter,
                        train=subject_indices[train_local].astype(np.intp),
                        test=subject_indices[test_local].astype(np.intp),
                        subject=subject,
                    )
                )
        else:
            block_cv, _ = run_aware_cv(subject_blocks, n_splits=n_splits)
            if block_cv is None:
                raise ValueError(f"Subject {subject}: insufficient runs for within-subject CV.")
            for train_local, test_local in block_cv.split(subject_indices, groups=subject_blocks):
                fold_counter += 1
                folds.append(
                    Fold(
                        index=fold_counter,
                        train=subject_indices[train_local].astype(np.intp),
                        test=subject_indices[test_local].astype(np.intp),
                        subject=subject,
                    )
                )

    return tuple(folds)


def inner_cv(
    train_groups: npt.NDArray[np.object_],
    split: InnerSplit,
    y_train: npt.NDArray[np.intp] | None = None,
    random_state: int | None = None,
) -> GroupKFold | StratifiedGroupKFold:
    groups_arr = np.asarray(train_groups)
    n_unique = len(np.unique(groups_arr))
    if split.grouping == "subject" and n_unique == 1:
        raise ValueError(
            "A within-subject fold cannot use subject-grouped inner CV (found only 1 group)."
        )
    if n_unique < 2:
        raise ValueError(f"Inner CV requires at least 2 unique groups, got {n_unique}.")

    effective_splits = max(2, min(split.n_splits, n_unique))
    if split.stratified:
        if y_train is None:
            raise ValueError("A stratified inner split requires y_train labels.")
        y_arr = np.asarray(y_train, dtype=np.intp)
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Stratified inner split requires at least 2 classes in y_train.")
        min_class_count = int(np.min(counts))
        if min_class_count < effective_splits:
            rarest = classes[int(np.argmin(counts))]
            tally = ", ".join(f"{c}: {n}" for c, n in zip(classes, counts, strict=True))
            raise ValueError(
                f"StratifiedGroupKFold requires each class to have at least {effective_splits} "
                f"training samples; class {rarest} has {min_class_count} "
                f"(class counts: {tally})."
            )
        return StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )
    return GroupKFold(n_splits=effective_splits)
