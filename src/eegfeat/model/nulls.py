from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from eegfeat.model.aggregate import AggregationConfig, subject_level_r
from eegfeat.model.crossfit import cross_fit_regression
from eegfeat.model.splits import Fold, InnerSplit
from eegfeat.model.tuning import FoldFitError

__all__ = [
    "NullConfig",
    "NullResult",
    "Scheme",
    "changed_fraction",
    "circular_shift_group",
    "is_permutation_valid_run",
    "permutation_test",
    "permute",
]

Scheme = Literal[
    "within_subject",
    "run_wise",
    "within_subject_within_run",
    "circular_shift_within_run",
]


@dataclass(frozen=True)
class NullConfig:
    scheme: Scheme = "within_subject"
    n_permutations: int = 1000
    min_complete_fraction: float = 0.9
    min_retained_trials: int = 8

    def __post_init__(self) -> None:
        valid: tuple[str, ...] = (
            "within_subject",
            "run_wise",
            "within_subject_within_run",
            "circular_shift_within_run",
        )
        if self.scheme not in valid:
            msg = f"Unknown scheme {self.scheme!r}. Expected one of: {valid}."
            raise ValueError(msg)
        if self.n_permutations <= 0:
            msg = f"n_permutations must be > 0, got {self.n_permutations}."
            raise ValueError(msg)
        if not (0.0 <= self.min_complete_fraction <= 1.0):
            msg = f"min_complete_fraction must be in [0, 1], got {self.min_complete_fraction}."
            raise ValueError(msg)
        if self.min_retained_trials < 1:
            msg = f"min_retained_trials must be >= 1, got {self.min_retained_trials}."
            raise ValueError(msg)


@dataclass(frozen=True)
class NullResult:
    p_value: float
    observed: float
    null: npt.NDArray[np.float64]
    changed_fractions: npt.NDArray[np.float64]
    n_incomplete: int


_DEFAULT_AGGREGATION = AggregationConfig()


def is_permutation_valid_run(retained: npt.NDArray[np.intp], min_retained: int = 8) -> bool:
    arr = np.asarray(retained, dtype=np.intp)
    return bool(arr.size >= min_retained and np.all(np.isfinite(arr)))


def circular_shift_group(n_retained: int) -> tuple[int, ...]:
    if n_retained <= 0:
        return ()
    return tuple(range(n_retained))


def changed_fraction(
    y_original: npt.NDArray[np.float64],
    y_permuted: npt.NDArray[np.float64],
) -> float:
    orig = np.asarray(y_original, dtype=np.float64)
    perm = np.asarray(y_permuted, dtype=np.float64)
    if orig.shape != perm.shape:
        msg = f"y_original and y_permuted shape mismatch: {orig.shape} vs {perm.shape}."
        raise ValueError(msg)
    if len(orig) == 0:
        return 0.0
    finite = np.isfinite(orig) & np.isfinite(perm)
    if np.sum(finite) == 0:
        return 0.0
    return float(np.mean(orig[finite] != perm[finite]))


def permute(
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None = None,
    trial_indices: npt.NDArray[np.intp] | None = None,
    *,
    config: NullConfig,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    values = np.asarray(y, dtype=np.float64)
    groups_arr = np.asarray(groups, dtype=object)
    if len(values) != len(groups_arr):
        msg = "Permutation values and groups must have the same length."
        raise ValueError(msg)

    source_indices = np.arange(len(values), dtype=np.intp)

    if config.scheme in ("run_wise", "within_subject_within_run", "circular_shift_within_run"):
        if runs is None:
            msg = f"Permutation scheme {config.scheme!r} requires run labels."
            raise ValueError(msg)
        runs_arr = np.asarray(runs)
        if len(runs_arr) != len(values):
            msg = (
                f"Permutation runs must have the same length as y when scheme is {config.scheme!r}."
            )
            raise ValueError(msg)
        n_unlabelled = int(np.sum(pd.isna(runs_arr)))
        if n_unlabelled:
            # Runs are paradigm-specific: a trial without a label has no run to be exchanged
            # within, and inventing one would change the hypothesis being tested.
            msg = (
                f"Permutation scheme {config.scheme!r} requires run labels for every trial; "
                f"{n_unlabelled} of {len(runs_arr)} have none."
            )
            raise ValueError(msg)
    else:
        runs_arr = None

    if config.scheme == "circular_shift_within_run":
        if trial_indices is None:
            msg = "circular_shift_within_run requires within-run trial indices."
            raise ValueError(msg)
        trial_arr = np.asarray(trial_indices, dtype=float)
        if len(trial_arr) != len(values):
            msg = (
                "Permutation trial indices must have the same length as y "
                "when scheme is 'circular_shift_within_run'."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(trial_arr)):
            msg = "circular_shift_within_run requires finite within-run trial indices."
            raise ValueError(msg)
        trial_indices_arr = trial_arr.astype(np.intp)
    else:
        trial_indices_arr = None

    unique_subs = [s for s in pd.unique(groups_arr) if not pd.isna(s)]

    for subj in unique_subs:
        subj_mask = groups_arr == subj
        if np.sum(subj_mask) < 2:
            continue

        if config.scheme == "within_subject":
            subj_idx = np.where(subj_mask)[0]
            source_indices[subj_idx] = rng.permutation(source_indices[subj_idx])

        elif config.scheme in ("run_wise", "within_subject_within_run") and runs_arr is not None:
            # A subject with one run is still shuffled within it; skipping it would put its
            # real labels into every draw.
            subj_idx = np.flatnonzero(subj_mask)
            for r in pd.unique(runs_arr[subj_idx]):
                run_idx = subj_idx[runs_arr[subj_idx] == r]
                if len(run_idx) >= 2:
                    source_indices[run_idx] = rng.permutation(source_indices[run_idx])

        elif (
            config.scheme == "circular_shift_within_run"
            and runs_arr is not None
            and trial_indices_arr is not None
        ):
            subj_runs = runs_arr[subj_mask]
            unique_runs = [r for r in pd.unique(subj_runs) if not pd.isna(r)]
            for r in unique_runs:
                run_idx = np.where(subj_mask & (runs_arr == r))[0]
                order = np.argsort(trial_indices_arr[run_idx], kind="stable")
                ordered_run_idx = run_idx[order]
                n_trials = len(ordered_run_idx)
                if n_trials < config.min_retained_trials:
                    msg = (
                        f"circular_shift_within_run requires runs with at "
                        f"least {config.min_retained_trials} retained trials, got {n_trials}."
                    )
                    raise ValueError(msg)
                shift_group = circular_shift_group(n_trials)
                shift = int(rng.choice(shift_group))
                source_indices[ordered_run_idx] = np.roll(source_indices[ordered_run_idx], shift)

    return values[source_indices]


def permutation_test(
    folds: Sequence[Fold],
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    runs: npt.NDArray[np.object_] | None,
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[object]],
    observed: float,
    *,
    config: NullConfig,
    inner: InnerSplit,
    seed: int,
    outer_n_jobs: int = 1,
    harmonization: str | None = None,
    covariates: npt.NDArray[np.float64] | None = None,
    residualize_on: Sequence[str] = (),
    scoring: object = None,
    refit: str | bool | None = None,
    metric_fn: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float] | None = None,
    trial_indices: npt.NDArray[np.intp] | None = None,
    aggregation: AggregationConfig = _DEFAULT_AGGREGATION,
) -> NullResult:
    rng = np.random.default_rng(seed)
    groups_arr = np.asarray(groups, dtype=object)
    # Each draw needs only the point estimate, so the per-draw bootstrap CI is skipped.
    null_aggregation = replace(aggregation, ci_method="fixed_effects")
    permuted_targets: list[npt.NDArray[np.float64]] = []
    sampled_changed_fractions: list[float] = []

    for _ in range(config.n_permutations):
        y_p = permute(y, groups, runs, trial_indices, config=config, rng=rng)
        cf = changed_fraction(y, y_p)
        permuted_targets.append(y_p)
        sampled_changed_fractions.append(cf)

    changed_arr = np.asarray(sampled_changed_fractions, dtype=np.float64)
    if np.all(changed_arr == 0.0):
        msg = "Permutation scheme never changes any labels under this design."
        raise ValueError(msg)

    null_scores: list[float] = []
    completed_changed_fractions: list[float] = []
    n_incomplete = 0

    for b, (y_perm, cf) in enumerate(zip(permuted_targets, sampled_changed_fractions, strict=True)):
        try:
            predictions = cross_fit_regression(
                folds,
                X,
                y_perm,
                groups,
                pipeline,
                grid,
                inner=inner,
                seed=seed + b,
                runs=runs,
                outer_n_jobs=outer_n_jobs,
                harmonization=harmonization,
                covariates=covariates,
                residualize_on=residualize_on,
                scoring=scoring,
                refit=refit,
            )
        except FoldFitError:
            n_incomplete += 1
            continue
        y_true_all = np.concatenate([p.y_true for p in predictions])
        y_pred_all = np.concatenate([p.y_pred for p in predictions])
        if not np.all(np.isfinite(y_pred_all)):
            n_incomplete += 1
            continue

        if metric_fn is not None:
            score = float(metric_fn(y_true_all, y_pred_all))
            if not np.isfinite(score):
                score = 0.0
        else:
            # The same statistic as the documented observed value: subject-level r over the
            # design's subjects, whatever the fold source. A subject whose predictions do
            # not vary counts as no correlation instead of voiding the other subjects.
            pred_df = pd.DataFrame(
                {
                    "subject_id": np.concatenate([groups_arr[p.rows] for p in predictions]),
                    "y_true": y_true_all,
                    "y_pred": y_pred_all,
                }
            )
            score = subject_level_r(pred_df, config=null_aggregation, undefined="zero").r

        null_scores.append(score)
        completed_changed_fractions.append(cf)

    n_completed = len(null_scores)
    completion_rate = n_completed / config.n_permutations
    if completion_rate < config.min_complete_fraction:
        msg = (
            f"Insufficient valid permutations ({n_completed}/{config.n_permutations}, "
            f"rate={completion_rate:.3f} < required {config.min_complete_fraction:.3f})."
        )
        raise ValueError(msg)

    null_arr = np.asarray(null_scores, dtype=np.float64)
    if np.isnan(observed):
        p_value = float("nan")
    else:
        # Incomplete draws are faults, not draws that fell short of the observed value, so
        # they are left out of both terms.
        count_extreme = int(np.sum(null_arr >= observed))
        p_value = float((count_extreme + 1) / (null_arr.size + 1))

    return NullResult(
        p_value=p_value,
        observed=observed,
        null=null_arr,
        changed_fractions=np.asarray(completed_changed_fractions, dtype=np.float64),
        n_incomplete=n_incomplete,
    )
