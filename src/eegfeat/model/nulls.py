from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_regression
from eegfeat.model.scoring import safe_pearsonr
from eegfeat.model.splits import Fold, InnerSplit

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

Scheme = Literal["within_subject", "run_wise", "circular_shift_within_run"]


@dataclass(frozen=True)
class NullConfig:
    scheme: Scheme = "within_subject"
    n_permutations: int = 1000
    min_complete_fraction: float = 0.9
    min_retained_trials: int = 8

    def __post_init__(self) -> None:
        valid: tuple[str, ...] = ("within_subject", "run_wise", "circular_shift_within_run")
        if self.scheme not in valid:
            msg = f"Unknown scheme {self.scheme!r}. Expected one of: {valid}."
            raise ValueError(msg)
        if self.n_permutations < 1:
            msg = f"n_permutations must be >= 1, got {self.n_permutations}."
            raise ValueError(msg)
        if not (0.0 <= self.min_complete_fraction <= 1.0):
            msg = "min_complete_fraction must be between 0.0 and 1.0."
            raise ValueError(msg)


@dataclass(frozen=True)
class NullResult:
    p_value: float
    observed: float
    null: npt.NDArray[np.float64]
    changed_fractions: npt.NDArray[np.float64]
    n_incomplete: int


def is_permutation_valid_run(
    trial_indices: npt.NDArray[np.intp] | Sequence[int],
    *,
    min_retained_trials: int = 8,
) -> bool:
    retained = np.asarray(trial_indices, dtype=np.intp)
    return bool(retained.size >= int(min_retained_trials) and retained.size >= 2)


def circular_shift_group(n_retained: int) -> tuple[int, ...]:
    """Return every within-run circular shift, the identity included.

    The upper-tail permutation p-value is justified by the transformations forming a
    group under composition, with the observed statistic as the identity element. The
    full cycle is that group; any subset of it generally is not.
    """
    count = int(n_retained)
    if count <= 0:
        return ()
    return tuple(range(count))


def changed_fraction(
    y_original: npt.NDArray[np.float64],
    y_permuted: npt.NDArray[np.float64],
) -> float:
    orig = np.asarray(y_original, dtype=np.float64)
    perm = np.asarray(y_permuted, dtype=np.float64)
    if orig.shape != perm.shape:
        msg = f"Permutation shape mismatch: {orig.shape} vs {perm.shape}."
        raise ValueError(msg)
    if orig.size == 0:
        return 0.0
    finite = np.isfinite(orig) & np.isfinite(perm)
    if not np.any(finite):
        return 0.0
    changed = int(np.sum(orig[finite] != perm[finite]))
    return float(changed / int(np.sum(finite)))


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

    if config.scheme in ("run_wise", "circular_shift_within_run"):
        if runs is None:
            msg = f"Permutation scheme {config.scheme!r} requires run labels."
            raise ValueError(msg)
        runs_arr = np.asarray(runs)
        if len(runs_arr) != len(values):
            msg = (
                f"Permutation runs must have the same length as y when scheme is {config.scheme!r}."
            )
            raise ValueError(msg)
        if np.all(pd.isna(runs_arr)):
            msg = f"Permutation scheme {config.scheme!r} requires run labels."
            raise ValueError(msg)
    else:
        runs_arr = None

    y_perm = values.copy()
    unique_subs = [s for s in pd.unique(groups_arr) if not pd.isna(s)]

    if config.scheme == "within_subject":
        for subj in unique_subs:
            mask = groups_arr == subj
            if np.sum(mask) >= 2:
                y_perm[mask] = rng.permutation(y_perm[mask])
        return y_perm

    if config.scheme == "run_wise" and runs_arr is not None:
        for subj in unique_subs:
            subj_mask = groups_arr == subj
            subj_runs = runs_arr[subj_mask]
            unique_runs = [r for r in pd.unique(subj_runs) if not pd.isna(r)]
            if len(unique_runs) < 2:
                continue
            perm_runs = rng.permutation(unique_runs)
            run_indices_orig = [np.where(subj_mask & (runs_arr == r))[0] for r in unique_runs]
            run_indices_perm = [np.where(subj_mask & (runs_arr == r))[0] for r in perm_runs]
            for orig_idx, perm_idx in zip(run_indices_orig, run_indices_perm, strict=True):
                min_len = min(len(orig_idx), len(perm_idx))
                if min_len > 0:
                    y_perm[orig_idx[:min_len]] = values[perm_idx[:min_len]]
        return y_perm

    if config.scheme == "circular_shift_within_run" and runs_arr is not None:
        for subj in unique_subs:
            subj_mask = groups_arr == subj
            subj_runs = runs_arr[subj_mask]
            unique_runs = [r for r in pd.unique(subj_runs) if not pd.isna(r)]
            for r in unique_runs:
                run_idx = np.where(subj_mask & (runs_arr == r))[0]
                if trial_indices is not None:
                    order = np.argsort(trial_indices[run_idx], kind="stable")
                    run_idx = run_idx[order]
                n_trials = len(run_idx)
                if n_trials < config.min_retained_trials:
                    msg = (
                        f"circular_shift_within_run requires runs with at "
                        f"least {config.min_retained_trials} retained trials, got {n_trials}."
                    )
                    raise ValueError(msg)
                shift_group = circular_shift_group(n_trials)
                shift = int(rng.choice(shift_group))
                y_perm[run_idx] = np.roll(values[run_idx], shift)
        return y_perm

    return y_perm


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
) -> NullResult:
    rng = np.random.default_rng(seed)
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
            y_true_all = np.concatenate([p.y_true for p in predictions])
            y_pred_all = np.concatenate([p.y_pred for p in predictions])
            if not np.all(np.isfinite(y_pred_all)):
                n_incomplete += 1
                continue
            if metric_fn is not None:
                score = float(metric_fn(y_true_all, y_pred_all))
            else:
                score, _ = safe_pearsonr(y_true_all, y_pred_all)
            if not np.isfinite(score):
                n_incomplete += 1
                continue
            null_scores.append(score)
            completed_changed_fractions.append(cf)
        except Exception:
            n_incomplete += 1

    n_completed = len(null_scores)
    completion_rate = n_completed / config.n_permutations
    if completion_rate < config.min_complete_fraction:
        msg = (
            f"Insufficient valid permutations ({n_completed}/{config.n_permutations}, "
            f"rate={completion_rate:.3f} < required {config.min_complete_fraction:.3f})."
        )
        raise ValueError(msg)

    null_arr = np.asarray(null_scores, dtype=np.float64)
    count_extreme = int(np.sum(null_arr >= observed))
    p_value = float((count_extreme + 1) / (len(null_arr) + 1))
    return NullResult(
        p_value=p_value,
        observed=observed,
        null=null_arr,
        changed_fractions=np.asarray(completed_changed_fractions, dtype=np.float64),
        n_incomplete=n_incomplete,
    )
