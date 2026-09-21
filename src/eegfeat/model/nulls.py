from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from eegfeat.model.aggregate import AggregationConfig, subject_level_r
from eegfeat.model.crossfit import FoldPrediction, cross_fit_regression
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

# "run_wise" is an alias for "within_subject_within_run", kept because the upstream pipeline
# calls this shuffle "runwise". Both shuffle labels within each run of each subject; neither
# exchanges whole runs. Run structure is paradigm-specific, so a run-block exchange is not
# offered rather than guessed at. The alias is pinned by a test.
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
    # Cast to float, not int: an int cast turns NaN into a large negative integer, which then
    # passes any finiteness test and reports a run of missing counts as usable.
    arr = np.asarray(retained, dtype=np.float64)
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
        if trial_arr.ndim != 1 or np.any(trial_arr != np.floor(trial_arr)):
            raise ValueError("Circular shifts require 1-D integer trial indices.")
        trial_indices_arr = trial_arr.astype(np.intp)
    else:
        trial_indices_arr = None

    # A trial with no subject belongs to no exchangeable block, so it would keep its observed
    # target in every draw and pull the null toward the observed statistic. Refuse it, exactly
    # as a missing run label is refused above.
    n_unlabelled = int(np.sum(pd.isna(groups_arr)))
    if n_unlabelled:
        msg = (
            f"Permutation requires subject labels for every trial; "
            f"{n_unlabelled} of {len(groups_arr)} have none."
        )
        raise ValueError(msg)

    for subj in pd.unique(groups_arr):
        subj_mask = groups_arr == subj
        if np.sum(subj_mask) < 2:
            # Nothing to exchange this trial with, so it would carry its observed target into
            # every draw. Skipping the subject also skipped the per-run size checks below, so a
            # one-trial subject slipped past min_retained_trials that a seven-trial run fails.
            msg = (
                f"Subject {subj!r} has a single trial, which cannot be permuted and would keep "
                "its observed target in every draw; drop the subject before testing."
            )
            raise ValueError(msg)

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
            # Missing run labels were refused above, so every run here is a real one.
            for r in pd.unique(runs_arr[subj_mask]):
                run_idx = np.where(subj_mask & (runs_arr == r))[0]
                if np.unique(trial_indices_arr[run_idx]).size != run_idx.size:
                    raise ValueError(
                        "Circular shifts require unique trial indices within each run."
                    )
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


def _prediction_statistic(
    predictions: Sequence[FoldPrediction],
    groups: npt.NDArray[np.object_],
    aggregation: AggregationConfig,
    metric_fn: (
        Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]],
            float,
        ]
        | None
    ),
) -> float:
    yt = np.concatenate([p.y_true for p in predictions])
    yp = np.concatenate([p.y_pred for p in predictions])

    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("Permutation statistic requires finite predictions.")

    if metric_fn is not None:
        score = float(metric_fn(yt, yp))
    else:
        frame = pd.DataFrame(
            {
                "subject_id": np.concatenate([groups[p.rows] for p in predictions]),
                "y_true": yt,
                "y_pred": yp,
            }
        )
        score = subject_level_r(
            frame,
            config=aggregation,
            undefined="zero",
        ).r

    if not np.isfinite(score):
        raise ValueError("Permutation statistic is undefined.")

    return score


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
    greater_is_better: bool = True,
    trial_indices: npt.NDArray[np.intp] | None = None,
    aggregation: AggregationConfig = _DEFAULT_AGGREGATION,
) -> NullResult:
    """Refit the whole pipeline under permuted targets and compare the observed statistic.

    ``greater_is_better`` picks the tail, and it is not cosmetic. The default suits
    a correlation or an R^2, where a good model scores high. An error metric --
    ``mean_squared_error``, ``mean_absolute_error`` -- scores *low* when the model
    is good, so leaving the default in place counts the wrong tail and returns
    ``p`` near 1 for a strong effect and a small ``p`` for a worthless one. Pass
    ``greater_is_better=False`` for any metric where smaller is better. There is no
    way to infer the direction from an arbitrary callable, so it has to be declared.

    ``observed`` must come from the same folds, model, seed, aggregation and metric
    as this call; it is recomputed and a mismatch is an error rather than a silently
    invalid p-value. ``residualize_on`` is refused: permuting raw targets and
    refitting the nuisance model does not give a nuisance-preserving conditional null.
    """
    if residualize_on:
        raise ValueError(
            "Nuisance-adjusted permutation inference requires a nuisance-preserving "
            "null procedure. This function permutes raw labels and does not support "
            "residualize_on; refitting nuisance regression after shuffling is insufficient."
        )
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

    if not np.isfinite(observed):
        raise ValueError("observed statistic must be finite.")

    def fit_targets(
        target_values: npt.NDArray[np.float64],
        fit_seed: int,
    ) -> tuple[FoldPrediction, ...]:
        return cross_fit_regression(
            folds,
            X,
            target_values,
            groups,
            pipeline,
            grid,
            inner=inner,
            seed=fit_seed,
            runs=runs,
            outer_n_jobs=outer_n_jobs,
            harmonization=harmonization,
            covariates=covariates,
            residualize_on=residualize_on,
            scoring=scoring,
            refit=refit,
        )

    observed_predictions = fit_targets(y, seed)
    recomputed_observed = _prediction_statistic(
        observed_predictions,
        groups_arr,
        null_aggregation,
        metric_fn,
    )

    if not np.isclose(
        observed,
        recomputed_observed,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "The supplied observed statistic differs from the statistic "
            "computed by this permutation procedure. Use the same folds, "
            "model, seed, aggregation, and metric for both."
        )

    null_scores: list[float] = []

    for b, y_perm in enumerate(permuted_targets):
        try:
            predictions = fit_targets(y_perm, seed)
            score = _prediction_statistic(
                predictions,
                groups_arr,
                null_aggregation,
                metric_fn,
            )
        except (FoldFitError, ValueError) as exc:
            raise RuntimeError(
                f"Permutation {b + 1} failed. No p-value is returned "
                "because dropping a failed permutation could alter "
                "the null distribution."
            ) from exc

        null_scores.append(score)

    null_arr = np.asarray(null_scores, dtype=np.float64)
    # "At least as extreme" means the tail the metric improves into.
    extreme = null_arr >= observed if greater_is_better else null_arr <= observed
    count_extreme = int(np.sum(extreme))
    p_value = float((count_extreme + 1) / (len(null_arr) + 1))

    return NullResult(
        p_value=p_value,
        observed=observed,
        null=null_arr,
        changed_fractions=changed_arr,
        n_incomplete=0,
    )
