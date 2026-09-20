from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from eegfeat.model.scoring import safe_pearsonr

__all__ = [
    "AggregationConfig",
    "SubjectLevelR",
    "bootstrap_mean_ci",
    "fold_results",
    "paired_signflip_p_value",
    "subject_level_errors",
    "subject_level_r",
]


@dataclass(frozen=True)
class AggregationConfig:
    subject_weighting: Literal["equal", "trial_count"] = "equal"
    bootstrap_iterations: int = 10_000
    ci_method: Literal["fixed_effects", "bootstrap"] = "fixed_effects"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.subject_weighting not in ("equal", "trial_count"):
            raise ValueError(
                f"subject_weighting must be 'equal' or 'trial_count', got {self.subject_weighting}"
            )
        if self.ci_method not in ("fixed_effects", "bootstrap"):
            raise ValueError(
                f"ci_method must be 'fixed_effects' or 'bootstrap', got {self.ci_method!r}"
            )
        if self.bootstrap_iterations <= 0:
            raise ValueError(f"bootstrap_iterations must be > 0, got {self.bootstrap_iterations}")


_DEFAULT_CONFIG = AggregationConfig()


@dataclass(frozen=True)
class SubjectLevelR:
    r: float
    per_subject: tuple[tuple[str, float], ...]
    ci_low: float
    ci_high: float


def bootstrap_mean_ci(
    values: npt.NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        v = float(vals[0])
        return v, v

    rng = np.random.default_rng(seed)
    n = len(vals)
    boot_means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means[i] = float(np.mean(sample))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_signflip_p_value(
    differences: npt.NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> float:
    vals = np.asarray(differences, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or iterations <= 0:
        return np.nan

    observed = float(abs(np.mean(vals)))
    rng = np.random.default_rng(seed)
    n = len(vals)
    count = 0
    for _ in range(iterations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
        if float(abs(np.mean(vals * signs))) >= observed:
            count += 1
    return float((count + 1) / (iterations + 1))


def fold_results(
    results: Sequence[object],
    groups: npt.NDArray[np.object_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str], list[int], list[int]]:
    def _get_fold(r: object) -> int:
        if hasattr(r, "fold"):
            return int(getattr(r, "fold"))  # noqa: B009
        if isinstance(r, dict):
            return int(str(r["fold"]))
        raise TypeError(f"Unsupported fold result record type: {type(r)}")

    sorted_results = sorted(results, key=_get_fold)
    # The design's groups label rows by index; without them each record labels itself, and
    # a leave-one-subject-out prediction has no subject of its own to offer.
    group_source = None if groups is None else np.asarray(groups, dtype=object)

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    groups_ordered: list[str] = []
    test_indices: list[int] = []
    fold_ids: list[int] = []
    labelled: list[bool] = []

    for record in sorted_results:
        fold = _get_fold(record)
        raw_groups: Sequence[object] | npt.NDArray[np.object_] | None
        if hasattr(record, "y_true") and hasattr(record, "y_pred"):
            raw_true = getattr(record, "y_true")  # noqa: B009
            raw_pred = getattr(record, "y_pred")  # noqa: B009
            raw_test = getattr(record, "rows", None)
            sub = getattr(record, "subject", None)
            raw_groups = [sub] * len(raw_true) if sub is not None else None
        elif isinstance(record, dict):
            raw_true = record["y_true"]
            raw_pred = record["y_pred"]
            raw_test = record.get("test_idx", record.get("rows"))
            raw_groups = record.get("groups")
        else:
            raise TypeError(f"Unsupported record type: {type(record)}")

        yt = [float(v) for v in raw_true]
        yp = [float(v) for v in raw_pred]
        if len(yt) != len(yp):
            msg = f"Fold {fold} has mismatched lengths: y_true ({len(yt)}) vs y_pred ({len(yp)})."
            raise ValueError(msg)

        if raw_test is not None:
            t_idx = [int(i) for i in raw_test]
            if len(t_idx) != len(yt):
                msg = (
                    f"Fold {fold} has mismatched test_idx length ({len(t_idx)}) "
                    f"vs y_true ({len(yt)})."
                )
                raise ValueError(msg)
        else:
            t_idx = []

        if group_source is not None and raw_test is not None:
            raw_groups = group_source[np.asarray(t_idx, dtype=np.intp)]
        if raw_groups is not None:
            g_arr = [str(g) for g in raw_groups]
            if len(g_arr) != len(yt):
                msg = (
                    f"Fold {fold} has mismatched groups length ({len(g_arr)}) "
                    f"vs y_true ({len(yt)})."
                )
                raise ValueError(msg)
            groups_ordered.extend(g_arr)
        labelled.append(raw_groups is not None)

        y_true_all.extend(yt)
        y_pred_all.extend(yp)
        test_indices.extend(t_idx)
        fold_ids.extend([fold] * len(yt))

    if any(labelled) and not all(labelled):
        msg = (
            "Some fold results carry subject labels and others do not, so the groups could "
            "not be aligned with y_true; pass groups= to label every row from the design."
        )
        raise ValueError(msg)

    return (
        np.asarray(y_true_all, dtype=float),
        np.asarray(y_pred_all, dtype=float),
        groups_ordered,
        test_indices,
        fold_ids,
    )


def subject_level_r(
    predictions: pd.DataFrame,
    *,
    config: AggregationConfig = _DEFAULT_CONFIG,
    undefined: Literal["raise", "zero"] = "raise",
) -> SubjectLevelR:
    if undefined not in ("raise", "zero"):
        raise ValueError(f"undefined must be 'raise' or 'zero', got {undefined!r}")
    per_subject: list[tuple[str, float]] = []
    valid_entries: list[tuple[float, int]] = []
    invalid_subjects: list[str] = []

    for subj, df_sub in predictions.groupby("subject_id"):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(yt) & np.isfinite(yp)
        n_trials = int(finite.sum())

        if n_trials < 3:
            invalid_subjects.append(f"{subj}: fewer than 3 finite predictions (got {n_trials})")
            continue

        r, _ = safe_pearsonr(yt[finite], yp[finite])
        if undefined == "zero" and not np.isfinite(r):
            # Nothing varies to correlate with, which is no linear association.
            r = 0.0
        per_subject.append((str(subj), float(r)))
        if np.isfinite(r):
            valid_entries.append((float(r), n_trials))
        else:
            invalid_subjects.append(f"{subj}: non-finite correlation (degenerate subject)")

    if invalid_subjects:
        details = "; ".join(invalid_subjects)
        raise ValueError(f"Invalid subject-level correlation inputs: {details}")

    if not valid_entries:
        return SubjectLevelR(
            r=np.nan,
            per_subject=tuple(per_subject),
            ci_low=np.nan,
            ci_high=np.nan,
        )

    r_vals = np.array([r for r, _ in valid_entries], dtype=float)
    n_vals = np.array([n for _, n in valid_entries], dtype=int)

    # Correlations are averaged in Fisher z rather than in r because r is not additive.
    clipped_r = np.clip(r_vals, -0.999999, 0.999999)
    z_vals = np.arctanh(clipped_r)

    if config.subject_weighting == "trial_count":
        # var(z) = 1/(n-3), so a 3-trial subject has infinite variance and no weight. Flooring
        # it at 1.0 would give it the same weight as a 4-trial subject, fabricating precision
        # the data does not have and making the fixed-effects interval far too narrow.
        underpowered = int(np.sum(n_vals < 4))
        if underpowered:
            raise ValueError(
                f"Trial-count weighting needs more than 3 trials per subject for the Fisher-z "
                f"variance to be defined; {underpowered} of {len(n_vals)} subject(s) have 3 or "
                "fewer. Use subject_weighting='equal' or drop those subjects."
            )
        weights = n_vals - 3.0
    else:
        weights = np.ones_like(z_vals, dtype=float)

    sum_weights = float(np.sum(weights))
    norm_weights = weights / sum_weights
    mean_z = float(np.average(z_vals, weights=norm_weights))
    agg_r = float(np.tanh(mean_z))

    ci_low, ci_high = np.nan, np.nan
    # Falling through to the fixed-effects branch would hand back a different estimator than
    # the one asked for, with nothing on the result to say so.
    if config.ci_method == "bootstrap" and 1 < len(z_vals) < 3:
        raise ValueError(
            f"Bootstrap confidence intervals need at least 3 subjects, got {len(z_vals)}. "
            "Use ci_method='fixed_effects'."
        )

    if config.ci_method == "bootstrap" and len(z_vals) >= 3:
        rng = np.random.default_rng(config.seed)
        n_sub = len(z_vals)
        boot_means = np.empty(config.bootstrap_iterations, dtype=float)
        for i in range(config.bootstrap_iterations):
            idx = rng.choice(n_sub, size=n_sub, replace=True)
            boot_z = z_vals[idx]
            if config.subject_weighting == "trial_count":
                boot_w = weights[idx]
                boot_means[i] = float(np.average(boot_z, weights=boot_w / np.sum(boot_w)))
            else:
                boot_means[i] = float(np.mean(boot_z))
        ci_low = float(np.tanh(np.percentile(boot_means, 2.5)))
        ci_high = float(np.tanh(np.percentile(boot_means, 97.5)))
    elif len(z_vals) > 1:
        if config.subject_weighting == "trial_count":
            # Known Fisher-z variance, so the normal quantile is the right multiplier.
            se = float(np.sqrt(1.0 / sum_weights))
            multiplier = 1.96
        else:
            # The between-subject SD is estimated from the subjects themselves, so the interval
            # needs Student's t. At 5 subjects the normal quantile gives a nominal 95% interval
            # that covers about 88%.
            se = float(np.std(z_vals, ddof=1) / np.sqrt(len(z_vals)))
            multiplier = float(stats.t.ppf(0.975, len(z_vals) - 1))
        if np.isfinite(se) and se > 0:
            delta = multiplier * se
            ci_low = float(np.tanh(mean_z - delta))
            ci_high = float(np.tanh(mean_z + delta))

    return SubjectLevelR(
        r=agg_r,
        per_subject=tuple(per_subject),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _weighted_mean(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    return float(np.average(values, weights=weights / np.sum(weights)))


def _weighted_bootstrap_ci(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    # Subjects are resampled, carrying their weights, so the interval reflects the same
    # weighting as the point estimate.
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.choice(n, size=n, replace=True)
        means[i] = _weighted_mean(values[idx], weights[idx])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def subject_level_errors(
    predictions: pd.DataFrame,
    *,
    config: AggregationConfig = _DEFAULT_CONFIG,
) -> dict[str, float]:
    per_subject_mae: list[float] = []
    per_subject_rmse: list[float] = []
    per_subject_n: list[int] = []
    invalid_subjects: list[str] = []

    for subj, df_sub in predictions.groupby("subject_id"):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(yt) & np.isfinite(yp)
        n_trials = int(finite.sum())
        if n_trials < 1:
            invalid_subjects.append(f"{subj}: no finite predictions")
            continue
        err = yp[finite] - yt[finite]
        per_subject_mae.append(float(np.mean(np.abs(err))))
        per_subject_rmse.append(float(np.sqrt(np.mean(err**2))))
        per_subject_n.append(n_trials)

    if invalid_subjects:
        details = "; ".join(invalid_subjects)
        raise ValueError(f"Invalid subject-level error inputs: {details}")

    if not per_subject_mae:
        return {
            "mean_mae": np.nan,
            "mean_rmse": np.nan,
            "ci_low_mae": np.nan,
            "ci_high_mae": np.nan,
            "ci_low_rmse": np.nan,
            "ci_high_rmse": np.nan,
        }

    maes = np.array(per_subject_mae, dtype=float)
    rmses = np.array(per_subject_rmse, dtype=float)
    # One config must not weight the correlation by trial count while leaving the errors
    # equal-weighted; subject_level_r honours this field, so these have to as well.
    counts = np.array(per_subject_n, dtype=float)
    weights = counts if config.subject_weighting == "trial_count" else np.ones_like(counts)

    mean_mae = _weighted_mean(maes, weights)
    mean_rmse = _weighted_mean(rmses, weights)

    ci_low_mae, ci_high_mae = np.nan, np.nan
    ci_low_rmse, ci_high_rmse = np.nan, np.nan

    if config.ci_method == "bootstrap" and len(maes) >= 3:
        ci_low_mae, ci_high_mae = _weighted_bootstrap_ci(
            maes, weights, iterations=config.bootstrap_iterations, seed=config.seed
        )
        ci_low_rmse, ci_high_rmse = _weighted_bootstrap_ci(
            rmses, weights, iterations=config.bootstrap_iterations, seed=config.seed
        )

    return {
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "ci_low_mae": ci_low_mae,
        "ci_high_mae": ci_high_mae,
        "ci_low_rmse": ci_low_rmse,
        "ci_high_rmse": ci_high_rmse,
    }
