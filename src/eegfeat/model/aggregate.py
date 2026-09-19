from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegfeat.model.scoring import safe_pearsonr


@dataclass(frozen=True)
class AggregationConfig:
    subject_weighting: Literal["equal", "trial_count"] = "equal"
    bootstrap_iterations: int = 10_000
    ci_method: Literal["fixed_effects", "bootstrap"] = "fixed_effects"

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
            raise ValueError(
                f"bootstrap_iterations must be > 0, got {self.bootstrap_iterations}"
            )


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


def fold_results(results: Sequence[dict[str, object]]) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], list[str], list[int], list[int]
]:
    sorted_results = sorted(results, key=lambda r: int(str(r["fold"])))

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    groups_ordered: list[str] = []
    test_indices: list[int] = []
    fold_ids: list[int] = []

    for record in sorted_results:
        raw_true = record["y_true"]
        raw_pred = record["y_pred"]
        raw_groups = record["groups"]
        raw_test = record["test_idx"]
        fold = int(str(record["fold"]))

        if isinstance(raw_true, Sequence | np.ndarray):
            y_true_all.extend(float(v) for v in raw_true)
        if isinstance(raw_pred, Sequence | np.ndarray):
            y_pred_all.extend(float(v) for v in raw_pred)
        if isinstance(raw_groups, Sequence | np.ndarray):
            groups_ordered.extend(str(g) for g in raw_groups)
        if isinstance(raw_test, Sequence | np.ndarray):
            test_indices.extend(int(i) for i in raw_test)
            fold_ids.extend([fold] * len(raw_test))

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
) -> SubjectLevelR:
    per_subject: list[tuple[str, float]] = []
    valid_entries: list[tuple[float, int]] = []
    invalid_subjects: list[str] = []

    for subj, df_sub in predictions.groupby("subject_id"):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(yt) & np.isfinite(yp)
        n_trials = int(finite.sum())

        if n_trials < 2:
            invalid_subjects.append(f"{subj}: fewer than 2 finite predictions")
            continue

        r, _ = safe_pearsonr(yt[finite], yp[finite])
        per_subject.append((str(subj), float(r)))
        if np.isfinite(r):
            valid_entries.append((float(r), n_trials))

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
    clipped_r = np.clip(r_vals, -1.0 + 1e-15, 1.0 - 1e-15)
    z_vals = np.arctanh(clipped_r)

    if config.subject_weighting == "trial_count":
        weights = np.maximum(n_vals - 3.0, 1.0)
    else:
        weights = np.ones_like(z_vals, dtype=float)

    sum_weights = float(np.sum(weights))
    norm_weights = weights / sum_weights
    mean_z = float(np.average(z_vals, weights=norm_weights))
    agg_r = float(np.tanh(mean_z))

    ci_low, ci_high = np.nan, np.nan
    if config.ci_method == "bootstrap" and len(z_vals) >= 3:
        rng = np.random.default_rng(42)
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
            se = float(np.sqrt(1.0 / sum_weights))
        else:
            se = float(np.std(z_vals, ddof=1) / np.sqrt(len(z_vals)))
        if np.isfinite(se) and se > 0:
            delta = 1.96 * se
            ci_low = float(np.tanh(mean_z - delta))
            ci_high = float(np.tanh(mean_z + delta))

    return SubjectLevelR(
        r=agg_r,
        per_subject=tuple(per_subject),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def subject_level_errors(
    predictions: pd.DataFrame,
    *,
    config: AggregationConfig = _DEFAULT_CONFIG,
) -> dict[str, float]:
    per_subject_mae: list[float] = []
    per_subject_rmse: list[float] = []
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
    mean_mae = float(np.mean(maes))
    mean_rmse = float(np.mean(rmses))

    ci_low_mae, ci_high_mae = np.nan, np.nan
    ci_low_rmse, ci_high_rmse = np.nan, np.nan

    if config.ci_method == "bootstrap" and len(maes) >= 3:
        ci_low_mae, ci_high_mae = bootstrap_mean_ci(
            maes, iterations=config.bootstrap_iterations, seed=42
        )
        ci_low_rmse, ci_high_rmse = bootstrap_mean_ci(
            rmses, iterations=config.bootstrap_iterations, seed=42
        )

    return {
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "ci_low_mae": ci_low_mae,
        "ci_high_mae": ci_high_mae,
        "ci_low_rmse": ci_low_rmse,
        "ci_high_rmse": ci_high_rmse,
    }
