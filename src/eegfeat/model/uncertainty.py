from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

__all__ = [
    "Method",
    "PredictionIntervals",
    "prediction_intervals",
]

Method = Literal["split", "cv_plus", "quantile"]


@dataclass(frozen=True)
class PredictionIntervals:
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    alpha: float
    method: Method
    calibration_unit: Literal["trial", "subject"]


def _compute_conformal_quantile(residuals: npt.NDArray[np.float64], alpha: float) -> float:
    n_cal = len(residuals)
    if n_cal == 0:
        return 0.0
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    return float(np.quantile(residuals, q_level))


def _order_stat_quantile(values: npt.NDArray[np.float64], q: float, *, tail: str) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    q_clamped = float(np.clip(q, 0.0, 1.0))
    if tail == "upper":
        rank = int(np.ceil(q_clamped * arr.size)) - 1
    else:
        rank = int(np.floor(q_clamped * arr.size))
    rank = int(np.clip(rank, 0, arr.size - 1))
    return float(arr[rank])


def _split_conformal(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    alpha: float,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = len(X_train)
    if n < 5:
        msg = "Split conformal requires at least 5 training samples."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    if groups is None:
        n_cal = min(max(int(0.2 * n), 2), n - 2)
        indices = rng.permutation(n)
        cal_idx = indices[:n_cal]
        train_idx = indices[n_cal:]
    else:
        groups_arr = np.asarray(groups, dtype=object)
        unique_groups = [g for g in pd.unique(groups_arr) if not pd.isna(g)]
        if len(unique_groups) < 2:
            msg = (
                f"Group-aware split conformal requires at least 2 unique groups, "
                f"got {len(unique_groups)}."
            )
            raise ValueError(msg)
        n_cal_groups = min(max(1, int(round(0.2 * len(unique_groups)))), len(unique_groups) - 1)
        test_size = float(n_cal_groups / len(unique_groups))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        dummy_x = np.zeros((n, 1), dtype=np.float64)
        train_idx, cal_idx = next(splitter.split(dummy_x, y_train, groups=groups_arr))

    if len(train_idx) < 2 or len(cal_idx) < 2:
        msg = "Split conformal could not form valid train/calibration splits."
        raise ValueError(msg)

    model_proper = cast(Pipeline, clone(model))
    model_proper.fit(X_train[train_idx], y_train[train_idx])

    residuals = np.abs(y_train[cal_idx] - model_proper.predict(X_train[cal_idx]))
    residuals = residuals[np.isfinite(residuals)]
    q_hat = _compute_conformal_quantile(residuals, alpha)

    y_test_pred = np.asarray(model_proper.predict(X_test), dtype=np.float64)
    return y_test_pred - q_hat, y_test_pred + q_hat


def _get_cv_splits(
    cv_splits: int,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    n = len(X_train)
    if groups is not None:
        groups_arr = np.asarray(groups, dtype=object)
        unique_groups = [g for g in pd.unique(groups_arr) if not pd.isna(g)]
        n_unique = len(unique_groups)
        if n_unique >= cv_splits:
            splitter = GroupKFold(n_splits=cv_splits)
            splits = splitter.split(X_train, y_train, groups=groups_arr)
        else:
            splitter_logo = LeaveOneGroupOut()
            splits = splitter_logo.split(X_train, y_train, groups=groups_arr)
    else:
        if cv_splits > n:
            msg = f"Cannot have cv_splits={cv_splits} greater than n_samples={n}."
            raise ValueError(msg)
        splitter_kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        splits = splitter_kf.split(X_train)

    return [(tr.astype(np.intp), val.astype(np.intp)) for tr, val in splits]


def _conformal_cv_plus(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    alpha: float,
    cv_splits: int,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    splits = _get_cv_splits(cv_splits, seed, groups, X_train, y_train)
    n_test = len(X_test)
    lower_chunks: list[npt.NDArray[np.float64]] = []
    upper_chunks: list[npt.NDArray[np.float64]] = []

    for train_idx, val_idx in splits:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model_fold = cast(Pipeline, clone(model))
        model_fold.fit(X_train[train_idx], y_train[train_idx])

        val_preds = np.asarray(model_fold.predict(X_train[val_idx]), dtype=np.float64)
        residuals = np.abs(y_train[val_idx] - val_preds)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            continue

        test_preds = np.asarray(model_fold.predict(X_test), dtype=np.float64)
        lower_chunks.append(test_preds[:, None] - residuals[None, :])
        upper_chunks.append(test_preds[:, None] + residuals[None, :])

    if not lower_chunks or not upper_chunks:
        msg = "CV+ calibration failed: no valid fold calibration chunks were produced."
        raise ValueError(msg)

    lower_candidates = np.concatenate(lower_chunks, axis=1)
    upper_candidates = np.concatenate(upper_chunks, axis=1)

    lower = np.full(n_test, np.nan, dtype=np.float64)
    upper = np.full(n_test, np.nan, dtype=np.float64)
    for i in range(n_test):
        lower[i] = _order_stat_quantile(lower_candidates[i, :], alpha, tail="lower")
        upper[i] = _order_stat_quantile(upper_candidates[i, :], 1.0 - alpha, tail="upper")

    return lower, upper


def _conformal_quantile(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    alpha: float,
    cv_splits: int,
    seed: int,
    groups: npt.NDArray[np.object_] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = len(X_train)
    alpha_lo = alpha / 2.0
    alpha_hi = 1.0 - alpha / 2.0
    splits = _get_cv_splits(cv_splits, seed, groups, X_train, y_train)

    loo_lower = np.zeros(n, dtype=np.float64)
    loo_upper = np.zeros(n, dtype=np.float64)

    for train_idx, val_idx in splits:
        qr_low = GradientBoostingRegressor(loss="quantile", alpha=alpha_lo, random_state=seed)
        qr_low.fit(X_train[train_idx], y_train[train_idx])
        loo_lower[val_idx] = qr_low.predict(X_train[val_idx])

        qr_high = GradientBoostingRegressor(loss="quantile", alpha=alpha_hi, random_state=seed)
        qr_high.fit(X_train[train_idx], y_train[train_idx])
        loo_upper[val_idx] = qr_high.predict(X_train[val_idx])

    e_lo = loo_lower - y_train
    e_hi = y_train - loo_upper
    scores = np.maximum(e_lo, e_hi)
    scores = scores[np.isfinite(scores)]
    q_hat = _compute_conformal_quantile(scores, alpha)

    qr_low_full = GradientBoostingRegressor(loss="quantile", alpha=alpha_lo, random_state=seed)
    qr_low_full.fit(X_train, y_train)

    qr_high_full = GradientBoostingRegressor(loss="quantile", alpha=alpha_hi, random_state=seed)
    qr_high_full.fit(X_train, y_train)

    lower = np.asarray(qr_low_full.predict(X_test) - q_hat, dtype=np.float64)
    upper = np.asarray(qr_high_full.predict(X_test) + q_hat, dtype=np.float64)
    return lower, upper


def prediction_intervals(
    model: Pipeline,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    *,
    alpha: float = 0.1,
    method: Method = "cv_plus",
    cv_splits: int = 5,
    seed: int = 42,
    groups: npt.NDArray[np.object_] | None = None,
) -> PredictionIntervals:
    """Compute conformal prediction intervals marginal over the calibration unit."""
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be between 0.0 and 1.0, got {alpha}."
        raise ValueError(msg)

    valid_methods: tuple[str, ...] = ("split", "cv_plus", "quantile")
    if method not in valid_methods:
        msg = f"Unknown method {method!r}. Expected one of: {valid_methods}."
        raise ValueError(msg)

    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)

    if len(X_tr) == 0 or len(y_tr) == 0 or len(X_te) == 0:
        msg = "Training and test arrays must not be empty."
        raise ValueError(msg)

    cal_unit: Literal["trial", "subject"] = "subject" if groups is not None else "trial"

    if method == "split":
        lower, upper = _split_conformal(model, X_tr, y_tr, X_te, alpha, seed, groups)
    elif method == "cv_plus":
        lower, upper = _conformal_cv_plus(model, X_tr, y_tr, X_te, alpha, cv_splits, seed, groups)
    else:
        lower, upper = _conformal_quantile(
            model, X_tr, y_tr, X_te, alpha, cv_splits, seed, groups
        )

    return PredictionIntervals(
        lower=lower,
        upper=upper,
        alpha=alpha,
        method=method,
        calibration_unit=cal_unit,
    )
