from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

__all__ = [
    "make_pearsonr_scorer",
    "pearsonr_scorer",
    "safe_pearsonr",
    "scoring_dict",
]


def safe_pearsonr(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    min_variance: float = 1e-10,
) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return np.nan, np.nan

    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(valid.sum()) < 2:
        return np.nan, np.nan

    x_valid, y_valid = x_arr[valid], y_arr[valid]
    var_x = float(np.var(x_valid, ddof=1))
    var_y = float(np.var(y_valid, ddof=1))
    if var_x < min_variance or var_y < min_variance:
        return np.nan, np.nan

    r, p = pearsonr(x_valid, y_valid)
    if not (np.isfinite(r) and np.isfinite(p)):
        return np.nan, np.nan

    # Guard against floating-point overshoot past the theoretical bound [-1, 1].
    return float(np.clip(r, -1.0, 1.0)), float(p)


def pearsonr_scorer() -> Callable[..., float]:
    scorer = make_scorer(
        lambda yt, yp: safe_pearsonr(np.asarray(yt, dtype=float), np.asarray(yp, dtype=float))[0],
        greater_is_better=True,
    )
    return cast(Callable[..., float], scorer)


make_pearsonr_scorer = pearsonr_scorer


def scoring_dict() -> dict[str, object]:
    return {"r": pearsonr_scorer(), "neg_mse": "neg_mean_squared_error"}
