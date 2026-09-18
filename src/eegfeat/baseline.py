from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eegfeat.table import Normalization

EPS = 1e-20
"""Power floor applied to both sides of a ratio.

Flooring only the numerator would bias every value computed against a small
baseline, so both sides get the same floor.
"""


def normalize(
    values: npt.NDArray[np.float64],
    *,
    baseline: npt.NDArray[np.float64] | None,
    mode: Normalization,
) -> npt.NDArray[np.float64]:
    """Apply a normalization to per-channel power.

    Parameters
    ----------
    values : ndarray, shape (n_epochs, n_channels, n_windows)
        Raw power.
    baseline : ndarray, shape (n_epochs, n_channels), or None
        Baseline power, one value per epoch and channel. Required by
        ``"log_ratio"`` and ``"db"``, forbidden by the others.
    mode : {"raw", "log10", "log_ratio", "db"}
        ``"raw"`` returns the input, ``"log10"`` returns ``log10(p)``,
        ``"log_ratio"`` returns ``log10(p / b)`` and ``"db"`` returns
        ``10 * log_ratio``.

    Returns
    -------
    ndarray
        Normalized values, shaped like ``values``.
    """
    needs_baseline = mode in ("log_ratio", "db")
    if needs_baseline and baseline is None:
        raise ValueError(f"mode {mode!r} requires a baseline.")
    if not needs_baseline and baseline is not None:
        raise ValueError(f"mode {mode!r} takes no baseline.")

    if mode == "raw":
        return values
    floored = np.maximum(values, EPS)
    if mode == "log10":
        return np.log10(floored)

    assert baseline is not None  # narrowed by the guard above
    base = np.where(np.isfinite(baseline), np.maximum(baseline, EPS), np.nan)
    ratio = np.log10(floored / base[:, :, np.newaxis])
    return 10.0 * ratio if mode == "db" else ratio
