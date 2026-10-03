from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eegtable.table import Normalization

FLOOR_FRACTION = 1e-12
"""Power floor, as a fraction of the largest power the same cell reaches.

The floor exists so a zero takes a logarithm and anchors a ratio. It is relative
because absolute power is a property of the unit, not of the data being valid:
EEG sits near 1e-12 V², an eLORETA source estimate in A·m near 1e-21, and a
fixed floor of 1e-20 turned every ratio of the latter into 1. Twelve decades
under the cell's own peak is a dropout at any unit, never a measurement.

The peak is taken within one epoch and channel, so no trial's floor depends on
another trial's power. Both sides of a ratio get the same floor, because
flooring only the numerator would bias every value computed against a small
baseline.
"""


def power_floor(peak: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Floor under power whose largest finite value is ``peak``, NaN where that is not positive."""
    # With no positive power there is no scale to floor against, so any finite
    # value would be the floor's choice rather than a measurement.
    return np.where(peak > 0.0, peak * FLOOR_FRACTION, np.nan)


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
        ``"log_ratio"``, ``"db"`` and ``"percent"``, forbidden by the others.
    mode : {"raw", "log10", "log_ratio", "db", "percent"}
        ``"raw"`` returns the input, ``"log10"`` returns ``log10(p)``,
        ``"log_ratio"`` returns ``log10(p / b)``, ``"db"`` returns
        ``10 * log_ratio``, and ``"percent"`` returns ``(p - b) / b * 100``.

    Returns
    -------
    ndarray
        Normalized values, shaped like ``values``. Power is floored at
        :data:`FLOOR_FRACTION` of the largest finite power of its epoch and
        channel, over every window and the baseline. An epoch and channel with
        no positive power is NaN.
    """
    needs_baseline = mode in ("log_ratio", "db", "percent")
    if needs_baseline and baseline is None:
        raise ValueError(f"mode {mode!r} requires a baseline.")
    if not needs_baseline and baseline is not None:
        raise ValueError(f"mode {mode!r} takes no baseline.")

    if mode == "raw":
        return values
    peak = np.max(np.where(np.isfinite(values), values, -np.inf), axis=2, initial=-np.inf)
    if baseline is not None:
        peak = np.maximum(peak, np.where(np.isfinite(baseline), baseline, -np.inf))
    floor = power_floor(peak)
    floored = np.maximum(values, floor[:, :, np.newaxis])
    if mode == "log10":
        return np.log10(floored)

    assert baseline is not None  # narrowed by the guard above
    base = np.where(np.isfinite(baseline), np.maximum(baseline, floor), np.nan)
    if mode == "percent":
        # Not floored in the numerator: a genuine zero is a real 100% decrease.
        return (values - base[:, :, np.newaxis]) / base[:, :, np.newaxis] * 100.0
    ratio = np.log10(floored / base[:, :, np.newaxis])
    return 10.0 * ratio if mode == "db" else ratio
