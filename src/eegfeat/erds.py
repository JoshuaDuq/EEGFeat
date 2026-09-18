from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand_signal, window_mask
from eegfeat.baseline import EPS
from eegfeat.baseline import normalize as _normalize
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

ErdsScale = Literal["percent", "db"]

_UNITS: dict[str, dict[str, str]] = {
    "percent": {
        "mean": "%",
        "slope": "%/s",
        "erd_magnitude": "%",
        "erd_duration": "s",
        "ers_magnitude": "%",
        "ers_duration": "s",
    },
    "db": {
        "mean": "dB",
        "slope": "dB/s",
        "erd_magnitude": "dB",
        "erd_duration": "s",
        "ers_magnitude": "dB",
        "ers_duration": "s",
    },
}


def erds(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Event-related desynchronization and synchronization.

    Expresses band power in each analysis window relative to a baseline window,
    then summarizes the resulting trace. Both baseline statistics are computed per
    epoch and per channel, so every trial is referenced to its own pre-stimulus
    power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window whose mean power each trial is referenced to. Required: ERDS
        without a baseline is not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels. Percent is the reference
        pipeline's default.

    Returns
    -------
    FeatureTable
        Columns for ``mean``, ``slope``, ``erd_magnitude``, ``erd_duration``,
        ``ers_magnitude`` and ``ers_duration``.
    """
    if normalize not in ("percent", "db"):
        raise ValueError(f"normalize must be 'percent' or 'db', got {normalize!r}.")

    def trace_of(signal: BandSignal) -> npt.NDArray[np.float64]:
        return _trace(signal, baseline, normalize)

    return expand_signal(
        signals,
        trace_of=trace_of,
        kernel=_measures,
        units=_UNITS[normalize],
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode=normalize,
    )


def _trace(signal: BandSignal, baseline: Window, mode: ErdsScale) -> npt.NDArray[np.float64]:
    power = signal.power
    mask = window_mask(signal.times, baseline)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        reference = np.nanmean(power[:, :, mask], axis=2)
    # A baseline at the power floor cannot anchor a ratio; say so rather than
    # returning a number that is arithmetically valid and physically meaningless.
    reference = np.where(reference > EPS, reference, np.nan)
    return _normalize(power, baseline=reference, mode=mode)


def _measures(
    signal: BandSignal,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    finite = np.isfinite(trace)
    usable: npt.NDArray[np.bool_] = np.asarray(finite.any(axis=2), dtype=np.bool_)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.where(usable, np.nanmean(trace, axis=2), np.nan)
    return {
        "mean": mean,
        "slope": _slope(trace, times, finite),
        "erd_magnitude": _signed_magnitude(trace, finite, usable, negative=True),
        "erd_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=True),
        "ers_magnitude": _signed_magnitude(trace, finite, usable, negative=False),
        "ers_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=False),
    }


def _slope(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    # Ordinary least squares in closed form, so it vectorizes over epochs and
    # channels whose finite samples differ.
    count = finite.sum(axis=2).astype(float)
    t = np.where(finite, times, 0.0)
    y = np.where(finite, trace, 0.0)
    sum_t, sum_y = t.sum(axis=2), y.sum(axis=2)
    denominator = count * (t * t).sum(axis=2) - sum_t**2
    numerator = count * (t * y).sum(axis=2) - sum_t * sum_y
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((count > 2) & (denominator != 0.0), numerator / denominator, np.nan)


def _signed_magnitude(
    trace: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    *,
    negative: bool,
) -> npt.NDArray[np.float64]:
    selected = finite & (trace < 0.0 if negative else trace > 0.0)
    total = np.where(selected, np.abs(trace) if negative else trace, 0.0).sum(axis=2)
    count = selected.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        magnitude = np.where(count > 0, total / count, 0.0)
    # Absent is zero; unmeasurable is NaN. They are different statements.
    return np.where(usable, magnitude, np.nan)


def _signed_duration(
    trace: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    sfreq: float,
    *,
    negative: bool,
) -> npt.NDArray[np.float64]:
    selected = finite & (trace < 0.0 if negative else trace > 0.0)
    return np.where(usable, selected.sum(axis=2) / sfreq, np.nan)
