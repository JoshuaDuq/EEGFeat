from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand_signal, window_mask
from eegfeat.baseline import normalize as _normalize
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

_MIN_BASELINE_POWER = 1e-12
"""Smallest baseline power that can anchor a ratio.

Deliberately not the ``1e-20`` power floor used elsewhere: a baseline that small
turns a quiet channel into an ERDS value of order 1e6 percent, which is
arithmetically valid and physically meaningless. The reference pipeline guards at
this value for the same reason, noting that clamping instead "would produce
artificially huge ERD/ERS ratios".
"""

ErdsScale = Literal["percent", "db"]

_UNITS: dict[str, dict[str, str]] = {
    "percent": {
        "mean": "%",
        "slope": "%/s",
        "erd_magnitude": "%",
        "erd_duration": "s",
        "ers_magnitude": "%",
        "ers_duration": "s",
        "peak_latency": "s",
        "onset_latency": "s",
        "rebound_latency": "s",
    },
    "db": {
        "mean": "dB",
        "slope": "dB/s",
        "erd_magnitude": "dB",
        "erd_duration": "s",
        "ers_magnitude": "dB",
        "ers_duration": "s",
        "peak_latency": "s",
        "onset_latency": "s",
        "rebound_latency": "s",
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
        ``ers_magnitude``, ``ers_duration``, ``peak_latency``, ``onset_latency``
        and ``rebound_latency``.
    """
    if normalize not in ("percent", "db"):
        raise ValueError(f"normalize must be 'percent' or 'db', got {normalize!r}.")

    thresholds: dict[int, npt.NDArray[np.float64]] = {}

    def trace_of(signal: BandSignal) -> npt.NDArray[np.float64]:
        trace, threshold = _trace(signal, baseline, normalize)
        thresholds[id(signal)] = threshold
        return trace

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        return _measures(signal, trace, times, thresholds[id(signal)])

    return expand_signal(
        signals,
        trace_of=trace_of,
        kernel=kernel,
        units=_UNITS[normalize],
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode=normalize,
    )


def _trace(
    signal: BandSignal, baseline: Window, mode: ErdsScale
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    power = signal.power
    mask = window_mask(signal.times, baseline)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        reference = np.nanmean(power[:, :, mask], axis=2)
        deviation = np.nanstd(power[:, :, mask], axis=2)
    # A baseline at the power floor cannot anchor a ratio; say so rather than
    # returning a number that is arithmetically valid and physically meaningless.
    reference = np.where(reference > _MIN_BASELINE_POWER, reference, np.nan)
    threshold = deviation / reference * 100.0
    return _normalize(power, baseline=reference, mode=mode), threshold


def _measures(
    signal: BandSignal,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    threshold: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    finite = np.isfinite(trace)
    usable: npt.NDArray[np.bool_] = np.asarray(finite.any(axis=2), dtype=np.bool_)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.where(usable, np.nanmean(trace, axis=2), np.nan)
    peak_index = _argmax_masked(np.abs(trace), finite)
    return {
        "mean": mean,
        "slope": _slope(trace, times, finite),
        "erd_magnitude": _signed_magnitude(trace, finite, usable, negative=True),
        "erd_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=True),
        "ers_magnitude": _signed_magnitude(trace, finite, usable, negative=False),
        "ers_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=False),
        "peak_latency": np.where(usable, times[peak_index], np.nan),
        "onset_latency": _onset(trace, times, finite, usable, threshold),
        "rebound_latency": _rebound(trace, times, finite, usable, peak_index),
    }


def _argmax_masked(
    values: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int_]:
    return np.asarray(np.argmax(np.where(finite, values, -np.inf), axis=2), dtype=np.int_)


def _onset(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    threshold: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    crossed = finite & (np.abs(trace) > threshold[:, :, np.newaxis])
    any_crossing: npt.NDArray[np.bool_] = np.asarray(crossed.any(axis=2), dtype=np.bool_)
    index = np.argmax(crossed, axis=2)
    return np.where(usable & any_crossing, times[index], np.nan)


def _rebound(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    finite: npt.NDArray[np.bool_],
    usable: npt.NDArray[np.bool_],
    peak_index: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    after = np.arange(trace.shape[2])[np.newaxis, np.newaxis, :] > peak_index[:, :, np.newaxis]
    eligible = finite & after
    any_eligible: npt.NDArray[np.bool_] = np.asarray(eligible.any(axis=2), dtype=np.bool_)
    index = np.argmax(np.where(eligible, trace, -np.inf), axis=2)
    return np.where(usable & any_eligible, times[index], np.nan)


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
