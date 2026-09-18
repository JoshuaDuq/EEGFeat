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
arithmetically valid and physically meaningless. Baselines below this floor
are invalidated to NaN to prevent artificially inflated ERD/ERS ratios.
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


def erds_mean(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Mean of the ERDS trace over the window.

    The conventional summary: negative is desynchronization, positive is synchronization.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "mean",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_slope(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Least-squares slope of the ERDS trace against time.

    Positive means the response is recovering across the window, negative that it is deepening.
    NaN with fewer than three finite samples.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "slope",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erd_magnitude(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Mean depth of the desynchronized part of the trace.

    The mean of ``abs(trace)`` over samples below zero. Exactly ``0.0`` when no sample is
    negative: no desynchronization is a measurement, not a missing value.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erd_magnitude",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erd_duration(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Time spent desynchronized.

    Count of samples below zero divided by the sampling rate. Exactly ``0.0`` when none are.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erd_duration",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def ers_magnitude(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Mean height of the synchronized part of the trace.

    The mean of the trace over samples above zero, and ``0.0`` when none are.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "ers_magnitude",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def ers_duration(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Time spent synchronized.

    Count of samples above zero divided by the sampling rate, and ``0.0`` when none are.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "ers_duration",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_peak_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Time of the largest excursion from baseline, in either direction.

    Located by ``argmax(abs(trace))``, so a deep desynchronization outranks a shallower
    synchronization.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "peak_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_onset_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Time the trace first leaves the baseline's own variability.

    The first sample where ``abs(trace)`` exceeds the baseline coefficient of variation in
    percent, so a noisy baseline demands a correspondingly larger excursion. A first crossing,
    not a sustained one. NaN when the trace never crosses.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "onset_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def erds_rebound_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "percent",
) -> FeatureTable:
    """Latency of the largest ERDS excursion after the peak.

    Evaluates to NaN when no sample follows the peak latency within the analysis window.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    baseline : Window
        Window each trial is referenced to. Required: ERDS without a baseline is
        not a defined quantity.
    windows : sequence of Window
        Analysis windows to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    normalize : {"percent", "db"}, default "percent"
        Percent change from baseline, or decibels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "rebound_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
    )


def _erds_measure(
    signals: Sequence[BandSignal],
    measure: str,
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    normalize: ErdsScale,
) -> FeatureTable:
    if normalize not in ("percent", "db"):
        raise ValueError(f"normalize must be 'percent' or 'db', got {normalize!r}.")

    def trace_of(signal: BandSignal) -> npt.NDArray[np.float64]:
        return _trace(signal, baseline, normalize)[0]

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        threshold = _baseline_threshold(signal, baseline)
        # Every measure derives from the same trace, so computing the set and
        # taking one is cheaper than it looks and keeps the definitions together.
        return {measure: _measures(signal, trace, times, threshold)[measure]}

    return expand_signal(
        signals,
        trace_of=trace_of,
        kernel=kernel,
        units={measure: _UNITS[normalize][measure]},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode=normalize,
    )


def _baseline_threshold(signal: BandSignal, baseline: Window) -> npt.NDArray[np.float64]:
    # The baseline's coefficient of variation in percent: a noisy baseline demands
    # a larger excursion before onset is declared.
    mask = window_mask(signal.times, baseline)
    power = signal.power[:, :, mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        reference = np.nanmean(power, axis=2)
        deviation = np.nanstd(power, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(reference > _MIN_BASELINE_POWER, deviation / reference * 100.0, np.nan)


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
