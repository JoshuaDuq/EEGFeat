from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from eegfeat._expand import SignalKernel, expand_signal
from eegfeat.signal import TimeSeries
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

Polarity = Literal["positive", "negative", "absolute"]

_SEARCH = {"positive": 1.0, "negative": -1.0}


def variance(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Variance of the signal within each window.

    Computed over finite samples within each window, with valid sample fractions
    recorded in the parallel ``coverage`` matrix.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes. The bands axis, if any, comes from this
        sequence.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Variance in the squared units of the input.
    """
    return _measure(series, "variance", "V^2", _variance_kernel, windows, groups, include_global)


def peak_to_peak(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Range of the signal within each window, maximum minus minimum.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak-to-peak amplitude in the units of the input.
    """
    return _measure(series, "ptp", "V", _ptp_kernel, windows, groups, include_global)


def mean_amplitude(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean of the signal within each window.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Mean amplitude in the units of the input.
    """
    return _measure(series, "mean_amplitude", "V", _mean_kernel, windows, groups, include_global)


def area_under_curve(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Signed area under the signal within each window.

    Integrated by the trapezoid rule over each contiguous run of finite samples,
    and summed. A gap is skipped rather than interpolated across, so a stretch of
    missing data contributes nothing instead of contributing a straight line.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Area in the units of the input times seconds.
    """
    return _measure(series, "auc", "V*s", _auc_kernel, windows, groups, include_global)


def peak_amplitude(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    polarity: Polarity = "absolute",
    prominence: float | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Signed value of the extremum within each window.

    ``polarity`` is specified as an explicit parameter (``"positive"``,
    ``"negative"``, or ``"absolute"``), independent of window labels or naming conventions.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    polarity : {"absolute", "positive", "negative"}, default "absolute"
        Whether to find the largest value, the most negative, or the largest
        excursion in either direction. The value returned is always signed.
    prominence : float, optional
        When given, the extremum is chosen among prominent local peaks rather than
        by a plain extremum, which is more stable on noisy traces. Falls back to
        the plain extremum when no peak meets the requirement.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak amplitude in the units of the input.
    """
    return _peak(
        series, "peak_amplitude", "V", windows, polarity, prominence, groups, include_global
    )


def peak_latency(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    polarity: Polarity = "absolute",
    prominence: float | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Time of the extremum within each window.

    See :func:`peak_amplitude` for how the extremum is located.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    polarity : {"absolute", "positive", "negative"}, default "absolute"
        Which extremum to locate.
    prominence : float, optional
        Minimum prominence for a local peak to qualify.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Latency in seconds, relative to the epoch origin.
    """
    return _peak(series, "peak_latency", "s", windows, polarity, prominence, groups, include_global)


def _measure(
    series: Sequence[TimeSeries],
    measure: str,
    unit: str,
    kernel: SignalKernel[TimeSeries],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> FeatureTable:
    return expand_signal(
        series,
        trace_of=_trace_of,
        kernel=kernel,
        units={measure: unit},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
    )


def _peak(
    series: Sequence[TimeSeries],
    measure: str,
    unit: str,
    windows: Sequence[Window],
    polarity: Polarity,
    prominence: float | None,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> FeatureTable:
    if polarity not in ("positive", "negative", "absolute"):
        raise ValueError(
            f"polarity must be 'positive', 'negative' or 'absolute', got {polarity!r}."
        )
    if prominence is not None and not prominence > 0.0:
        raise ValueError(f"prominence must be positive when given, got {prominence}.")

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s
        amplitude, latency = _find_peak(trace, times, polarity, prominence)
        return {measure: amplitude if measure == "peak_amplitude" else latency}

    return expand_signal(
        series,
        trace_of=_trace_of,
        kernel=kernel,
        units={measure: unit},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
    )


def _trace_of(series: TimeSeries) -> npt.NDArray[np.float64]:
    return series.amplitude


def _finite(
    trace: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    finite = np.isfinite(trace)
    usable: npt.NDArray[np.bool_] = np.asarray(finite.any(axis=2), dtype=np.bool_)
    return finite, usable


def _variance_kernel(
    series: TimeSeries, trace: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times
    finite, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        values = np.where(usable, np.nanvar(trace, axis=2), np.nan)
    del finite
    return {"variance": values}


def _ptp_kernel(
    series: TimeSeries, trace: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times
    finite, usable = _finite(trace)
    high = np.where(finite, trace, -np.inf).max(axis=2)
    low = np.where(finite, trace, np.inf).min(axis=2)
    return {"ptp": np.where(usable, high - low, np.nan)}


def _mean_kernel(
    series: TimeSeries, trace: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times
    _, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return {"mean_amplitude": np.where(usable, np.nanmean(trace, axis=2), np.nan)}


def _auc_kernel(
    series: TimeSeries, trace: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    del series
    n_epochs, n_channels, _ = trace.shape
    out = np.full((n_epochs, n_channels), np.nan)
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            out[epoch, channel] = _auc_one(trace[epoch, channel], times)
    return {"auc": out}


def _auc_one(trace: npt.NDArray[np.float64], times: npt.NDArray[np.float64]) -> float:
    valid = np.flatnonzero(np.isfinite(trace) & np.isfinite(times))
    if valid.size < 2:
        return float("nan")
    runs = np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1)
    total, measured = 0.0, False
    for run in runs:
        if run.size < 2:
            continue
        measured = True
        total += float(trapezoid(trace[run], times[run]))
    return total if measured else float("nan")


def _find_peak(
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    polarity: Polarity,
    prominence: float | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    finite, usable = _finite(trace)
    sign = _SEARCH.get(polarity)
    search = np.abs(trace) if sign is None else sign * trace
    filled = np.where(finite, search, -np.inf)

    index = np.argmax(filled, axis=2)
    if prominence is not None:
        for epoch in range(trace.shape[0]):
            for channel in range(trace.shape[1]):
                peaks, properties = find_peaks(filled[epoch, channel], prominence=prominence)
                if peaks.size:
                    index[epoch, channel] = peaks[int(np.argmax(properties["prominences"]))]

    amplitude = np.take_along_axis(trace, index[..., np.newaxis], axis=2)[..., 0]
    return (
        np.where(usable, amplitude, np.nan),
        np.where(usable, times[index], np.nan),
    )
