from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand_signal, window_mask
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

_UNITS: dict[str, str] = {
    "count": "count",
    "rate": "1/s",
    "duration_mean": "s",
    "amp_mean": "V",
    "fraction_above": "ratio",
}


def burst_count(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    baseline: Window | None = None,
    threshold: float | npt.NDArray[np.float64] = 0.75,
    min_duration_ms: float = 100.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Number of bursts in the window.

    A burst is a contiguous run of samples above threshold lasting at least
    ``min_duration_ms``. Runs touching a window edge are counted with their visible
    duration. Exactly ``0.0`` when none survive.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    baseline : Window, optional
        Window the percentile threshold is calibrated on. Strongly preferred for
        task data: calibrating on the analysis window instead lets the stimulus
        response raise the very threshold used to detect it. When omitted the
        threshold is calibrated on the analysis windows, which is right only for
        resting state. Ignored when ``threshold`` is an array.
    threshold : float or ndarray, default 0.75
        A float in ``(0, 1)`` is a percentile of the envelope within each epoch
        and channel, carrying no cross-trial leakage. An array broadcastable to
        ``(n_epochs, n_channels)`` is used as absolute envelope values.
    min_duration_ms : float, default 100.0
        Shortest run retained, in milliseconds.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window. The unit records how the
        threshold was set.
    """
    return _burst_measure(
        signals,
        "count",
        windows=windows,
        baseline=baseline,
        threshold=threshold,
        min_duration_ms=min_duration_ms,
        groups=groups,
        include_global=include_global,
    )


def burst_rate(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    baseline: Window | None = None,
    threshold: float | npt.NDArray[np.float64] = 0.75,
    min_duration_ms: float = 100.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Bursts per second.

    ``count`` divided by the window duration, so windows of different lengths are
    comparable.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    baseline : Window, optional
        Window the percentile threshold is calibrated on. Strongly preferred for
        task data: calibrating on the analysis window instead lets the stimulus
        response raise the very threshold used to detect it. When omitted the
        threshold is calibrated on the analysis windows, which is right only for
        resting state. Ignored when ``threshold`` is an array.
    threshold : float or ndarray, default 0.75
        A float in ``(0, 1)`` is a percentile of the envelope within each epoch
        and channel, carrying no cross-trial leakage. An array broadcastable to
        ``(n_epochs, n_channels)`` is used as absolute envelope values.
    min_duration_ms : float, default 100.0
        Shortest run retained, in milliseconds.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window. The unit records how the
        threshold was set.
    """
    return _burst_measure(
        signals,
        "rate",
        windows=windows,
        baseline=baseline,
        threshold=threshold,
        min_duration_ms=min_duration_ms,
        groups=groups,
        include_global=include_global,
    )


def burst_duration(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    baseline: Window | None = None,
    threshold: float | npt.NDArray[np.float64] = 0.75,
    min_duration_ms: float = 100.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean burst duration in seconds.

    Averaged over surviving runs only. NaN when none survive, because there is genuinely
    nothing to average.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    baseline : Window, optional
        Window the percentile threshold is calibrated on. Strongly preferred for
        task data: calibrating on the analysis window instead lets the stimulus
        response raise the very threshold used to detect it. When omitted the
        threshold is calibrated on the analysis windows, which is right only for
        resting state. Ignored when ``threshold`` is an array.
    threshold : float or ndarray, default 0.75
        A float in ``(0, 1)`` is a percentile of the envelope within each epoch
        and channel, carrying no cross-trial leakage. An array broadcastable to
        ``(n_epochs, n_channels)`` is used as absolute envelope values.
    min_duration_ms : float, default 100.0
        Shortest run retained, in milliseconds.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window. The unit records how the
        threshold was set.
    """
    return _burst_measure(
        signals,
        "duration_mean",
        windows=windows,
        baseline=baseline,
        threshold=threshold,
        min_duration_ms=min_duration_ms,
        groups=groups,
        include_global=include_global,
    )


def burst_amplitude(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    baseline: Window | None = None,
    threshold: float | npt.NDArray[np.float64] = 0.75,
    min_duration_ms: float = 100.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean peak envelope amplitude across bursts.

    The maximum of the envelope within each surviving run, averaged. NaN when none
    survive.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    baseline : Window, optional
        Window the percentile threshold is calibrated on. Strongly preferred for
        task data: calibrating on the analysis window instead lets the stimulus
        response raise the very threshold used to detect it. When omitted the
        threshold is calibrated on the analysis windows, which is right only for
        resting state. Ignored when ``threshold`` is an array.
    threshold : float or ndarray, default 0.75
        A float in ``(0, 1)`` is a percentile of the envelope within each epoch
        and channel, carrying no cross-trial leakage. An array broadcastable to
        ``(n_epochs, n_channels)`` is used as absolute envelope values.
    min_duration_ms : float, default 100.0
        Shortest run retained, in milliseconds.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window. The unit records how the
        threshold was set.
    """
    return _burst_measure(
        signals,
        "amp_mean",
        windows=windows,
        baseline=baseline,
        threshold=threshold,
        min_duration_ms=min_duration_ms,
        groups=groups,
        include_global=include_global,
    )


def fraction_above_threshold(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    baseline: Window | None = None,
    threshold: float | npt.NDArray[np.float64] = 0.75,
    min_duration_ms: float = 100.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Share of window samples above threshold.

    Computed before the duration filter, quantifying the overall fraction of samples
    exceeding the envelope amplitude threshold across the window regardless of
    burst duration.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    baseline : Window, optional
        Window the percentile threshold is calibrated on. Strongly preferred for
        task data: calibrating on the analysis window instead lets the stimulus
        response raise the very threshold used to detect it. When omitted the
        threshold is calibrated on the analysis windows, which is right only for
        resting state. Ignored when ``threshold`` is an array.
    threshold : float or ndarray, default 0.75
        A float in ``(0, 1)`` is a percentile of the envelope within each epoch
        and channel, carrying no cross-trial leakage. An array broadcastable to
        ``(n_epochs, n_channels)`` is used as absolute envelope values.
    min_duration_ms : float, default 100.0
        Shortest run retained, in milliseconds.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window. The unit records how the
        threshold was set.
    """
    return _burst_measure(
        signals,
        "fraction_above",
        windows=windows,
        baseline=baseline,
        threshold=threshold,
        min_duration_ms=min_duration_ms,
        groups=groups,
        include_global=include_global,
    )


def _burst_measure(
    signals: Sequence[BandSignal],
    measure: str,
    *,
    windows: Sequence[Window],
    baseline: Window | None,
    threshold: float | npt.NDArray[np.float64],
    min_duration_ms: float,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> FeatureTable:
    if min_duration_ms < 0.0:
        raise ValueError(f"min_duration_ms must be non-negative, got {min_duration_ms}.")
    if not isinstance(threshold, np.ndarray):
        value = float(threshold)
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        label = f"percentile {value} of {'baseline' if baseline else 'analysis windows'}"
    else:
        label = "absolute"

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del times
        level = _resolve_threshold(signal, threshold, baseline, windows)
        return {measure: _measures(trace, level, signal.sfreq, min_duration_ms)[measure]}

    return expand_signal(
        signals,
        trace_of=lambda signal: signal.envelope,
        kernel=kernel,
        units={measure: f"{_UNITS[measure]} ({label})"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
    )


def _resolve_threshold(
    signal: BandSignal,
    threshold: float | npt.NDArray[np.float64],
    baseline: Window | None,
    windows: Sequence[Window],
) -> npt.NDArray[np.float64]:
    n_epochs, n_channels = signal.analytic.shape[:2]
    if isinstance(threshold, np.ndarray):
        try:
            return np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n_epochs, n_channels))
        except ValueError as exc:
            raise ValueError(
                f"threshold array shape {np.shape(threshold)} cannot broadcast to "
                f"({n_epochs}, {n_channels})."
            ) from exc

    calibration = (
        window_mask(signal.times, baseline)
        if baseline is not None
        else np.logical_or.reduce([window_mask(signal.times, w) for w in windows])
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        return np.nanquantile(signal.envelope[:, :, calibration], float(threshold), axis=2)


def _measures(
    trace: npt.NDArray[np.float64],
    thresholds: npt.NDArray[np.float64],
    sfreq: float,
    min_duration_ms: float,
) -> dict[str, npt.NDArray[np.float64]]:
    n_epochs, n_channels, n_times = trace.shape
    min_samples = max(1, int(round(min_duration_ms * sfreq / 1000.0))) if sfreq > 0 else 1
    duration_sec = n_times / sfreq if sfreq > 0 else np.nan

    counts = np.full((n_epochs, n_channels), np.nan)
    rates = np.full((n_epochs, n_channels), np.nan)
    durations = np.full((n_epochs, n_channels), np.nan)
    amplitudes = np.full((n_epochs, n_channels), np.nan)
    fractions = np.full((n_epochs, n_channels), np.nan)

    for e in range(n_epochs):
        for c in range(n_channels):
            tr = trace[e, c]
            thr = thresholds[e, c]
            if not (np.isfinite(thr) and np.any(np.isfinite(tr))):
                continue
            cnt, rt, dur, amp, frac = _extract_single(tr, thr, sfreq, min_samples, duration_sec)
            counts[e, c] = cnt
            rates[e, c] = rt
            durations[e, c] = dur
            amplitudes[e, c] = amp
            fractions[e, c] = frac

    return {
        "count": counts,
        "rate": rates,
        "duration_mean": durations,
        "amp_mean": amplitudes,
        "fraction_above": fractions,
    }


def _extract_single(
    trace: npt.NDArray[np.float64],
    threshold: float,
    sfreq: float,
    min_samples: int,
    duration_sec: float,
) -> tuple[float, float, float, float, float]:
    above = trace > threshold
    if not np.any(above):
        rate = 0.0 if np.isfinite(duration_sec) and duration_sec > 0 else np.nan
        return 0.0, rate, np.nan, np.nan, 0.0

    fraction_above = float(np.mean(above))
    starts, ends = _intervals(above, trace.size)
    run_lengths = np.array([e - s for s, e in zip(starts, ends, strict=True)], dtype=int)
    surviving = run_lengths >= min_samples

    if not np.any(surviving):
        rate = 0.0 if np.isfinite(duration_sec) and duration_sec > 0 else np.nan
        return 0.0, rate, np.nan, np.nan, fraction_above

    valid_starts = [s for s, keep in zip(starts, surviving, strict=True) if keep]
    valid_ends = [e for e, keep in zip(ends, surviving, strict=True) if keep]
    valid_lengths = run_lengths[surviving]

    peak_amplitudes = [
        float(np.nanmax(trace[s:e])) for s, e in zip(valid_starts, valid_ends, strict=True)
    ]
    burst_count = float(len(valid_lengths))
    burst_rate = (
        burst_count / duration_sec if np.isfinite(duration_sec) and duration_sec > 0 else np.nan
    )
    mean_duration_sec = float(np.mean(valid_lengths) / sfreq) if sfreq > 0 else np.nan
    mean_amplitude = float(np.mean(peak_amplitudes)) if peak_amplitudes else np.nan

    return burst_count, burst_rate, mean_duration_sec, mean_amplitude, fraction_above


def _intervals(
    above: npt.NDArray[np.bool_],
    n_samples: int,
) -> tuple[list[int], list[int]]:
    diff = np.diff(above.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [n_samples]
    return starts, ends
