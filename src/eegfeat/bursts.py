from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand_signal
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


def burst_features(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    threshold: float | npt.NDArray[np.float64] = 0.75,
    min_duration_ms: float = 100.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Extract oscillatory burst features from band-limited amplitude envelopes.

    Parameters
    ----------
    signals : sequence of BandSignal
        Band-limited signals whose envelopes are analyzed.
    windows : sequence of Window
        Time windows to restrict the analysis to.
    threshold : float or ndarray, default 0.75
        Either a float in ``(0, 1)``, interpreted as a per-trial envelope
        percentile, or an array broadcastable to ``(n_epochs, n_channels)``
        holding absolute envelope thresholds.
    min_duration_ms : float, default 100.0
        Minimum duration in milliseconds for an excursion above threshold to
        qualify as a burst. Non-negative.
    groups : mapping of str to sequence of str, optional
        ROI name to channel names. None yields per-channel features.
    include_global : bool, default True
        Whether to compute features across all channels.

    Returns
    -------
    FeatureTable
        Columns for ``count``, ``rate``, ``duration_mean``, ``amp_mean``
        and ``fraction_above``.
    """
    if min_duration_ms < 0.0:
        raise ValueError(f"min_duration_ms must be non-negative, got {min_duration_ms}.")

    thresholds: dict[int, npt.NDArray[np.float64]] = {}

    def trace_of(signal: BandSignal) -> npt.NDArray[np.float64]:
        envelope = signal.envelope
        thresholds[id(signal)] = _resolve_threshold(envelope, threshold)
        return envelope

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del times
        return _measures(
            trace,
            thresholds[id(signal)],
            signal.sfreq,
            min_duration_ms,
        )

    return expand_signal(
        signals,
        trace_of=trace_of,
        kernel=kernel,
        units=_UNITS,
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
    )


def _resolve_threshold(
    envelope: npt.NDArray[np.float64],
    threshold: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_epochs, n_channels = envelope.shape[:2]
    if isinstance(threshold, (int, float)):
        val = float(threshold)
        if not 0.0 < val < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            return np.nanquantile(envelope, val, axis=2)
    array = np.asarray(threshold, dtype=np.float64)
    try:
        return np.broadcast_to(array, (n_epochs, n_channels))
    except ValueError as exc:
        raise ValueError(
            f"threshold array shape {array.shape} cannot broadcast to ({n_epochs}, {n_channels})."
        ) from exc


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
