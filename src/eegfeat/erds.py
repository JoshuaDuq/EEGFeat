from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand_signal, window_mask
from eegfeat._validation import blank_non_finite
from eegfeat.baseline import normalize as _normalize
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

_MIN_BASELINE_FRACTION = 1e-6
"""Smallest baseline power that can anchor a ratio, as a fraction of the epoch's own.

A baseline near zero turns a quiet channel into an ERDS value of order 1e6
percent, which is arithmetically valid and physically meaningless. The guard is
relative rather than an absolute number of V², because absolute power is a
property of the band and the montage, not of the data being valid: a gamma
envelope of well under a microvolt is ordinary EEG, and an absolute floor near
1e-12 V² discards it. A baseline a millionth of the same channel's power over
the whole epoch is a dropout, not a quiet channel, at any montage scale.

Refused baselines yield NaN and set the ``baseline_degenerate`` flag, so a
withheld value is distinguishable from one that was never measurable.
"""

_EXTREME_POWER_RATIO = 1e4
"""Power ratio above which a value is reported but flagged rather than withheld.

A baseline a hundred times smaller in amplitude than the epoch's peak yields an
ERDS of order 1e6 percent. That is suspicious, but it is a measurement, not a
missing value: a hundredfold response is what stimulation artifact and muscle
look like, and no scale-free rule separates that from a true response. So the
value stands and ``baseline_extreme_ratio`` marks it for the caller's own QC.
"""

ErdsScale = Literal["percent", "db"]

_UNITS: dict[str, dict[str, str]] = {
    "percent": {
        "erds_mean": "%",
        "erds_slope": "%/s",
        "erd_magnitude": "%",
        "erd_duration": "s",
        "ers_magnitude": "%",
        "ers_duration": "s",
        "erds_peak_latency": "s",
        "erds_onset_latency": "s",
        "erds_rebound_latency": "s",
    },
    "db": {
        "erds_mean": "dB",
        "erds_slope": "dB/s",
        "erd_magnitude": "dB",
        "erd_duration": "s",
        "ers_magnitude": "dB",
        "ers_duration": "s",
        "erds_peak_latency": "s",
        "erds_onset_latency": "s",
        "erds_rebound_latency": "s",
    },
}


def erds_mean(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Mean of the ERDS trace over the window.

    The conventional summary: negative is desynchronization, positive is synchronization.

    Band power in each analysis window is expressed relative to ``baseline``,
    per epoch and per channel, so every trial is referenced to its own
    pre-stimulus power and no cross-trial leakage arises.

    **Decibels are the default because these are single-trial values.** Percent
    change is right-skewed on a single trial: a trial whose baseline happens to be
    quiet reports several hundred percent, and a handful of those dominate a mean
    over forty-five trials. On twenty subjects of a public motor dataset the
    trial-mean percent ERDS showed mu desynchronization in about half of them; the
    decibel mean, a symmetric log ratio, showed it in every one. Percent remains
    available for display and for comparison with the classic literature.

    **Two ways to average decibels.** This function averages the per-sample dB
    trace over the window. :func:`~eegfeat.mean_tfr_power` with a baseline and
    ``normalize="db"`` takes the dB of the window-mean power instead. They are not
    the same number: for the near-exponential distribution of instantaneous power
    the per-sample mean sits about 2.5 dB lower (the mean of a log is below the log
    of a mean). Report which one you used.

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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_mean",
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
    normalize: ErdsScale = "db",
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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_slope",
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
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Mean depth of the desynchronized part of the trace.

    The mean of ``abs(trace)`` over samples below zero. Exactly ``0.0`` when no sample is
    negative: no desynchronization is a measurement, not a missing value.

    .. warning::

       **Zero is not this measure's null.** Instantaneous band power is close to
       exponentially distributed, so on a window with no task effect at all the
       trace is below its own baseline mean about ``1 - 1/e`` of the time. Because this
       averages only the samples that fall below zero, it is a conditional mean and
       lands near **55-60%** with no effect present; ``0.0`` is reachable only when no
       sample is negative at all.
       Measured on stationary noise and on a real recording referenced to its own
       pre-stimulus baseline, the two agree to within a few tenths of a percent.
       Compare against a null you construct -- a shuffled or pre-stimulus window
       -- rather than against zero or against half the window.

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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

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
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Time spent desynchronized.

    Count of samples below zero divided by the sampling rate. Exactly ``0.0`` when none are.

    .. warning::

       **Zero is not this measure's null.** Instantaneous band power is close to
       exponentially distributed, so on a window with no task effect at all the
       trace is below its own baseline mean about ``1 - 1/e`` of the time. That puts
       this measure at roughly **63% of the window**, not half, and ``ers_duration`` at
       the remaining 37%.
       Measured on stationary noise and on a real recording referenced to its own
       pre-stimulus baseline, the two agree to within a few tenths of a percent.
       Compare against a null you construct -- a shuffled or pre-stimulus window
       -- rather than against zero or against half the window.

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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

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
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Mean height of the synchronized part of the trace.

    The mean of the trace over samples above zero, and ``0.0`` when none are.

    .. warning::

       **Zero is not this measure's null.** Instantaneous band power is close to
       exponentially distributed, so on a window with no task effect at all the
       trace is below its own baseline mean about ``1 - 1/e`` of the time. Because this
       averages only the samples above zero, it is a conditional mean and lands near
       **90-120%** with no effect present; the long right tail of a power ratio makes it
       larger than its ERD counterpart.
       Measured on stationary noise and on a real recording referenced to its own
       pre-stimulus baseline, the two agree to within a few tenths of a percent.
       Compare against a null you construct -- a shuffled or pre-stimulus window
       -- rather than against zero or against half the window.

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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

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
    normalize: ErdsScale = "db",
) -> FeatureTable:
    """Time spent synchronized.

    Count of samples above zero divided by the sampling rate, and ``0.0`` when none are.

    .. warning::

       **Zero is not this measure's null.** Instantaneous band power is close to
       exponentially distributed, so on a window with no task effect at all the
       trace is below its own baseline mean about ``1 - 1/e`` of the time. That puts
       this measure at roughly **37% of the window**, not half, and ``erd_duration`` at
       the remaining 63%.
       Measured on stationary noise and on a real recording referenced to its own
       pre-stimulus baseline, the two agree to within a few tenths of a percent.
       Compare against a null you construct -- a shuffled or pre-stimulus window
       -- rather than against zero or against half the window.

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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

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
    normalize: ErdsScale = "db",
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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_peak_latency",
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
    normalize: ErdsScale = "db",
    min_duration_ms: float | None = None,
    min_duration_cycles: float = 6.0,
) -> FeatureTable:
    """Time the trace first leaves the baseline's own variability and stays out.

    The start of the first run of consecutive samples whose absolute raw-power
    departure from the baseline mean exceeds one baseline standard deviation and
    persists for the required duration. The criterion is independent of percent
    versus decibel output. NaN when no run lasts that long.

    **The persistence is in cycles of the band's low edge by default**, because
    the null rate of this detector depends on the band. A narrow band's envelope
    changes slowly, so consecutive samples are far from independent and a fixed
    number of milliseconds is a much weaker requirement at 4 Hz than at 30 Hz.
    Measured on rest epochs of a public motor dataset, where nothing happens, a
    100 ms requirement produced an onset on essentially every trial in theta, mu
    and beta; about six cycles of the low edge brought the false-onset rate to
    roughly 5 percent in every band (1.5 s at 4 Hz, 0.75 s at 8 Hz, 0.46 s at
    13 Hz, 0.2 s at 30 Hz). That is what the default encodes.

    .. warning::

       On the same dataset the onset fired on movement trials at the same rate as
       on rest trials at every persistence, so a single-trial onset is a weak
       detector even when the group-level desynchronization is unmistakable. Read
       an onset rate against a null you construct from pre-stimulus or shuffled
       windows, not as evidence that a response occurred on that trial.

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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean`.
    min_duration_ms : float, optional
        Shortest excursion that counts as an onset, in milliseconds. When given it
        replaces ``min_duration_cycles`` for every band. Zero restores the first
        single-sample crossing, which fires on every trial.
    min_duration_cycles : float, default 6.0
        Shortest excursion in cycles of each band's ``fmin``, used when
        ``min_duration_ms`` is None. A band with ``fmin`` of zero has no cycle
        length and must be given ``min_duration_ms``.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    if min_duration_ms is not None and (not np.isfinite(min_duration_ms) or min_duration_ms < 0.0):
        raise ValueError(f"min_duration_ms must be finite and non-negative, got {min_duration_ms}.")
    if not np.isfinite(min_duration_cycles) or min_duration_cycles < 0.0:
        raise ValueError(
            f"min_duration_cycles must be finite and non-negative, got {min_duration_cycles}."
        )
    if min_duration_ms is None:
        for signal in signals:
            if signal.band.fmin <= 0.0:
                raise ValueError(
                    f"band {signal.band.name!r} starts at 0 Hz, so a persistence in cycles is "
                    "undefined; pass min_duration_ms."
                )
    return _erds_measure(
        signals,
        "erds_onset_latency",
        baseline=baseline,
        windows=windows,
        groups=groups,
        include_global=include_global,
        normalize=normalize,
        onset_persistence=(min_duration_ms, min_duration_cycles),
    )


def erds_rebound_latency(
    signals: Sequence[BandSignal],
    *,
    baseline: Window,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    normalize: ErdsScale = "db",
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
    normalize : {"percent", "db"}, default "db"
        Decibels, or percent change from baseline. See :func:`erds_mean` for why
        decibels are the default.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and window.
    """
    return _erds_measure(
        signals,
        "erds_rebound_latency",
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
    onset_persistence: tuple[float | None, float] = (None, 6.0),
) -> FeatureTable:
    if normalize not in ("percent", "db"):
        raise ValueError(f"normalize must be 'percent' or 'db', got {normalize!r}.")
    min_duration_ms, min_duration_cycles = onset_persistence

    def onset_seconds(signal: BandSignal) -> float:
        if min_duration_ms is not None:
            return min_duration_ms / 1000.0
        return min_duration_cycles / signal.band.fmin

    def trace_of(signal: BandSignal) -> npt.NDArray[np.float64]:
        return _trace(signal, baseline, normalize)

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        reference, deviation = _baseline_stats(signal, baseline)
        # The expander's own selector, not one recovered from the time values.
        power = signal.power[:, :, mask]
        onset_crossing = (
            np.isfinite(power)
            & np.isfinite(reference[:, :, np.newaxis])
            & (np.abs(power - reference[:, :, np.newaxis]) > deviation[:, :, np.newaxis])
        )
        onset_samples = max(1, int(round(onset_seconds(signal) * signal.sfreq)))
        # Every measure derives from the same trace, so computing the set and
        # taking one is cheaper than it looks and keeps the definitions together.
        return {measure: _measures(signal, trace, times, onset_crossing, onset_samples)[measure]}

    def flags_of(signal: BandSignal) -> dict[str, npt.NDArray[np.bool_]]:
        reference, _, degenerate = _baseline_reference(signal, baseline)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            peak = np.nanmax(_power(signal), axis=2)
            extreme = ~degenerate & (peak > _EXTREME_POWER_RATIO * reference)
        return {
            "baseline_degenerate": degenerate,
            "baseline_extreme_ratio": np.asarray(extreme, dtype=np.bool_),
        }

    return expand_signal(
        signals,
        trace_of=trace_of,
        kernel=kernel,
        flags_of=flags_of,
        units={measure: _UNITS[normalize][measure]},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode=normalize,
        parameters={
            "baseline": {"name": baseline.name, "tmin": baseline.tmin, "tmax": baseline.tmax},
            # Only the onset is defined by the persistence, so only its columns carry it.
            **(
                {
                    "onset_criterion": "absolute_power_deviation_exceeds_baseline_sd_sustained",
                    "onset_min_duration_ms": min_duration_ms,
                    "onset_min_duration_cycles": (
                        None if min_duration_ms is not None else min_duration_cycles
                    ),
                }
                if measure == "erds_onset_latency"
                else {}
            ),
        },
    )


def _baseline_stats(
    signal: BandSignal, baseline: Window
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    reference, deviation, degenerate = _baseline_reference(signal, baseline)
    return np.where(degenerate, np.nan, reference), deviation


def _power(signal: BandSignal) -> npt.NDArray[np.float64]:
    """Instantaneous power with non-finite samples blanked.

    Read here rather than through the analysis trace, so nothing has blanked it
    yet. It matters more than a wrong mean: an infinity makes the baseline
    reference non-finite, which is read as degenerate, and every ERDS measure for
    that channel is withheld over a single bad sample.
    """
    return blank_non_finite(signal.power)


def _baseline_reference(
    signal: BandSignal, baseline: Window
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Baseline mean power, its spread, and which cells cannot anchor a ratio."""
    mask = window_mask(signal.times, baseline)
    full = _power(signal)
    power = full[:, :, mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        reference = np.nanmean(power, axis=2)
        deviation = np.nanstd(power, axis=2)
        # The whole epoch, not just the baseline, so the comparison is against this
        # channel's own scale rather than an assumed unit of measurement.
        scale = np.nanmean(full, axis=2)
    with np.errstate(invalid="ignore"):
        degenerate: npt.NDArray[np.bool_] = np.asarray(
            ~np.isfinite(reference)
            | (reference <= 0.0)
            | (np.isfinite(scale) & (reference <= _MIN_BASELINE_FRACTION * scale)),
            dtype=np.bool_,
        )
    return reference, deviation, degenerate


def _trace(signal: BandSignal, baseline: Window, mode: ErdsScale) -> npt.NDArray[np.float64]:
    power = _power(signal)
    reference, _ = _baseline_stats(signal, baseline)
    return _normalize(power, baseline=reference, mode=mode)


def _measures(
    signal: BandSignal,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    onset_crossing: npt.NDArray[np.bool_],
    onset_samples: int = 1,
) -> dict[str, npt.NDArray[np.float64]]:
    finite = np.isfinite(trace)
    usable: npt.NDArray[np.bool_] = np.asarray(finite.any(axis=2), dtype=np.bool_)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.where(usable, np.nanmean(trace, axis=2), np.nan)
    peak_index = _argmax_masked(np.abs(trace), finite)
    return {
        "erds_mean": mean,
        "erds_slope": _slope(trace, times, finite),
        "erd_magnitude": _signed_magnitude(trace, finite, usable, negative=True),
        "erd_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=True),
        "ers_magnitude": _signed_magnitude(trace, finite, usable, negative=False),
        "ers_duration": _signed_duration(trace, finite, usable, signal.sfreq, negative=False),
        "erds_peak_latency": np.where(usable, times[peak_index], np.nan),
        "erds_onset_latency": _onset(times, usable, onset_crossing, onset_samples),
        "erds_rebound_latency": _rebound(trace, times, finite, usable, peak_index),
    }


def _argmax_masked(
    values: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int_]:
    return np.asarray(np.argmax(np.where(finite, values, -np.inf), axis=2), dtype=np.int_)


def _onset(
    times: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    crossed: npt.NDArray[np.bool_],
    min_samples: int,
) -> npt.NDArray[np.float64]:
    """Start of the first run of at least ``min_samples`` consecutive crossings.

    A single sample beyond one baseline SD is met by chance on essentially every
    trial, so a crossing only counts once it has persisted. A non-finite sample
    breaks a run: missing data is not evidence that the excursion continued.
    """
    n_times = crossed.shape[2]
    run = np.zeros(crossed.shape[:2], dtype=int)
    start = np.full(crossed.shape[:2], -1, dtype=int)
    for index in range(n_times):
        run = np.where(crossed[:, :, index], run + 1, 0)
        qualifies = (run >= min_samples) & (start < 0)
        start = np.where(qualifies, index - min_samples + 1, start)
    found = start >= 0
    return np.where(usable & found, times[np.maximum(start, 0)], np.nan)


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
