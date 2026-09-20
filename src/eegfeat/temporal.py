from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from eegfeat._expand import SignalKernel, expand_signal
from eegfeat._validation import blank_non_finite
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


def hjorth_mobility(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    r"""Hjorth mobility: the signal's mean frequency, estimated in the time domain.

    .. math:: \mathrm{mobility} = \frac{1}{2\pi}
              \sqrt{\frac{\operatorname{Var}(\mathrm{d}x/\mathrm{d}t)}
              {\operatorname{Var}(x)}}

    **Reported in hertz.** The derivative is taken with respect to time, not per
    sample, and the result is divided by :math:`2\pi`, so a 10 Hz sine gives 10.0
    at every sampling rate. The usual ``std(diff(x)) / std(x)`` omits both and
    returns :math:`2\pi f / f_s`, which makes the same recording resampled to a
    different rate report a different number -- a real hazard when features are
    pooled across datasets.

    The derivative is a finite difference, whose gain is ``sin(pi f / f_s)``
    rather than ``pi f / f_s``, so content close to Nyquist is reported a little
    low: a 25 Hz sine sampled at 128 Hz gives 23.5 Hz, not 25. The shortfall is
    under 2% while the signal stays below a tenth of the sampling rate, and it
    is a property of the time-domain estimator, not of this implementation --
    use :func:`~eegfeat.spectral_centroid` if you need the spectrum's own answer.

    This is the time-domain counterpart of :func:`~eegfeat.spectral_centroid`,
    and the two agree on a pure oscillation. They diverge on broadband signals,
    where mobility weights the spectrum by :math:`f^2` and the centroid by
    :math:`f`: mobility is the root-mean-square frequency, the centroid the mean.

    Hjorth activity is the variance of the signal, which :func:`~eegfeat.variance`
    already computes; it is not duplicated under a second name.

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
        Mobility in Hz. NaN where the window holds fewer than three finite samples
        or the signal does not vary.

    References
    ----------
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    return _measure(
        series, "hjorth_mobility", "Hz", _mobility_kernel, windows, groups, include_global
    )


def hjorth_complexity(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    r"""Hjorth complexity: how far the signal departs from a pure sine.

    .. math:: \mathrm{complexity} =
              \frac{\mathrm{mobility}(\mathrm{d}x/\mathrm{d}t)}{\mathrm{mobility}(x)}

    Exactly 1 for a pure sine and larger as the spectrum broadens. Dimensionless,
    and the sampling interval cancels out of the ratio, so unlike
    :func:`hjorth_mobility` this one is already rate-independent however it is
    written.

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
        Complexity, dimensionless. NaN where the window holds fewer than four
        finite samples or either variance in the ratio vanishes.

    References
    ----------
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    return _measure(
        series, "hjorth_complexity", "a.u.", _complexity_kernel, windows, groups, include_global
    )


def root_mean_square(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Root mean square amplitude within each window.

    Equal to the standard deviation for a signal with no offset, and larger when
    there is one; :func:`~eegfeat.variance` removes the mean first and this does not.

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
        RMS amplitude in the units of the input.
    """
    return _measure(series, "rms", "V", _rms_kernel, windows, groups, include_global)


def skewness(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Asymmetry of the amplitude distribution within each window.

    Zero for a symmetric distribution, positive when the long tail points up.
    Dimensionless, so it does not change with the recording's amplitude scale.

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
        Skewness, dimensionless.
    """
    return _measure(series, "skewness", "a.u.", _skewness_kernel, windows, groups, include_global)


def kurtosis(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Tail weight of the amplitude distribution within each window.

    Excess kurtosis: **zero for a Gaussian**, positive for heavier tails. High
    values are the usual signature of a transient artifact sitting in an
    otherwise ordinary segment.

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
        Excess kurtosis, dimensionless.
    """
    return _measure(series, "kurtosis", "a.u.", _kurtosis_kernel, windows, groups, include_global)


def line_length(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Mean absolute rate of change within each window.

    **Per second, not per sample.** The conventional ``mean(abs(diff(x)))`` is an
    amplitude per sample, so the same recording at another sampling rate reports a
    different number; this multiplies by the sampling rate and reports V/s, which
    does not move. It is otherwise the same quantity, and dividing by the sampling
    rate recovers the per-sample form if you need to compare against one.

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
        Mean absolute slope, in the units of the input per second.
    """
    return _measure(
        series, "line_length", "V/s", _line_length_kernel, windows, groups, include_global
    )


def zero_crossing_rate(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Sign changes per second within each window.

    **A rate, not a count**, so windows of different lengths and recordings at
    different sampling rates are comparable. A crossing is a change of sign
    between consecutive finite samples; a sample of exactly zero is treated as
    continuing the run it sits in rather than as two crossings.

    For a *band-limited* zero-mean Gaussian signal this is approximately twice
    :func:`~eegfeat.hjorth_mobility` (Rice's formula), within a few percent while
    the content stays below roughly a third of the sampling rate. The agreement
    degrades as the band approaches Nyquist, and does not hold at all for raw
    white noise, whose consecutive samples are independent rather than
    band-limited: there the rate is simply half the sampling rate.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes. An envelope is non-negative, so its rate
        is zero by construction; this measure is for signals that cross zero.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Crossings per second.
    """
    return _measure(
        series, "zero_crossing_rate", "1/s", _zero_crossing_kernel, windows, groups, include_global
    )


def amplitude_quantile(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    q: float = 0.5,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """A quantile of the amplitude distribution within each window.

    The default is the median, which is far less sensitive to a single transient
    than :func:`~eegfeat.mean_amplitude`.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    q : float, default 0.5
        Quantile in ``[0, 1]``. Note this is a fraction, not a percentage.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        The quantile, in the units of the input.
    """
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be a finite fraction in [0, 1], got {q}.")

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask
        _, usable = _finite(trace)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            values = np.nanquantile(trace, q, axis=2)
        return {"amplitude_quantile": np.where(usable, values, np.nan)}

    return expand_signal(
        series,
        trace_of=_trace_of,
        kernel=kernel,
        units={"amplitude_quantile": "V"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"q": q},
    )


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
        parameters={},
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
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, mask
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
        parameters={"polarity": polarity, "prominence": prominence},
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
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    finite, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        values = np.where(usable, np.nanvar(trace, axis=2), np.nan)
    del finite
    return {"variance": values}


def _ptp_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    finite, usable = _finite(trace)
    high = np.where(finite, trace, -np.inf).max(axis=2)
    low = np.where(finite, trace, np.inf).min(axis=2)
    return {"ptp": np.where(usable, high - low, np.nan)}


def _mean_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    _, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return {"mean_amplitude": np.where(usable, np.nanmean(trace, axis=2), np.nan)}


def _finite_variance(values: npt.NDArray[np.float64], minimum: int) -> npt.NDArray[np.float64]:
    """Variance over finite samples, NaN where too few of them survive."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        variance = np.nanvar(blank_non_finite(values), axis=2)
    enough = np.isfinite(values).sum(axis=2) >= minimum
    return np.asarray(np.where(enough, variance, np.nan), dtype=np.float64)


def _mobility_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del times, mask
    # Per second, not per sample: np.diff alone would carry the sampling interval
    # into the result and make the same signal report differently at another rate.
    derivative = np.diff(trace, axis=2) * series.sfreq
    # Two derivative samples, so three of the signal: the variance of a single
    # value is zero, which would report a still signal rather than an unusable one.
    signal_variance = _finite_variance(trace, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            signal_variance > 0.0, _finite_variance(derivative, 2) / signal_variance, np.nan
        )
        mobility = np.sqrt(ratio) / (2.0 * np.pi)
    return {"hjorth_mobility": np.asarray(mobility, dtype=np.float64)}


def _complexity_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    # A ratio of two mobilities, so the sampling interval cancels and the raw
    # differences are enough; scaling them would divide out again.
    first = np.diff(trace, axis=2)
    second = np.diff(trace, n=2, axis=2)
    signal_variance = _finite_variance(trace, 3)
    first_variance = _finite_variance(first, 2)
    # Two second-difference samples, so four of the signal; see _mobility_kernel.
    second_variance = _finite_variance(second, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = np.where(
            (signal_variance > 0.0) & (first_variance > 0.0),
            np.sqrt(second_variance * signal_variance) / first_variance,
            np.nan,
        )
    return {"hjorth_complexity": np.asarray(complexity, dtype=np.float64)}


def _rms_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    _, usable = _finite(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        values = np.sqrt(np.nanmean(np.square(blank_non_finite(trace)), axis=2))
    return {"rms": np.where(usable, values, np.nan)}


def _moment_kernel(
    trace: npt.NDArray[np.float64], measure: str, minimum: int
) -> dict[str, npt.NDArray[np.float64]]:
    # scipy returns a masked array under nan_policy="omit"; fill it so the column is
    # plain float, and withhold cells with too few samples for the moment to exist.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = (
            stats.skew(trace, axis=2, nan_policy="omit")
            if measure == "skewness"
            else stats.kurtosis(trace, axis=2, nan_policy="omit")
        )
    values: npt.NDArray[np.float64] = np.ma.filled(np.ma.asarray(raw, dtype=float), np.nan)
    enough = np.isfinite(trace).sum(axis=2) >= minimum
    return {measure: np.asarray(np.where(enough, values, np.nan), dtype=np.float64)}


def _skewness_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    return _moment_kernel(trace, "skewness", 3)


def _kurtosis_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, times, mask
    return _moment_kernel(trace, "kurtosis", 4)


def _line_length_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del times, mask
    steps = blank_non_finite(np.abs(np.diff(blank_non_finite(trace), axis=2)))
    usable = np.isfinite(steps).any(axis=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        # Per second: np.diff alone carries the sampling interval into the result.
        values = np.nanmean(steps, axis=2) * series.sfreq
    return {"line_length": np.where(usable, values, np.nan)}


def _zero_crossing_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del times, mask
    finite = np.isfinite(trace)
    # A sample of exactly zero continues the run it sits in. Counting it as a
    # crossing on the way in and again on the way out reports two where a signal
    # that merely touched the axis made none.
    sign = np.sign(np.where(finite, trace, np.nan))
    previous = np.zeros(sign.shape[:2])
    crossings = np.zeros(sign.shape[:2])
    for index in range(sign.shape[2]):
        current = sign[:, :, index]
        moved = np.isfinite(current) & (current != 0.0)
        crossings += moved & (previous != 0.0) & (current != previous)
        previous = np.where(moved, current, previous)
    seconds = float(trace.shape[2]) / series.sfreq
    usable = finite.any(axis=2) & (seconds > 0)
    return {"zero_crossing_rate": np.where(usable, crossings / seconds, np.nan)}


def _auc_kernel(
    series: TimeSeries,
    trace: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    del series, mask
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
