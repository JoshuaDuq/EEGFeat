from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

from eegfeat._expand import Kernel, expand
from eegfeat.aperiodic import aperiodic_ratio
from eegfeat.bands import Band
from eegfeat.spectra import Spectra, band_integration_weights
from eegfeat.table import FeatureTable

_POWER_FLOOR = 1e-20


def peak_frequency(
    spectra: Spectra,
    *,
    band: Band,
    aperiodic_adjusted: bool = True,
    smoothing_hz: float = 1.0,
    min_prominence: float = 0.1,
    interpolate: bool = True,
    fit_range: tuple[float, float] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Frequency of the dominant oscillation within a band.

    A bare argmax is a poor peak estimator on real spectra, so three corrections
    are applied by default and each can be switched off:

    - ``aperiodic_adjusted`` divides out the fitted 1/f component first, via
      :func:`~eegfeat.aperiodic.aperiodic_ratio`, so the peak is measured against
      the local aperiodic floor. Without it a steep spectrum reports the low edge
      of the band whatever the oscillation is doing. The fit spans ``fit_range``,
      which deliberately reaches outside the band: a 1/f slope estimated from a
      five-hertz window is not a 1/f slope.
    - ``smoothing_hz`` averages over a centred window of that width before the
      search, so a single noisy bin cannot win. The average ignores non-finite
      bins and renormalizes rather than closing the gap and averaging across it.
    - ``min_prominence`` guards against reporting noise as a peak. When the
      maximum stands less than this far, in log10 units, above the band median,
      the centre of gravity is reported instead and ``"cog_fallback"`` is set. A
      centre of gravity degrades gracefully when no oscillation is present; an
      argmax does not.

    Every column reports ``freq_resolution_hz``, and every cell carries
    ``"edge_hit"``, set when the maximum landed on the first or last bin of the
    band so the true peak may lie outside it. Bands holding fewer than three bins
    raise, because an interior maximum is undefined there.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to search.
    aperiodic_adjusted : bool, default True
        Divide out the fitted 1/f component before searching. Sets the measure
        name to ``"peak_freq_adjusted"``.
    smoothing_hz : float, default 1.0
        Width of the smoothing window in Hz. Zero disables smoothing.
    min_prominence : float, default 0.1
        Minimum height above the band median, in log10 units, for the maximum to
        be reported as a peak. Zero disables the centre-of-gravity fallback.
    interpolate : bool, default True
        Refine the result by parabolic interpolation through the maximum and its
        neighbours, so it is not quantized to the frequency grid.
    fit_range : tuple of float, optional
        Frequency range for the aperiodic fit. Defaults to
        ``(min(2.0, band.fmin), max(40.0, band.fmax))``. Requires
        ``aperiodic_adjusted``.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak frequency in Hz, with the ``"edge_hit"`` and ``"cog_fallback"`` flags.
    """
    if smoothing_hz < 0.0:
        raise ValueError(f"smoothing_hz must be >= 0, got {smoothing_hz}.")
    if min_prominence < 0.0:
        raise ValueError(f"min_prominence must be >= 0, got {min_prominence}.")
    if fit_range is not None and not aperiodic_adjusted:
        raise ValueError("fit_range applies only when aperiodic_adjusted is True.")

    if aperiodic_adjusted:
        span = fit_range or (min(2.0, band.fmin), max(40.0, band.fmax))
        spectra = aperiodic_ratio(spectra, fit_range=span)

    def kernel(
        data: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
        del weights  # a peak location does not depend on bin widths
        return _find_peak(data, freqs, smoothing_hz, min_prominence, interpolate)

    return expand(
        spectra,
        kernel,
        measure="peak_freq_adjusted" if aperiodic_adjusted else "peak_freq",
        unit="Hz",
        bands=(band,),
        groups=groups,
        include_global=include_global,
        baseline=None,
        mode="raw",
        min_bins=3,
        parameters={
            "aperiodic_adjusted": aperiodic_adjusted,
            "smoothing_hz": smoothing_hz,
            "min_prominence": min_prominence,
            "interpolate": interpolate,
            "fit_range": fit_range,
        },
    )


def _smooth(
    values: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    smoothing_hz: float,
) -> npt.NDArray[np.float64]:
    if smoothing_hz <= 0.0 or freqs.size <= 3:
        return values
    finite = np.isfinite(values)
    radius = smoothing_hz / 2.0
    out = np.full(values.shape, np.nan)
    for index, frequency in enumerate(freqs):
        weights = band_integration_weights(
            freqs,
            max(float(freqs[0]), frequency - radius),
            min(float(freqs[-1]), frequency + radius),
        )
        spread = np.broadcast_to(weights, values.shape)
        selected = finite & (spread > 0.0)
        count = np.where(selected, spread, 0.0).sum(axis=3)
        total = np.where(selected, values * spread, 0.0).sum(axis=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., index] = np.where(count > 0, total / count, np.nan)
    return out


def _centred_boxcar(values: npt.NDArray[np.float64], width: int) -> npt.NDArray[np.float64]:
    # scipy puts an even window one bin further below its output bin than above it,
    # which moves the spectrum, and so the peak, half a bin up. Averaging it with the
    # same window one bin higher centres it, with half weight on the two end bins.
    boxcar = uniform_filter1d(values, size=width, axis=3, mode="nearest")
    if width % 2:
        return boxcar
    return 0.5 * (boxcar + uniform_filter1d(values, size=width, axis=3, mode="nearest", origin=-1))


def _find_peak(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    smoothing_hz: float,
    min_prominence: float,
    interpolate: bool,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    finite = np.isfinite(data)
    usable = finite.any(axis=3)
    present = np.where(finite, data, np.nan)

    power = _smooth(present, freqs, smoothing_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = _smooth(np.log10(np.maximum(present, _POWER_FLOOR)), freqs, smoothing_hz)

    filled = np.where(np.isfinite(power), power, -np.inf)
    index = np.argmax(filled, axis=3)

    last = freqs.size - 1
    interior = (index > 0) & (index < last)
    safe = np.clip(index, 1, max(last - 1, 1))
    peak = freqs[index]

    if interpolate:
        left, centre, right = (_at(filled, np.clip(safe + o, 0, last)) for o in (-1, 0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            # An all-NaN slice leaves -inf on every side, so these subtractions are
            # inf - inf by design; the usable mask discards the result below.
            denominator = left - 2.0 * centre + right
            delta = np.where(denominator != 0.0, 0.5 * (left - right) / denominator, 0.0)
        delta = np.where(np.isfinite(delta), np.clip(delta, -0.5, 0.5), 0.0)
        # Local half-spacing, so interpolation is correct on a non-uniform grid too.
        spacing = (freqs[np.clip(safe + 1, 0, last)] - freqs[np.clip(safe - 1, 0, last)]) / 2.0
        peak = peak + np.where(interior, delta * spacing, 0.0)

    use_cog = np.zeros(usable.shape, dtype=bool)
    if min_prominence > 0.0:
        with warnings.catch_warnings():
            # A cell with no finite bin is an all-NaN slice by design.
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            floor = np.nanmedian(residual, axis=3)
        prominence = _at(residual, index) - floor
        use_cog = usable & np.isfinite(prominence) & (prominence < min_prominence)

    if bool(use_cog.any()):
        weights = np.where(np.isfinite(power) & (power > 0.0), power, 0.0)
        total = weights.sum(axis=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid = np.where(total > 0.0, (weights * freqs).sum(axis=3) / total, np.nan)
        use_cog &= np.isfinite(centroid)
        peak = np.where(use_cog, centroid, peak)

    return (
        np.where(usable, peak, np.nan),
        {"edge_hit": usable & ~interior & ~use_cog, "cog_fallback": use_cog},
    )


def _at(values: npt.NDArray[np.float64], index: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    return np.take_along_axis(values, index[..., np.newaxis], axis=3)[..., 0]


def spectral_centroid(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Centre of mass of the spectrum within a band.

    Computed as ``sum(f * P * df) / sum(P * df)``, so a non-uniform frequency
    grid is handled correctly.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Centroid frequency in Hz.
    """
    return _descriptor(
        spectra,
        _centroid_kernel,
        "spectral_centroid",
        "Hz",
        band,
        groups,
        include_global,
    )


def spectral_bandwidth(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Spread of the spectrum about its centroid, within a band.

    The mass-weighted standard deviation of frequency.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Bandwidth in Hz.
    """
    return _descriptor(
        spectra,
        _bandwidth_kernel,
        "spectral_bandwidth",
        "Hz",
        band,
        groups,
        include_global,
    )


def spectral_entropy(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Shannon entropy of the normalized spectrum, scaled to ``[0, 1]``.

    One means power is spread evenly across the band; zero means it is
    concentrated in a single bin. Normalized by ``log(n_bins)``, so values from
    bands holding different numbers of bins are not directly comparable. The
    discrete definition requires an approximately uniform frequency grid.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Normalized entropy, dimensionless.
    """
    return _descriptor(
        spectra,
        _entropy_kernel,
        "spectral_entropy",
        "a.u.",
        band,
        groups,
        include_global,
        weighting="uniform",
    )


def spectral_edge(
    spectra: Spectra,
    *,
    band: Band,
    percentile: float = 0.95,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Frequency below which a given fraction of the band's power lies.

    Returns the frequency of the bin at which the normalized cumulative mass
    first reaches ``percentile``. It is not interpolated, so the result is
    always a frequency present on the input grid.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to summarize.
    percentile : float, default 0.95
        Cumulative power fraction in ``(0, 1]``. Note this is a fraction, not a
        percentage.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Edge frequency in Hz.
    """
    if not np.isfinite(percentile) or not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be a finite fraction in (0, 1], got {percentile}.")

    def kernel(
        data: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
        mass, total = _mass(data, weights)
        with np.errstate(invalid="ignore", divide="ignore"):
            cumulative = np.cumsum(mass, axis=3) / total[..., np.newaxis]
        reached = cumulative >= percentile
        # Rounding can leave the final cumulative a hair under the threshold; fall back
        # to the last bin rather than to argmax's 0, which would be the first one.
        index = np.where(reached.any(axis=3), reached.argmax(axis=3), freqs.size - 1)
        return np.where(total > 0.0, freqs[index], np.nan), {}

    return _descriptor(
        spectra,
        kernel,
        "spectral_edge",
        "Hz",
        band,
        groups,
        include_global,
        {"percentile": percentile},
    )


def _descriptor(
    spectra: Spectra,
    kernel: Kernel,
    measure: str,
    unit: str,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    parameters: Mapping[str, object] | None = None,
    weighting: Literal["trapezoid", "gradient", "band_integral", "uniform"] = "gradient",
) -> FeatureTable:
    return expand(
        spectra,
        kernel,
        measure=measure,
        unit=unit,
        bands=(band,),
        groups=groups,
        include_global=include_global,
        baseline=None,
        mode="raw",
        min_bins=3,
        parameters={} if parameters is None else parameters,
        weighting=weighting,
    )


def _mass(
    data: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    spread = np.asarray(np.broadcast_to(weights, data.shape), dtype=np.float64)
    mass = np.where(np.isfinite(data), data * spread, 0.0)
    return mass, mass.sum(axis=3)


def _centroid_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(total > 0.0, (mass * freqs).sum(axis=3) / total, np.nan)
    return centroid, {}


def _bandwidth_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(total > 0.0, (mass * freqs).sum(axis=3) / total, np.nan)
        deviation = (freqs[np.newaxis, np.newaxis, np.newaxis, :] - centroid[..., np.newaxis]) ** 2
        variance = np.where(total > 0.0, (mass * deviation).sum(axis=3) / total, np.nan)
    return np.sqrt(variance), {}


def _entropy_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    spacing = np.diff(freqs)
    tolerance = np.finfo(float).eps * max(1.0, abs(float(spacing[0]))) * 8.0
    if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=tolerance):
        raise ValueError(
            "spectral_entropy requires an approximately uniform frequency grid; "
            "recompute or interpolate the PSD onto a uniform-Hz grid."
        )
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.where(total[..., np.newaxis] > 0.0, mass / total[..., np.newaxis], 0.0)
        # 0 log 0 is 0 here, so an empty bin contributes nothing rather than NaN.
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
        entropy = -terms.sum(axis=3) / np.log(float(mass.shape[3]))
    return np.where(total > 0.0, entropy, np.nan), {}
