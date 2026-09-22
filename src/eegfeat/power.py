from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand
from eegfeat.bands import BANDS_STANDARD, Band
from eegfeat.spectra import Spectra
from eegfeat.table import FeatureTable, Normalization

_PSD_MEAN_UNITS: dict[str, str] = {
    "raw": "V^2/Hz",
    "log10": "log10(V^2/Hz)",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}

_PSD_INTEGRAL_UNITS: dict[str, str] = {
    "raw": "V^2",
    "log10": "log10(V^2)",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}

# Morlet power arrives as a density: Spectra.from_tfr divides MNE's energy-2
# wavelet power by the sampling rate, so these are the same units as mean_psd.
_TFR_MEAN_UNITS: dict[str, str] = {
    "raw": "V^2/Hz",
    "log10": "log10(V^2/Hz)",
    "log_ratio": "log10 ratio",
    "db": "dB",
    "percent": "%",
}


def mean_psd(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Frequency-weighted mean power spectral density in each band.

    The band value is a trapezoidally weighted mean over the frequencies in the
    band, which makes it an estimate of the integral over the band divided by
    its width. On a log-spaced grid a plain mean would weight the bottom of
    every band far too heavily.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    bands : sequence of Band, default BANDS_STANDARD
        Bands to compute.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    baseline : str, optional
        Name of the window to normalize against. That window is consumed and
        does not appear in the output. Normalized coverage is the minimum of
        analysis and baseline coverage; baseline flags propagate to the output.
    normalize : {"raw", "log10", "log_ratio", "db", "percent"}, default "raw"
        Normalization. ``"log_ratio"``, ``"db"`` and ``"percent"`` require ``baseline``.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and emitted window.
    """
    _require_representation(spectra, "psd", "mean_psd")
    return expand(
        spectra,
        _weighted_band_mean,
        measure="mean_psd",
        unit=_PSD_MEAN_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
        parameters={"quantity": "mean_power_spectral_density"},
        weighting="band_integral",
    )


def integrated_band_power(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Integral of a power spectral density over exact band boundaries.

    Piecewise-linear quadrature integrates from each band's numerical ``fmin``
    through ``fmax``, including interpolated boundary contributions. Raw EEG PSD
    input therefore yields V² rather than V²/Hz.
    """
    _require_representation(spectra, "psd", "integrated_band_power")
    return expand(
        spectra,
        _weighted_band_integral,
        measure="band_power",
        unit=_PSD_INTEGRAL_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=2,
        parameters={"quantity": "integrated_power_spectral_density"},
        weighting="band_integral",
    )


def mean_tfr_power(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Frequency-weighted mean of wavelet time-frequency power in each band.

    The value is a smoothed spectral density in V²/Hz, because
    :meth:`~eegfeat.Spectra.from_tfr` divides MNE's Morlet power by the sampling
    rate; without that step the same recording reports a different number at
    every sampling rate. It is averaged over the band rather than integrated, so it stays a density
    comparable to :func:`mean_psd`; integrating it would give wavelet-smoothed
    band power in V².

    With a ``baseline`` and ``normalize="db"`` this is the decibel of the
    **window-mean** power. :func:`~eegfeat.erds_mean` averages a per-sample dB
    trace instead, which for near-exponential instantaneous power sits about
    2.5 dB lower. The two are both ERDS in decibels and are not interchangeable.
    """
    _require_representation(spectra, "time_frequency_power", "mean_tfr_power")
    return expand(
        spectra,
        _weighted_band_mean,
        measure="mean_tfr_power",
        unit=_TFR_MEAN_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
        parameters={"quantity": "mean_time_frequency_power"},
        weighting="band_integral",
    )


def _weighted_band_mean(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del freqs
    finite = np.isfinite(data)
    spread: npt.NDArray[np.float64] = np.asarray(
        np.broadcast_to(weights, data.shape), dtype=np.float64
    )
    numerator = np.sum(np.where(finite, data * spread, 0.0), axis=3)
    # A frequency that produced nothing leaves the average rather than entering it as zero.
    denominator = np.sum(np.where(finite, spread, 0.0), axis=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return values, {}


def _weighted_band_integral(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del freqs
    finite = np.isfinite(data)
    complete = finite.all(axis=3)
    spread = np.asarray(np.broadcast_to(weights, data.shape), dtype=np.float64)
    values = np.sum(np.where(finite, data * spread, 0.0), axis=3)
    return np.where(complete, values, np.nan), {}


def _require_representation(spectra: Spectra, expected: str, operation: str) -> None:
    if spectra.representation != expected:
        raise ValueError(
            f"{operation} requires spectral representation {expected!r}, got "
            f"{spectra.representation!r}."
        )
