from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import Kernel, expand
from eegfeat.bands import Band
from eegfeat.spectra import Spectra
from eegfeat.table import FeatureTable


def peak_frequency(
    spectra: Spectra,
    *,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Frequency of the largest peak within a band.

    The bin-resolution argmax is refined by parabolic interpolation through the
    maximum and its two neighbours, so the result is not quantized to the grid.

    The estimate is only as trustworthy as the grid it came from, so every
    column reports ``freq_resolution_hz`` and every cell carries an
    ``"edge_hit"`` flag, set when the maximum landed on the first or last bin of
    the band and the true peak may therefore lie outside it. Bands holding fewer
    than three bins raise, because an interior maximum is undefined there.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    band : Band
        Band to search.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Peak frequency in Hz, with the ``"edge_hit"`` flag.
    """
    return expand(
        spectra,
        _peak_kernel,
        measure="peak_freq",
        unit="Hz",
        bands=(band,),
        groups=groups,
        include_global=include_global,
        baseline=None,
        mode="raw",
        min_bins=3,
    )


def _peak_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del weights  # a peak location does not depend on bin widths
    usable = np.isfinite(data).any(axis=3)
    filled = np.where(np.isfinite(data), data, -np.inf)
    index = np.argmax(filled, axis=3)

    last = freqs.size - 1
    interior = (index > 0) & (index < last)
    safe = np.clip(index, 1, max(last - 1, 1))

    def take(offset: int) -> npt.NDArray[np.float64]:
        shifted = np.clip(safe + offset, 0, last)
        return np.take_along_axis(filled, shifted[..., np.newaxis], axis=3)[..., 0]

    left, centre, right = take(-1), take(0), take(1)

    with np.errstate(invalid="ignore", divide="ignore"):
        # An all-NaN slice leaves -inf on every side, so these subtractions are
        # inf - inf by design; the usable mask discards the result below.
        denominator = left - 2.0 * centre + right
        delta = np.where(denominator != 0.0, 0.5 * (left - right) / denominator, 0.0)
    delta = np.where(np.isfinite(delta), np.clip(delta, -0.5, 0.5), 0.0)

    # Local half-spacing, so interpolation is correct on a non-uniform grid too.
    spacing = (freqs[np.clip(safe + 1, 0, last)] - freqs[np.clip(safe - 1, 0, last)]) / 2.0
    peak = freqs[index] + np.where(interior, delta * spacing, 0.0)

    return np.where(usable, peak, np.nan), {"edge_hit": usable & ~interior}


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
    bands holding different numbers of bins are not directly comparable.

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

    return _descriptor(spectra, kernel, "spectral_edge", "Hz", band, groups, include_global)


def _descriptor(
    spectra: Spectra,
    kernel: Kernel,
    measure: str,
    unit: str,
    band: Band,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
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
        weighting="gradient",
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
    del freqs
    mass, total = _mass(data, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.where(total[..., np.newaxis] > 0.0, mass / total[..., np.newaxis], 0.0)
        # 0 log 0 is 0 here, so an empty bin contributes nothing rather than NaN.
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
        entropy = -terms.sum(axis=3) / np.log(float(mass.shape[3]))
    return np.where(total > 0.0, entropy, np.nan), {}
