from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand
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
