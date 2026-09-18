from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand
from eegfeat.bands import BANDS_STANDARD, Band
from eegfeat.spectra import Spectra
from eegfeat.table import FeatureTable, Normalization

_UNITS: dict[str, str] = {
    "raw": "V^2/Hz",
    "log10": "log10(V^2/Hz)",
    "log_ratio": "log10 ratio",
    "db": "dB",
}


def band_power(
    spectra: Spectra,
    *,
    bands: Sequence[Band] = BANDS_STANDARD,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    baseline: str | None = None,
    normalize: Normalization = "raw",
) -> FeatureTable:
    """Mean power in each band.

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
        does not appear in the output.
    normalize : {"raw", "log10", "log_ratio", "db"}, default "raw"
        Normalization. ``"log_ratio"`` and ``"db"`` require ``baseline``.

    Returns
    -------
    FeatureTable
        One column per band, spatial unit and emitted window.
    """
    return expand(
        spectra,
        _weighted_band_mean,
        measure="power",
        unit=_UNITS[normalize],
        bands=bands,
        groups=groups,
        include_global=include_global,
        baseline=baseline,
        mode=normalize,
        min_bins=1,
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
