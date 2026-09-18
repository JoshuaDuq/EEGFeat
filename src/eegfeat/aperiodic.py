from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from scipy import stats

from eegfeat._expand import Kernel, expand
from eegfeat.bands import Band
from eegfeat.spectra import Spectra
from eegfeat.table import FeatureTable, concat

_MIN_FIT_POINTS = 5


def aperiodic(
    spectra: Spectra,
    *,
    fit_range: tuple[float, float] = (2.0, 40.0),
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    peak_rejection_z: float = 2.5,
    max_iterations: int = 3,
) -> FeatureTable:
    """Fit the aperiodic (1/f) component of the spectrum.

    Fits ``log10(P) = offset + slope * log10(f)`` over ``fit_range``, then
    iteratively discards points lying more than ``peak_rejection_z`` robust
    deviations **above** the fit and refits. Only positive residuals are
    rejected, because oscillatory peaks sit above the aperiodic line and would
    otherwise tilt it; troughs carry no such bias.

    A log-spaced frequency grid is well suited to this fit, since it spaces the
    abscissae evenly in ``log10(f)``.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range to fit, in Hz.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    peak_rejection_z : float, default 2.5
        Residual threshold in robust deviations.
    max_iterations : int, default 3
        Maximum refit rounds.

    Returns
    -------
    FeatureTable
        Columns for ``slope`` (negative for a typical spectrum) and ``offset``.
    """
    band = Band("fit", *fit_range)
    tables: list[FeatureTable] = []
    for which in ("slope", "offset"):

        def make_kernel(w: str) -> Kernel:
            def kernel(
                data: npt.NDArray[np.float64],
                freqs: npt.NDArray[np.float64],
                weights: npt.NDArray[np.float64],
            ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
                return _fit_kernel(
                    data, freqs, weights, which=w, z=peak_rejection_z, iterations=max_iterations
                )

            return kernel

        table = expand(
            spectra,
            make_kernel(which),
            measure=which,
            unit="log10 power per log10 Hz" if which == "slope" else "log10 power",
            bands=(band,),
            groups=groups,
            include_global=include_global,
            baseline=None,
            mode="raw",
            min_bins=_MIN_FIT_POINTS,
        )
        tables.append(table)

    # The fit range is not a named band, so these columns are broadband.
    stripped = [
        FeatureTable(
            values=t.values,
            coverage=t.coverage,
            meta=tuple(replace(m, band=None) for m in t.meta),
            flags=t.flags,
        )
        for t in tables
    ]
    return concat(stripped)


def _fit_kernel(
    data: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    *,
    which: str,
    z: float,
    iterations: int,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
    del weights  # the fit is unweighted in log-log space
    log_f = np.log10(freqs)
    n_epochs, n_channels, n_windows, _ = data.shape
    out = np.full((n_epochs, n_channels, n_windows), np.nan)
    for e in range(n_epochs):
        for c in range(n_channels):
            for w in range(n_windows):
                slope, offset = _fit_one(log_f, data[e, c, w, :], z, iterations)
                out[e, c, w] = slope if which == "slope" else offset
    return out, {}


def _fit_one(
    log_f: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    z: float,
    iterations: int,
) -> tuple[float, float]:
    usable = np.isfinite(power) & (power > 0.0)
    if int(usable.sum()) < _MIN_FIT_POINTS:
        return np.nan, np.nan
    log_p = np.full(power.shape, np.nan)
    log_p[usable] = np.log10(power[usable])

    keep = usable.copy()
    slope, offset = np.nan, np.nan
    for _ in range(iterations):
        picks = np.flatnonzero(keep)
        if picks.size < _MIN_FIT_POINTS:
            break
        poly = np.polyfit(log_f[picks], log_p[picks], 1)
        slope, offset = poly[0], poly[1]
        residuals = log_p - (offset + slope * log_f)
        mad = stats.median_abs_deviation(residuals[keep], scale="normal", nan_policy="omit")
        if not np.isfinite(mad) or mad < 1e-12:
            break
        tightened = keep & (residuals <= z * mad)
        if int(tightened.sum()) < _MIN_FIT_POINTS or np.array_equal(tightened, keep):
            break
        keep = tightened
    return float(slope), float(offset)
