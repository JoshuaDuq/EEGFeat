from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from scipy import stats

from eegfeat._expand import Kernel, expand
from eegfeat.bands import Band
from eegfeat.spectra import Spectra
from eegfeat.table import ComputationSpec, FeatureTable, concat

_MIN_FIT_POINTS = 5

# In the order _fit_one returns the three measures.
_UNITS: dict[str, str] = {
    "slope": "log10 power per log10 Hz",
    "offset": "log10 power",
    "r_squared": "a.u.",
}


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
    _validate_fit_settings(peak_rejection_z, max_iterations)
    band = Band("fit", *fit_range)
    mask = band.mask(spectra.freqs)
    # One fit per cell serves all three measures; expand only lays out the columns.
    fits = _fit_cells(
        spectra.data[..., mask], spectra.freqs[mask], peak_rejection_z, max_iterations
    )

    def column(index: int) -> Kernel:
        def kernel(
            data: npt.NDArray[np.float64],
            freqs: npt.NDArray[np.float64],
            weights: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]]:
            del data, freqs, weights
            return fits[..., index], {}

        return kernel

    tables = [
        expand(
            spectra,
            column(index),
            measure=which,
            unit=_UNITS[which],
            bands=(band,),
            groups=groups,
            include_global=include_global,
            baseline=None,
            mode="raw",
            min_bins=_MIN_FIT_POINTS,
            parameters={
                "fit_range": fit_range,
                "peak_rejection_z": peak_rejection_z,
                "max_iterations": max_iterations,
            },
        )
        for index, which in enumerate(_UNITS)
    ]

    # The fit range is not a named band, so these columns are broadband.
    stripped = [
        FeatureTable(
            values=t.values,
            coverage=t.coverage,
            meta=tuple(replace(m, band=None) for m in t.meta),
            flags=t.flags,
            row_labels=t.row_labels,
            row_ids=t.row_ids,
        )
        for t in tables
    ]
    return concat(stripped)


def aperiodic_ratio(
    spectra: Spectra,
    *,
    fit_range: tuple[float, float] = (2.0, 40.0),
    peak_rejection_z: float = 2.5,
    max_iterations: int = 3,
) -> Spectra:
    """Divide out the fitted aperiodic component, leaving the periodic residual.

    Returns spectra in which a pure power law is flat at 1.0, so an oscillation is
    measured against the local 1/f floor rather than against zero. This is what
    makes peak finding work on real data: a steep spectrum puts its largest raw
    value at the low edge of any band, whatever the oscillation is doing.

    The fit is the same iterative, positive-residual-rejecting fit used by
    :func:`aperiodic`. A cell whose fit cannot be estimated is entirely NaN and
    carries ``aperiodic_fit_failed``; raw spectra are never silently substituted.

    Parameters
    ----------
    spectra : Spectra
        Input spectra.
    fit_range : tuple of float, default (2.0, 40.0)
        Frequency range to fit, in Hz. The fitted curve is divided out across the
        whole frequency axis, not only this range.
    peak_rejection_z : float, default 2.5
        Residual threshold in robust deviations.
    max_iterations : int, default 3
        Maximum refit rounds.

    Returns
    -------
    Spectra
        Power divided by the fitted aperiodic component. Frequencies at or below
        zero carry no 1/f value and pass through unchanged.
    """
    _validate_fit_settings(peak_rejection_z, max_iterations)
    mask = Band("fit", *fit_range).mask(spectra.freqs)
    n_bins = int(mask.sum())
    if n_bins < _MIN_FIT_POINTS:
        raise ValueError(
            f"fit_range {fit_range} holds {n_bins} frequency bins of the axis spanning "
            f"({spectra.freqs[0]}, {spectra.freqs[-1]}), but the aperiodic fit needs at "
            f"least {_MIN_FIT_POINTS}."
        )

    fits = _fit_cells(
        spectra.data[..., mask], spectra.freqs[mask], peak_rejection_z, max_iterations
    )
    slope, offset = fits[..., 0, np.newaxis], fits[..., 1, np.newaxis]
    failed = ~(np.isfinite(slope) & np.isfinite(offset))[..., 0]
    curve = 10.0 ** (offset + slope * _log_frequency(spectra.freqs))
    out = np.where(spectra.freqs > 0.0, spectra.data / curve, spectra.data)
    out[failed] = np.nan

    return replace(
        spectra,
        data=out,
        coverage=np.where(failed[:, :, :, np.newaxis], 0.0, spectra.coverage),
        source=f"{spectra.source}+aperiodic_ratio",
        computation=ComputationSpec.create(
            "aperiodic_ratio",
            input_computation=spectra.computation.record(),
            fit_range=fit_range,
            peak_rejection_z=peak_rejection_z,
            max_iterations=max_iterations,
        ),
        flags={**spectra.flags, "aperiodic_fit_failed": failed},
    )


def _validate_fit_settings(peak_rejection_z: float, max_iterations: int) -> None:
    if (
        isinstance(peak_rejection_z, bool)
        or not isinstance(peak_rejection_z, Real)
        or not np.isfinite(peak_rejection_z)
        or peak_rejection_z <= 0.0
    ):
        raise ValueError(f"peak_rejection_z must be finite and positive, got {peak_rejection_z}.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}.")


def _fit_cells(
    data: npt.NDArray[np.float64], freqs: npt.NDArray[np.float64], z: float, iterations: int
) -> npt.NDArray[np.float64]:
    """Slope, offset and r_squared of every cell, stacked on a last axis of three."""
    log_f = _log_frequency(freqs)
    fits = np.full((*data.shape[:3], 3), np.nan)
    for index in np.ndindex(data.shape[:3]):
        fits[index] = _fit_one(log_f, data[index], z, iterations)
    return fits


def _log_frequency(freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # A power law has no value at 0 Hz: NaN there keeps a DC bin out of every fit.
    positive = freqs > 0.0
    log_f = np.full(freqs.shape, np.nan)
    log_f[positive] = np.log10(freqs[positive])
    return log_f


def _fit_one(
    log_f: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    z: float,
    iterations: int,
) -> tuple[float, float, float]:
    usable = np.isfinite(power) & (power > 0.0) & np.isfinite(log_f)
    if int(usable.sum()) < _MIN_FIT_POINTS:
        return np.nan, np.nan, np.nan
    log_p = np.full(power.shape, np.nan)
    log_p[usable] = np.log10(power[usable])

    keep = usable.copy()
    slope, offset = np.polyfit(log_f[keep], log_p[keep], 1)
    for _ in range(iterations):
        residuals = log_p - (offset + slope * log_f)
        mad = stats.median_abs_deviation(residuals[keep], scale="normal", nan_policy="omit")
        if not np.isfinite(mad) or mad < 1e-12:
            break
        tightened = keep & (residuals <= z * mad)
        if int(tightened.sum()) < _MIN_FIT_POINTS or np.array_equal(tightened, keep):
            break
        # Refit on every tightened mask, so the line and r_squared share one point set.
        keep = tightened
        slope, offset = np.polyfit(log_f[keep], log_p[keep], 1)
    return float(slope), float(offset), _r_squared(log_f, log_p, keep, slope, offset)


def _r_squared(
    log_f: npt.NDArray[np.float64],
    log_p: npt.NDArray[np.float64],
    keep: npt.NDArray[np.bool_],
    slope: float,
    offset: float,
) -> float:
    """Fit quality over the points the line was actually fitted to.

    Rejected peaks are excluded on purpose: an alpha peak is not a failure of the
    aperiodic model, so scoring against it would read low for every healthy
    spectrum. What this does catch is a bend the line cannot follow -- a knee
    inside the fit range -- which is the failure that silently biases the exponent.
    """
    picks = np.flatnonzero(keep)
    if picks.size < _MIN_FIT_POINTS or not np.isfinite(slope) or not np.isfinite(offset):
        return float("nan")
    observed = log_p[picks]
    residual = float(np.sum((observed - (offset + slope * log_f[picks])) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - residual / total)
