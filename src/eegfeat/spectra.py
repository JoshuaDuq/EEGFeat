from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from eegfeat.identity import epoch_row_ids
from eegfeat.table import ComputationSpec, RowId


@dataclass(frozen=True)
class Window:
    """A named time window in seconds relative to the epoch origin.

    Parameters
    ----------
    name : str
        Window label, used in feature names.
    tmin, tmax : float
        Bounds in seconds. Infinite bounds denote the whole segment.
    """

    name: str
    tmin: float
    tmax: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Window name must be a non-empty string.")
        if not self.tmin < self.tmax:
            raise ValueError(
                f"Window {self.name!r} requires tmin < tmax, got {self.tmin} >= {self.tmax}."
            )


def trapezoid_weights(freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Trapezoidal integration weights for a frequency axis.

    Weighting by bin width rather than averaging bins is what makes a band value
    an estimate of the integral over the band. It matters because a log-spaced
    grid samples low frequencies far more densely than high ones, so a plain
    mean would weight the bottom of every band too heavily.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Ascending frequency axis.

    Returns
    -------
    ndarray, shape (n_freqs,)
        Weights summing to ``freqs[-1] - freqs[0]``.
    """
    f = np.asarray(freqs, dtype=float)
    if f.size <= 1:
        return np.ones(f.size, dtype=float)
    weights = np.zeros(f.size, dtype=float)
    weights[0] = (f[1] - f[0]) / 2.0
    weights[1:-1] = (f[2:] - f[:-2]) / 2.0
    weights[-1] = (f[-1] - f[-2]) / 2.0
    return weights


def gradient_weights(freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Central-difference bin widths for a frequency axis.

    Identical to :func:`trapezoid_weights` in the interior and twice its value
    at each endpoint. Spectral descriptors use this weighting, whereas band power
    uses the trapezoid rule. On a uniform grid every weight here is equal, which is
    what gives a flat spectrum exactly maximal entropy.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Ascending frequency axis.

    Returns
    -------
    ndarray, shape (n_freqs,)
        Bin widths.
    """
    f = np.asarray(freqs, dtype=float)
    if f.size <= 1:
        return np.ones(f.size, dtype=float)
    out: npt.NDArray[np.float64] = np.asarray(np.gradient(f), dtype=np.float64)
    return out


def band_integration_weights(
    freqs: npt.NDArray[np.float64], fmin: float, fmax: float
) -> npt.NDArray[np.float64]:
    """Piecewise-linear quadrature weights over exact frequency bounds."""
    axis = np.asarray(freqs, dtype=float)
    if axis.size < 2 or axis[0] > fmin or axis[-1] < fmax:
        raise ValueError(
            f"frequency axis ({axis[0]}, {axis[-1]}) does not span the complete "
            f"band [{fmin}, {fmax}]."
        )
    weights = np.zeros(axis.size)
    for index, (left, right) in enumerate(zip(axis[:-1], axis[1:], strict=True)):
        start = max(left, fmin)
        stop = min(right, fmax)
        if stop <= start:
            continue
        width = right - left
        start_fraction = (start - left) / width
        stop_fraction = (stop - left) / width
        square_difference = stop_fraction**2 - start_fraction**2
        weights[index] += width * (stop_fraction - start_fraction - square_difference / 2.0)
        weights[index + 1] += width * square_difference / 2.0
    return weights


@dataclass(frozen=True, eq=False)
class Spectra:
    """Power spectra over epochs, channels and time windows.

    This is the single spectral container for the library. It is built from an
    MNE ``Spectrum`` or ``EpochsTFR`` while retaining whether values are a PSD or
    time-frequency power, so dimensionally incompatible reductions are rejected.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_windows, n_freqs)
        Power.
    freqs : ndarray, shape (n_freqs,)
        Strictly ascending frequency axis in Hz.
    ch_names : tuple of str
        Channel names, one per channel axis entry.
    windows : tuple of Window
        Time windows, one per window axis entry.
    coverage : ndarray, same shape as ``data``
        Fraction of support-valid coefficients that were finite, per frequency.
    support : ndarray, same shape as ``data``
        Fraction of the requested window whose coefficients have complete
        temporal support inside that window. This is distinct from numerical
        finiteness: clean data can have full ``coverage`` but partial ``support``.
    source : str
        Provenance, e.g. ``"morlet"``, ``"multitaper"``, ``"welch"``.
    representation : {"psd", "time_frequency_power"}
        A density per Hz or wavelet time-frequency power.
    """

    data: npt.NDArray[np.float64]
    freqs: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    windows: tuple[Window, ...]
    coverage: npt.NDArray[np.float64]
    source: str
    representation: Literal["psd", "time_frequency_power"]
    support: npt.NDArray[np.float64]
    row_ids: tuple[RowId, ...]
    computation: ComputationSpec
    flags: Mapping[str, npt.NDArray[np.bool_]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(
                "data must be 4-D (n_epochs, n_channels, n_windows, n_freqs), "
                f"got {self.data.shape}."
            )
        if self.representation not in ("psd", "time_frequency_power"):
            raise ValueError(f"unknown spectral representation {self.representation!r}.")
        n_channels, n_windows, n_freqs = self.data.shape[1:]
        if len(self.ch_names) != n_channels:
            raise ValueError(
                f"ch_names has {len(self.ch_names)} entries but data has {n_channels} channels."
            )
        if len(self.windows) != n_windows:
            raise ValueError(
                f"windows has {len(self.windows)} entries but data has {n_windows} windows."
            )
        if self.freqs.ndim != 1 or self.freqs.size != n_freqs:
            raise ValueError(
                f"freqs must be 1-D of length {n_freqs}, got shape {self.freqs.shape}."
            )
        if n_freqs > 1 and not np.all(np.diff(self.freqs) > 0):
            raise ValueError("freqs must be strictly ascending.")
        if self.coverage.shape != self.data.shape:
            raise ValueError(
                f"coverage shape {self.coverage.shape} does not match data {self.data.shape}."
            )
        if self.support.shape != self.data.shape:
            raise ValueError(
                f"support shape {self.support.shape} does not match data {self.data.shape}."
            )
        if len(self.row_ids) != self.n_epochs:
            raise ValueError(
                f"row_ids has {len(self.row_ids)} entries but data has {self.n_epochs} epochs."
            )
        for name, flag in self.flags.items():
            if flag.shape != self.data.shape[:3]:
                raise ValueError(
                    f"flag {name!r} shape {flag.shape} does not match spectral cells "
                    f"{self.data.shape[:3]}."
                )

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.data.shape[0])

    @classmethod
    def from_spectrum(
        cls, spectrum: Any, *, recording: str, estimator_parameters: dict[str, object]
    ) -> Spectra:
        """Build from an MNE ``Spectrum`` or ``EpochsSpectrum``.

        Parameters
        ----------
        spectrum : mne.time_frequency.Spectrum or EpochsSpectrum
            A computed power spectrum. A continuous ``Spectrum`` gains a
            leading epoch axis of length 1.

        Returns
        -------
        Spectra
            With a single window named ``"all"`` spanning the whole segment.
        """
        data = np.asarray(spectrum.get_data(), dtype=float)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            raise ValueError(
                f"expected a (channels, freqs) or (epochs, channels, freqs) spectrum, "
                f"got shape {data.shape}."
            )
        data = data[:, :, np.newaxis, :]
        return cls(
            data=data,
            freqs=np.asarray(spectrum.freqs, dtype=float),
            ch_names=tuple(spectrum.ch_names),
            windows=(Window("all", -np.inf, np.inf),),
            coverage=np.isfinite(data).astype(float),
            source=str(getattr(spectrum, "method", "unknown")),
            representation="psd",
            support=np.ones(data.shape, dtype=float),
            row_ids=epoch_row_ids(spectrum, recording, data.shape[0]),
            computation=ComputationSpec.create(
                str(getattr(spectrum, "method", "unknown")), **estimator_parameters
            ),
        )

    @classmethod
    def from_tfr(
        cls,
        tfr: Any,
        windows: Sequence[Window],
        *,
        recording: str,
        n_cycles: float | npt.NDArray[np.float64],
    ) -> Spectra:
        """Build from an MNE ``EpochsTFR`` by averaging over time windows.

        Parameters
        ----------
        tfr : mne.time_frequency.EpochsTFR
            Real-valued power, not baseline-corrected.
        windows : sequence of Window
            Time windows to average over, inclusive of both bounds.
        n_cycles : float or ndarray
            Morlet cycle count used to compute the TFR. Required because MNE
            does not store it on the TFR object and safe temporal attribution
            cannot be inferred without it.

        Returns
        -------
        Spectra
            One spectrum per window.
        """
        if getattr(tfr, "baseline", None) is not None:
            raise ValueError(
                "this TFR is already baseline-corrected "
                f"(baseline={tfr.baseline!r}); normalizing it again is meaningless. "
                "Pass an uncorrected TFR and use the baseline argument of the feature function."
            )
        if not windows:
            raise ValueError("from_tfr requires at least one window.")

        data = np.asarray(tfr.get_data())
        if np.iscomplexobj(data):
            raise ValueError("from_tfr requires real power; got a complex TFR.")
        data = data.astype(float)
        if data.ndim != 4:
            raise ValueError(
                "expected an EpochsTFR of shape (epochs, channels, freqs, times), "
                f"got shape {data.shape}."
            )

        times = np.asarray(tfr.times, dtype=float)
        freqs = np.asarray(tfr.freqs, dtype=float)
        per_window = [_reduce_window(data, times, freqs, window, n_cycles) for window in windows]
        return cls(
            data=np.stack([values for values, _, _ in per_window], axis=2),
            freqs=freqs,
            ch_names=tuple(tfr.ch_names),
            windows=tuple(windows),
            coverage=np.stack([cover for _, cover, _ in per_window], axis=2),
            source=str(getattr(tfr, "method", "unknown")),
            representation="time_frequency_power",
            support=np.stack([support for _, _, support in per_window], axis=2),
            row_ids=epoch_row_ids(tfr, recording, data.shape[0]),
            computation=ComputationSpec.create(
                "morlet",
                n_cycles=np.asarray(n_cycles, dtype=float),
                frequencies_hz=freqs,
            ),
        )


def support_restricted_mask(
    times: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    window: Window,
    n_cycles: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Per-frequency time mask of coefficients a window can account for.

    MNE constructs a Morlet wavelet to five Gaussian standard deviations in
    either direction. With ``sigma_t = n_cycles / (2*pi*f)``, its temporal
    half-support is therefore ``5*n_cycles / (2*pi*f)`` seconds. Only
    coefficients whose complete wavelet lies inside the window are attributable
    to it.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Time axis in seconds.
    freqs : ndarray, shape (n_freqs,)
        Frequency axis in Hz.
    window : Window
        The window to restrict to.
    n_cycles : float or ndarray
        Cycle count, scalar or one value per frequency.

    Returns
    -------
    ndarray of bool, shape (n_freqs, n_times)
        True where the coefficient is attributable to the window. A row is all
        False when no coefficient at that frequency fits.
    """
    f = np.asarray(freqs, dtype=float)
    cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), f.shape)
    half_support = 5.0 * cycles / (2.0 * np.pi * f)
    lower = window.tmin + half_support
    upper = window.tmax - half_support
    t = np.asarray(times, dtype=float)[np.newaxis, :]
    return (t >= lower[:, np.newaxis]) & (t <= upper[:, np.newaxis])


def _reduce_window(
    data: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    window: Window,
    n_cycles: float | npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    requested = (times >= window.tmin) & (times <= window.tmax)
    if not requested.any():
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) selects no samples "
            f"from a time axis spanning ({times[0]}, {times[-1]})."
        )
    mask_2d = support_restricted_mask(times, freqs, window, n_cycles)
    if not mask_2d.any():
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) retains no coefficients "
            f"at any frequency once Morlet support is accounted for. Widen the window or "
            f"lower n_cycles."
        )
    selected = np.where(mask_2d[np.newaxis, np.newaxis, :, :], data, np.nan)
    finite = np.isfinite(selected)
    n_selected = mask_2d.sum(axis=1).astype(float)
    with warnings.catch_warnings():
        # A frequency whose support never fits the window is an all-NaN slice by
        # design; the finite.any() guard already discards its mean. np.errstate
        # does not suppress this one, because nanmean raises it through warnings.
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        values = np.where(
            finite.any(axis=3), np.nanmean(np.where(finite, selected, np.nan), axis=3), np.nan
        )
    coverage = finite.sum(axis=3) / np.where(n_selected > 0, n_selected, np.nan)
    support_by_frequency = n_selected / float(requested.sum())
    support = np.broadcast_to(support_by_frequency, values.shape)
    return values, np.nan_to_num(coverage, nan=0.0), support
