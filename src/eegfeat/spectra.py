from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


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
    at each endpoint. The spectral descriptors use this weighting and band power
    uses the trapezoid rule, matching the reference pipeline, which applies both.
    On a uniform grid every weight here is equal, which is what gives a flat
    spectrum exactly maximal entropy.

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


@dataclass(frozen=True, eq=False)
class Spectra:
    """Power spectra over epochs, channels and time windows.

    This is the single input type for every measure in the library. It is built
    from an MNE ``Spectrum`` or ``EpochsTFR``; past those constructors the two
    are indistinguishable.

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
        Fraction of contributing samples that were finite, per frequency.
        Per-frequency rather than per-window because a Morlet kernel is wider at
        low frequencies, so frequencies drop out of a window individually.
    source : str
        Provenance, e.g. ``"morlet"``, ``"multitaper"``, ``"welch"``.
    """

    data: npt.NDArray[np.float64]
    freqs: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    windows: tuple[Window, ...]
    coverage: npt.NDArray[np.float64]
    source: str

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(
                "data must be 4-D (n_epochs, n_channels, n_windows, n_freqs), "
                f"got {self.data.shape}."
            )
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

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.data.shape[0])

    @classmethod
    def from_spectrum(cls, spectrum: Any) -> Spectra:
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
        )
