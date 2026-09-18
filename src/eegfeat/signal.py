from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eegfeat.bands import Band


@dataclass(frozen=True, eq=False)
class BandSignal:
    """A band-filtered analytic signal over epochs, channels and time.

    Only the complex analytic signal is stored; ``envelope``, ``phase`` and
    ``power`` are derived on access. The reference implementation keeps five
    parallel arrays of this shape, which holds the same information at five times
    the memory.

    Parameters
    ----------
    analytic : ndarray of complex, shape (n_epochs, n_channels, n_times)
        Analytic signal, the bandpass output plus its Hilbert transform.
    times : ndarray, shape (n_times,)
        Time axis in seconds, ascending, relative to the epoch origin.
    ch_names : tuple of str
        Channel names, one per channel axis entry.
    band : Band
        The band this signal was filtered to.
    sfreq : float
        Sampling frequency in Hz.
    coverage : ndarray, same shape as ``analytic``
        Fraction of each sample that was finite, in ``[0, 1]``.
    """

    analytic: npt.NDArray[np.complex128]
    times: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    band: Band
    sfreq: float
    coverage: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.analytic.ndim != 3:
            raise ValueError(
                "analytic must be 3-D (n_epochs, n_channels, n_times), "
                f"got {self.analytic.shape}."
            )
        if not np.iscomplexobj(self.analytic):
            raise TypeError("analytic must be complex; a real array has already lost its phase.")
        n_channels, n_times = self.analytic.shape[1:]
        if len(self.ch_names) != n_channels:
            raise ValueError(
                f"ch_names has {len(self.ch_names)} entries but analytic has "
                f"{n_channels} channels."
            )
        if self.times.ndim != 1 or self.times.size != n_times:
            raise ValueError(f"times must be 1-D of length {n_times}, got {self.times.shape}.")
        if n_times > 1 and not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly ascending.")
        if self.coverage.shape != self.analytic.shape:
            raise ValueError(
                f"coverage shape {self.coverage.shape} does not match analytic "
                f"{self.analytic.shape}."
            )
        if not self.sfreq > 0.0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}.")

    @property
    def envelope(self) -> npt.NDArray[np.float64]:
        """Instantaneous amplitude."""
        return np.abs(self.analytic)

    @property
    def phase(self) -> npt.NDArray[np.float64]:
        """Instantaneous phase in radians, in ``[-pi, pi]``."""
        return np.angle(self.analytic)

    @property
    def power(self) -> npt.NDArray[np.float64]:
        """Instantaneous power, the squared envelope."""
        return np.abs(self.analytic) ** 2

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.analytic.shape[0])

    @classmethod
    def from_arrays(
        cls,
        *,
        analytic: npt.NDArray[np.complex128],
        times: npt.NDArray[np.float64],
        ch_names: tuple[str, ...],
        band: Band,
        sfreq: float,
        coverage: npt.NDArray[np.float64] | None = None,
    ) -> BandSignal:
        """Build from arrays the caller filtered themselves.

        Parameters
        ----------
        analytic : ndarray of complex, shape (n_epochs, n_channels, n_times)
            Analytic signal.
        times : ndarray, shape (n_times,)
            Time axis in seconds.
        ch_names : tuple of str
            Channel names.
        band : Band
            The band the signal was filtered to.
        sfreq : float
            Sampling frequency in Hz.
        coverage : ndarray, optional
            Per-sample coverage. Defaults to where ``analytic`` is finite.

        Returns
        -------
        BandSignal
        """
        array = np.asarray(analytic)
        if coverage is None:
            coverage = np.isfinite(array).astype(float)
        return cls(
            analytic=array,
            times=np.asarray(times, dtype=float),
            ch_names=tuple(ch_names),
            band=band,
            sfreq=float(sfreq),
            coverage=coverage,
        )
