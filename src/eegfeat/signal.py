from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mne  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert

from eegfeat.bands import Band

_FILTER_LENGTH_MULTIPLIER = 6.6
_MIN_FILTER_LENGTH = 3
_DEFAULT_LOW_FREQ_HZ = 0.1


@runtime_checkable
class TimeSeries(Protocol):
    """What a feature function needs from a time-domain container.

    Structural, not inherited: :class:`Signal` and :class:`BandSignal` satisfy it
    without sharing a base class, so a measure defined on a trace works on a raw
    recording and on a band envelope alike.
    """

    @property
    def times(self) -> npt.NDArray[np.float64]: ...

    @property
    def ch_names(self) -> tuple[str, ...]: ...

    @property
    def sfreq(self) -> float: ...

    @property
    def coverage(self) -> npt.NDArray[np.float64]: ...

    @property
    def amplitude(self) -> npt.NDArray[np.float64]:
        """The real-valued series a time-domain measure reads."""

    @property
    def band(self) -> Band | None:
        """The band this series is restricted to, or None for broadband."""

    @property
    def source(self) -> str:
        """Provenance, recorded on every feature derived from this series."""


@dataclass(frozen=True, eq=False)
class Signal:
    """A broadband time series over epochs, channels and time.

    The raw counterpart to :class:`BandSignal`: no filtering, no analytic signal,
    nothing derived. Measures that read the signal itself rather than a band take
    this.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)
        Signal amplitude.
    times : ndarray, shape (n_times,)
        Time axis in seconds, ascending, relative to the epoch origin.
    ch_names : tuple of str
        Channel names, one per channel axis entry.
    sfreq : float
        Sampling frequency in Hz.
    coverage : ndarray, same shape as ``data``
        Fraction of each sample that was finite, in ``[0, 1]``.
    """

    data: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    sfreq: float
    coverage: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_series(self.data, self.times, self.ch_names, self.coverage, self.sfreq, "data")

    @property
    def amplitude(self) -> npt.NDArray[np.float64]:
        """The signal itself."""
        return self.data

    @property
    def band(self) -> Band | None:
        """Always None: a raw signal is broadband."""
        return None

    @property
    def source(self) -> str:
        """Provenance label recorded on derived features."""
        return "signal"

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return int(self.data.shape[0])

    @classmethod
    def from_epochs(cls, epochs: Any) -> Signal:
        """Wrap an ``mne.Epochs`` without transforming it.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data. Every channel present is carried through, including
            non-EEG channels and those marked bad; pass an already-picked object
            if that matters.

        Returns
        -------
        Signal
        """
        data = np.asarray(epochs.get_data(), dtype=float)
        return cls(
            data=data,
            times=np.asarray(epochs.times, dtype=float),
            ch_names=tuple(epochs.ch_names),
            sfreq=float(epochs.info["sfreq"]),
            coverage=np.isfinite(data).astype(float),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        data: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        ch_names: tuple[str, ...],
        sfreq: float,
        coverage: npt.NDArray[np.float64] | None = None,
    ) -> Signal:
        """Build from arrays.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            Signal amplitude.
        times : ndarray, shape (n_times,)
            Time axis in seconds.
        ch_names : tuple of str
            Channel names.
        sfreq : float
            Sampling frequency in Hz.
        coverage : ndarray, optional
            Per-sample coverage. Defaults to where ``data`` is finite.

        Returns
        -------
        Signal
        """
        array = np.asarray(data, dtype=float)
        return cls(
            data=array,
            times=np.asarray(times, dtype=float),
            ch_names=tuple(ch_names),
            sfreq=float(sfreq),
            coverage=np.isfinite(array).astype(float) if coverage is None else coverage,
        )


def _validate_series(
    values: npt.NDArray[Any],
    times: npt.NDArray[np.float64],
    ch_names: tuple[str, ...],
    coverage: npt.NDArray[np.float64],
    sfreq: float,
    label: str,
) -> None:
    if values.ndim != 3:
        raise ValueError(
            f"{label} must be 3-D (n_epochs, n_channels, n_times), got {values.shape}."
        )
    n_channels, n_times = values.shape[1:]
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names has {len(ch_names)} entries but {label} has {n_channels} channels."
        )
    if times.ndim != 1 or times.size != n_times:
        raise ValueError(f"times must be 1-D of length {n_times}, got {times.shape}.")
    if n_times > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly ascending.")
    if coverage.shape != values.shape:
        raise ValueError(f"coverage shape {coverage.shape} does not match {label} {values.shape}.")
    if not sfreq > 0.0:
        raise ValueError(f"sfreq must be positive, got {sfreq}.")


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
    def amplitude(self) -> npt.NDArray[np.float64]:
        """The envelope: a band signal's amplitude over time."""
        return self.envelope

    @property
    def source(self) -> str:
        """Provenance label recorded on derived features."""
        return "hilbert"

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

    @classmethod
    def from_epochs(
        cls,
        epochs: Any,
        band: Band,
        *,
        pad_sec: float = 0.5,
        pad_cycles: float = 3.0,
        n_jobs: int = 1,
    ) -> BandSignal:
        """Bandpass epochs and take their Hilbert transform.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data.
        band : Band
            Band to filter to. ``band.fmax`` must be below Nyquist.
        pad_sec : float, default 0.5
            Minimum reflect padding in seconds.
        pad_cycles : float, default 3.0
            Padding expressed in cycles of ``band.fmin``. The padding actually
            applied is the larger of the two, clamped to one sample short of the
            epoch length.
        n_jobs : int, default 1
            Passed to MNE's filter.

        Returns
        -------
        BandSignal
            With the padding removed, so ``times`` matches ``epochs.times``.
        """
        sfreq = float(epochs.info["sfreq"])
        if band.fmax > sfreq / 2.0:
            raise ValueError(
                f"band {band.name!r} reaches {band.fmax} Hz, above the Nyquist "
                f"frequency {sfreq / 2.0} of this recording."
            )

        data = np.asarray(epochs.get_data(), dtype=float)
        n_epochs, n_channels, n_times = data.shape
        flat = data.reshape(-1, n_times)

        pad = _padding_samples(pad_sec, pad_cycles, band.fmin, sfreq, n_times)
        padded = np.pad(flat, ((0, 0), (pad, pad)), mode="reflect") if pad else flat

        filtered = mne.filter.filter_data(
            padded,
            sfreq,
            l_freq=band.fmin,
            h_freq=band.fmax,
            filter_length=_filter_length(padded.shape[-1], sfreq, band.fmin),
            n_jobs=n_jobs,
            verbose=False,
        )
        analytic = hilbert(filtered, axis=-1)
        if pad:
            analytic = analytic[:, pad:-pad]

        analytic = analytic.reshape(n_epochs, n_channels, n_times)
        return cls.from_arrays(
            analytic=analytic,
            times=np.asarray(epochs.times, dtype=float),
            ch_names=tuple(epochs.ch_names),
            band=band,
            sfreq=sfreq,
            # Coverage is taken from the analytic signal, not the input: a single
            # non-finite input sample propagates through the FIR convolution and the
            # Hilbert transform and destroys the whole epoch, so input finiteness
            # would claim a channel is intact when every output sample is NaN.
            coverage=np.isfinite(analytic).astype(float),
        )


def _padding_samples(
    pad_sec: float, pad_cycles: float, fmin: float, sfreq: float, n_times: int
) -> int:
    cycles_sec = pad_cycles / fmin if np.isfinite(fmin) and fmin > 0 and pad_cycles > 0 else 0.0
    seconds = max(pad_sec, cycles_sec)
    if not (np.isfinite(seconds) and seconds > 0) or n_times <= 1:
        return 0
    return max(0, min(int(round(seconds * sfreq)), n_times - 1))


def _filter_length(n_times: int, sfreq: float, fmin: float) -> str:
    # MNE's own default would exceed the signal for a low fmin on a short epoch,
    # so fall back to the longest odd length that fits.
    low = fmin if fmin > 0 else _DEFAULT_LOW_FREQ_HZ
    if int(_FILTER_LENGTH_MULTIPLIER * sfreq / low) < n_times:
        return "auto"
    safe = n_times - 1
    if safe % 2 == 0:
        safe -= 1
    return str(max(safe, _MIN_FILTER_LENGTH))
