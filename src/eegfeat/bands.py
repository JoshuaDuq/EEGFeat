from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Band:
    """A named frequency band with half-open bounds ``[fmin, fmax)``.

    Parameters
    ----------
    name : str
        Band label, used in feature names.
    fmin, fmax : float
        Lower (inclusive) and upper (exclusive) bounds in Hz.
    """

    name: str
    fmin: float
    fmax: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Band name must be a non-empty string.")
        # Canonical JSON writes 8 and 8.0 as different tokens, so an int-bounded
        # band would hash differently from the identical float-bounded one and
        # their columns would never match across tables.
        object.__setattr__(self, "fmin", float(self.fmin))
        object.__setattr__(self, "fmax", float(self.fmax))
        if not np.isfinite((self.fmin, self.fmax)).all():
            raise ValueError(f"Band {self.name!r} bounds must be finite.")
        if self.fmin < 0.0:
            raise ValueError(f"Band {self.name!r} fmin must be >= 0, got {self.fmin}.")
        if self.fmin >= self.fmax:
            raise ValueError(
                f"Band {self.name!r} requires fmin < fmax, got {self.fmin} >= {self.fmax}."
            )

    def mask(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting the frequencies that fall in this band.

        Bounds are half-open so that adjacent bands tile a frequency axis
        without assigning any bin to two bands.

        .. note::

           A band means ``[fmin, fmax)`` to everything that selects bins, which
           is every descriptor, the entropy, the peak search, the aperiodic fit
           and the band reduction in :func:`~eegfeat.wpli`. It means the closed
           ``[fmin, fmax]`` to the quadrature in
           :func:`~eegfeat.spectra.band_integration_weights`, which
           :func:`~eegfeat.integrated_band_power`, :func:`~eegfeat.mean_psd` and
           :func:`~eegfeat.mean_tfr_power` use.

           Both are right for what they do -- an integral has to reach ``fmax``,
           and a bin belongs to one band or the other -- but they are not the
           same support. On a 0.25 Hz grid an alpha band of ``(8, 13)`` is
           integrated over 8 to 13 Hz and summarized over 8 to 12.75 Hz, so a
           centroid and a band power reported for the same band describe
           slightly different stretches of the spectrum. The difference grows as
           the grid coarsens; compute on a finer grid if it matters to you.
        """
        f = np.asarray(freqs, dtype=float)
        return (f >= self.fmin) & (f < self.fmax)


BANDS_STANDARD: tuple[Band, ...] = (
    Band("delta", 1.0, 4.0),
    Band("theta", 4.0, 8.0),
    Band("alpha", 8.0, 13.0),
    Band("beta", 13.0, 30.0),
    Band("gamma", 30.0, 45.0),
)


def passband_fraction(band: Band, highpass: float | None, lowpass: float | None) -> float:
    """Fraction of ``band`` lying inside a recording's filter passband.

    Parameters
    ----------
    band : Band
        The requested band.
    highpass, lowpass : float or None
        The recording's filter edges in Hz, as MNE reports them in
        ``info['highpass']`` and ``info['lowpass']``. None, or a non-finite
        value, means that edge is unknown and does not constrain anything.

    Returns
    -------
    float
        1.0 when the band lies wholly inside the passband, 0.0 when the two are
        disjoint, and the covered fraction in between.
    """
    low = highpass if highpass is not None and np.isfinite(highpass) else -np.inf
    high = lowpass if lowpass is not None and np.isfinite(lowpass) else np.inf
    overlap = min(band.fmax, high) - max(band.fmin, low)
    return float(np.clip(overlap / (band.fmax - band.fmin), 0.0, 1.0))


def check_passband(
    band: Band, highpass: float | None, lowpass: float | None, *, source: str
) -> float:
    """Refuse a band the recording cannot carry, and warn about a truncated one.

    Preprocessing decides what frequencies survive, and asking for a band outside
    that range does not fail: it returns filter roll-off and numerical noise,
    shaped like a real measurement and carrying no signal. MNE tracks the edges
    through filtering, epoching and spectral estimation, so the mismatch is
    detectable rather than merely unfortunate.

    Parameters
    ----------
    band : Band
        The requested band.
    highpass, lowpass : float or None
        The recording's filter edges in Hz.
    source : str
        What is being computed, for the message.

    Returns
    -------
    float
        The covered fraction, as :func:`passband_fraction` defines it.

    Raises
    ------
    ValueError
        When the band lies entirely outside the passband.
    """
    fraction = passband_fraction(band, highpass, lowpass)
    edges = f"[{highpass}, {lowpass}] Hz"
    if fraction <= 0.0:
        raise ValueError(
            f"band {band.name!r} [{band.fmin}, {band.fmax}) lies outside the passband "
            f"{edges} this recording was filtered to, so {source} would measure filter "
            "roll-off rather than signal. Choose a band inside the passband, or pass "
            "data that was not filtered this narrowly."
        )
    if fraction < 1.0:
        warnings.warn(
            f"band {band.name!r} [{band.fmin}, {band.fmax}) extends past the passband "
            f"{edges} this recording was filtered to; {fraction:.0%} of it carries "
            f"signal and {source} is computed over the whole band regardless.",
            UserWarning,
            stacklevel=3,
        )
    return fraction
