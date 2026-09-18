from __future__ import annotations

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
