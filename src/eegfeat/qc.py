from __future__ import annotations

import numpy as np
import numpy.typing as npt


def band_coverage(
    coverage: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reduce per-frequency coverage to one value per window.

    Weighted by bin width rather than bin count, so the result reads as the
    fraction of the band's spectral span that was measurable. A count-based
    reduction would understate coverage on a log grid, where a single wide bin
    at the top of a band carries more of the band than several narrow ones at
    the bottom.

    Parameters
    ----------
    coverage : ndarray, shape (n_epochs, n_channels, n_windows, n_freqs)
        Per-frequency coverage in ``[0, 1]``.
    weights : ndarray, shape (n_freqs,)
        Trapezoidal weights for the same frequencies.

    Returns
    -------
    ndarray, shape (n_epochs, n_channels, n_windows)
        Coverage in ``[0, 1]``. Zero when nothing was measurable.
    """
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros(coverage.shape[:3], dtype=float)
    reduced = np.tensordot(coverage, np.asarray(weights, dtype=float), axes=([3], [0])) / total
    return np.clip(reduced, 0.0, 1.0)
