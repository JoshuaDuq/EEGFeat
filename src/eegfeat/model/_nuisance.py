"""Least-squares nuisance fitting independent of covariate measurement units."""

import numpy as np
import numpy.typing as npt


def fit_coefficients(
    design: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    scales = np.linalg.norm(design, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0.0):
        raise ValueError("Nuisance design requires finite, nonzero columns.")
    coefficients, _, rank, _ = np.linalg.lstsq(design / scales, target, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Nuisance design is rank deficient after scaling.")
    return np.asarray(coefficients / (scales if target.ndim == 1 else scales[:, None]))
