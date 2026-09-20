from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def validate_fraction_array(values: npt.NDArray[np.float64], name: str) -> None:
    """Require finite fractions on the closed unit interval."""
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1].")


def validate_nonempty_shape(shape: tuple[int, ...], name: str) -> None:
    """Reject arrays with an empty scientific axis."""
    if any(length == 0 for length in shape):
        raise ValueError(f"{name} axes must be non-empty, got shape {shape}.")


def validate_names(names: Sequence[str], name: str) -> None:
    """Require non-empty, unique labels."""
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must be unique.")


def blank_non_finite(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Non-finite samples replaced by NaN, so a nan-aware reduction really skips them.

    Coverage and every kernel's finiteness guard call a sample missing when it is
    not finite, but ``np.nanmean`` and friends skip only NaN: an infinity survives
    the very average those guards excluded it from, and ``np.nanvar`` turns the
    whole window into NaN instead of dropping the one bad sample.
    """
    finite = np.isfinite(values)
    # Clean data is the common case and is handed back untouched: the test costs a
    # pass over a bool array, where blanking costs a second float array the size of
    # the recording.
    if bool(finite.all()):
        return np.asarray(values, dtype=np.float64)
    return np.asarray(np.where(finite, values, np.nan), dtype=np.float64)
