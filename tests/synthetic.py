"""Synthetic epochs for the runner tests: a 10 Hz sine locked to every epoch.

The signal is known exactly, so tests can derive expected feature values from it
instead of from the library.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd

SFREQ = 250.0
CHANNELS = ("Fz", "F3", "F4", "Cz", "Pz")
AMPLITUDE = 10e-6
NOISE = 1e-6


def make_epochs(
    n_epochs: int = 12,
    tmin: float = -0.5,
    seconds: float = 2.0,
    channels: Sequence[str] = CHANNELS,
    bads: Sequence[str] = (),
) -> mne.EpochsArray:
    """Epochs of a 10 Hz sine plus white noise, alternating 'left' and 'right' events."""
    rng = np.random.default_rng(0)
    times = tmin + np.arange(int(round(seconds * SFREQ)) + 1) / SFREQ
    sine = AMPLITUDE * np.sin(2 * np.pi * 10.0 * times)
    data = sine + NOISE * rng.standard_normal((n_epochs, len(channels), times.size))
    codes = np.where(np.arange(n_epochs) % 2 == 0, 1, 2)
    events = np.column_stack([np.arange(n_epochs) * 1000, np.zeros(n_epochs, int), codes])
    metadata = pd.DataFrame({"rating": np.arange(n_epochs) % 5})
    info = mne.create_info(list(channels), SFREQ, "eeg")
    info["bads"] = list(bads)
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=tmin,
        event_id={"left": 1, "right": 2},
        metadata=metadata,
        verbose="error",
    )


def save_epochs(path: Path, **kwargs: object) -> Path:
    """Write :func:`make_epochs` to ``path``, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    make_epochs(**kwargs).save(path, overwrite=True, verbose="error")  # type: ignore[arg-type]
    return path
