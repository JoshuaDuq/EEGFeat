"""Epoch construction and explicit final transforms."""

from __future__ import annotations

from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
from scipy.signal import detrend

from .config import EventEpochSettings, FixedEpochSettings, ReferenceSettings, ThresholdSettings
from .events import EventData, fixed_bounds
from .raw import good_eeg_names, validate_geometry


def analysis_bounds(
    settings: EventEpochSettings | FixedEpochSettings, sfreq: float
) -> tuple[float, float]:
    if isinstance(settings, FixedEpochSettings):
        return fixed_bounds(settings, sfreq)
    return settings.tmin, settings.tmax


def validate_epochs(epochs: Any) -> None:
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("epochs: expected MNE BaseEpochs")
    if not len(epochs):
        raise ValueError(f"epochs: all epochs rejected; drop log: {epochs.drop_log}")
    good_eeg_names(epochs)
    if not np.isfinite(epochs.get_data()).all():
        raise ValueError("epochs: nonfinite samples")


def make_epochs(
    raw: Any,
    events: EventData,
    settings: EventEpochSettings | FixedEpochSettings,
    rejection: ThresholdSettings | None = None,
) -> Any:
    # The rejection window is fixed here, on the analysis bounds, not on the padding.
    tmin, tmax = analysis_bounds(settings, events.original_sfreq)
    reject_tmin = tmin if rejection is None or rejection.tmin is None else rejection.tmin
    reject_tmax = tmax if rejection is None or rejection.tmax is None else rejection.tmax
    if not tmin <= reject_tmin <= reject_tmax <= tmax:
        raise ValueError("rejection: window must be inside the analysis interval")
    epochs = mne.Epochs(
        raw,
        events.events,
        dict(events.event_id),
        tmin=tmin - settings.padding,
        tmax=tmax + settings.padding,
        baseline=None,
        picks=list(raw.ch_names),
        preload=True,
        reject=None,
        flat=None,
        proj=False,
        decim=1,
        reject_tmin=reject_tmin,
        reject_tmax=reject_tmax,
        detrend=None,
        on_missing="raise",
        reject_by_annotation=True,
        metadata=events.metadata,
        event_repeated="error",
    )
    validate_epochs(epochs)
    return epochs


def interpolate_channels(epochs: Any) -> Any:
    # Only EEG bads are targets; EOG/ECG bad labels are restored afterwards.
    validate_epochs(epochs)
    working = epochs.copy()
    eeg = set(
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types(), strict=True)
        if kind == "eeg"
    )
    targets = [name for name in epochs.info["bads"] if name in eeg]
    if not targets:
        return working
    validate_geometry(epochs)
    other = [name for name in epochs.info["bads"] if name not in eeg]
    working.info["bads"] = targets
    working.interpolate_bads(reset_bads=True, method={"eeg": "spline"})
    working.info["bads"] = other
    return working


def reference_epochs(epochs: Any, settings: ReferenceSettings) -> Any:
    working = epochs.copy().load_data()
    if settings.add_channels:
        overlap = set(settings.add_channels) & set(working.ch_names)
        if overlap:
            raise ValueError(f"reference.add_channels: already present {sorted(overlap)}")
        working = mne.add_reference_channels(working, list(settings.add_channels), copy=False)
    if settings.channels is None:
        return working
    good = good_eeg_names(working)
    if settings.channels == "average":
        if len(good) < 2:
            raise ValueError("reference.channels: average requires >=2 good EEG channels")
        reference: str | list[str] = "average"
    else:
        if not set(settings.channels) <= set(good):
            raise ValueError(
                "reference.channels: reference must contain only good retained EEG names"
            )
        reference = list(settings.channels)
    working.set_eeg_reference(ref_channels=reference, projection=False)
    return working


def detrend_epochs(epochs: Any, method: str) -> Any:
    if method not in ("constant", "linear"):
        raise ValueError("epochs.detrend: expected constant or linear")
    picks = [
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types(), strict=True)
        if kind == "eeg"
    ]
    return epochs.copy().apply_function(detrend, picks=picks, type=method, channel_wise=False)


def baseline_epochs(epochs: Any, baseline: tuple[float | None, float | None]) -> Any:
    # Like MNE, accept bounds up to one sample outside the rounded final grid.
    tstep = 1.0 / epochs.info["sfreq"]
    start = epochs.times[0] if baseline[0] is None else baseline[0]
    stop = epochs.times[-1] if baseline[1] is None else baseline[1]
    if start < epochs.times[0] - tstep or stop > epochs.times[-1] + tstep:
        raise ValueError("epochs.baseline: bounds must be within final epoch times")
    return epochs.copy().apply_baseline(baseline)
