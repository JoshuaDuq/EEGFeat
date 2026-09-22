"""Original-grid event identities and sample-exact fixed epoch geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import EventSettings, FixedEpochSettings
from .raw import validate_raw


@dataclass(frozen=True)
class EventData:
    events: NDArray[np.int64]
    event_id: dict[str, int]
    metadata: pd.DataFrame | None
    original_sfreq: float
    original_row: NDArray[np.int64]
    original_samples: NDArray[np.int64]
    delay: float = 0.0
    shift_samples: int = 0


def validate_events(raw: Any, events: NDArray[Any], event_id: dict[str, int]) -> None:
    if (
        events.ndim != 2
        or events.shape[1] != 3
        or len(events) == 0
        or events.dtype.kind not in "iu"
    ):
        raise ValueError("events: expected nonempty integer (n, 3) array")
    if np.any(np.diff(events[:, 0]) <= 0):
        raise ValueError("events: samples must be strictly increasing and unique")
    if events[0, 0] < raw.first_samp or events[-1, 0] >= raw.first_samp + raw.n_times:
        raise ValueError("events: sample outside recording")
    missing = set(event_id.values()) - set(events[:, 2])
    if missing:
        raise ValueError(f"events.event_id: missing codes {sorted(missing)}")


def fixed_bounds(settings: FixedEpochSettings, sfreq: float) -> tuple[float, float]:
    samples = round(settings.duration * sfreq)
    stride = (settings.duration - settings.overlap) * sfreq
    if samples < 2 or not np.isclose(samples, settings.duration * sfreq, atol=1e-9, rtol=0):
        raise ValueError(
            "epochs.duration: must align with acquisition samples and contain >=2 samples"
        )
    if not np.isclose(stride, round(stride), atol=1e-9, rtol=0):
        raise ValueError("epochs.overlap: stride must align with acquisition samples")
    return 0.0, (samples - 1) / sfreq


def resolve_events(
    raw: Any,
    settings: EventSettings | FixedEpochSettings,
    *,
    events: NDArray[Any] | None = None,
    metadata: pd.DataFrame | None = None,
) -> EventData:
    validate_raw(raw)
    sfreq = float(raw.info["sfreq"])
    if isinstance(settings, FixedEpochSettings):
        fixed_bounds(settings, sfreq)
        if events is not None:
            raise ValueError("events: explicit array incompatible with fixed epochs")
        if settings.stop is not None and settings.stop > raw.n_times / sfreq:
            raise ValueError("epochs.stop: exceeds acquisition duration")
        resolved = mne.make_fixed_length_events(
            raw,
            start=settings.start,
            stop=settings.stop,
            duration=settings.duration,
            overlap=settings.overlap,
            first_samp=True,
        )
        event_id, delay = {"fixed": 1}, 0.0
    else:
        event_id, delay = dict(settings.event_id), settings.delay
        if events is not None:
            resolved = np.asarray(events).copy()
        elif settings.source == "annotations":
            resolved, _ = mne.events_from_annotations(raw, event_id=event_id, use_rounding=True)
        elif settings.source == "stim":
            resolved = mne.find_events(
                raw,
                stim_channel=settings.stim_channel,
                shortest_event=settings.shortest_event,
                min_duration=settings.min_duration,
            )
        else:
            resolved = mne.read_events(settings.path)
    validate_events(raw, resolved, event_id)
    original = resolved[:, 0].copy()
    shift = round(delay * sfreq)
    resolved = resolved.astype(np.int64, copy=True)
    resolved[:, 0] -= shift
    validate_events(raw, resolved, event_id)
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame) or len(metadata) != len(resolved):
            raise ValueError("epochs.metadata: one row per input event required")
        metadata = metadata.copy().reset_index(drop=True)
    return EventData(
        resolved,
        event_id,
        metadata,
        sfreq,
        np.arange(len(resolved), dtype=np.int64),
        original,
        delay,
        shift,
    )
