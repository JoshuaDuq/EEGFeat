"""Separate rejection fitting, transformation, and explicit manual exclusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deps import require
from .config import AutoRejectSettings, ThresholdSettings
from .epochs import validate_epochs
from .raw import good_eeg_names, validate_geometry


@dataclass(frozen=True)
class RejectionModel:
    model: Any
    channels: tuple[str, ...]
    sfreq: float
    tmin: float
    tmax: float


def fit_rejection(
    epochs: Any, settings: AutoRejectSettings, tmin: float, tmax: float
) -> RejectionModel:
    autoreject = require("autoreject", "preprocessing-auto")
    validate_epochs(epochs)
    validate_geometry(epochs)
    picks = good_eeg_names(epochs)
    if len(epochs) < settings.cv:
        raise ValueError("rejection.cv: more folds than usable epochs")
    if max(settings.n_interpolate) >= len(picks):
        raise ValueError("rejection.n_interpolate: must be below good EEG channel count")
    model = autoreject.AutoReject(
        n_interpolate=list(settings.n_interpolate),
        consensus=list(settings.consensus),
        cv=settings.cv,
        random_state=settings.random_state,
        picks=picks,
        n_jobs=1,
    )
    # Fit on the analysis window only; padding samples must not drive rejection.
    model.fit(epochs.copy().crop(tmin, tmax))
    return RejectionModel(model, tuple(epochs.ch_names), float(epochs.info["sfreq"]), tmin, tmax)


def reject_epochs(epochs: Any, settings: ThresholdSettings) -> Any:
    validate_epochs(epochs)
    present = set(epochs.get_channel_types())
    if not set(settings.reject or {}) | set(settings.flat or {}) <= present:
        raise ValueError("rejection: threshold names absent channel type")
    # The decision window was fixed at construction (reject_tmin/reject_tmax) and
    # survives FIF round trips, so it is not restated here.
    working = epochs.copy().drop_bad(reject=settings.reject, flat=settings.flat)
    validate_epochs(working)
    return working


def apply_rejection(epochs: Any, model: RejectionModel) -> tuple[Any, Any]:
    if tuple(epochs.ch_names) != model.channels or epochs.info["sfreq"] != model.sfreq:
        raise ValueError("rejection: incompatible channels or sample rate")
    # Decide on the fitted window, then repair the padded epochs with that same log.
    log = model.model.get_reject_log(epochs.copy().crop(model.tmin, model.tmax))
    result = model.model.transform(epochs.copy(), reject_log=log)
    validate_epochs(result)
    return result, log


def apply_epoch_review(epochs: Any, original_ids: tuple[int, ...]) -> Any:
    if any(type(value) is not int for value in original_ids) or len(set(original_ids)) != len(
        original_ids
    ):
        raise ValueError("review.epochs: expected unique integer original event IDs")
    # IDs are original event rows (epochs.selection), never displayed row positions.
    missing = set(original_ids) - set(epochs.selection)
    if missing:
        raise ValueError(f"review.epochs: original event IDs already absent: {sorted(missing)}")
    working = epochs.copy().drop(
        np.flatnonzero(np.isin(epochs.selection, original_ids)), reason="USER"
    )
    validate_epochs(working)
    return working
