"""Reviewed artifact operators with explicit fit/application compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mne  # type: ignore[import-untyped]

from ._validation import integer, mapping
from .config import ReferenceSettings, RegressionSettings, SSPSettings
from .epochs import reference_epochs
from .provenance import channel_identity, fingerprint, identity
from .raw import good_eeg_names, require_names


@dataclass(frozen=True)
class ArtifactModel:
    method: str
    model: Any
    fit_id: str
    compatibility: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewedArtifact:
    artifact: ArtifactModel
    decision: dict[str, Any]


def reference_artifact_data(raw: Any, reference: str | tuple[str, ...]) -> Any:
    return reference_epochs(raw, ReferenceSettings(channels=reference))


def fit_ssp(raw: Any, settings: SSPSettings) -> ArtifactModel:
    good_eeg_names(raw)
    channels = [*settings.eog_channels, *([settings.ecg_channel] if settings.ecg_channel else [])]
    require_names(raw, channels, "artifact.ssp")
    if settings.h_freq >= raw.info["sfreq"] / 2:
        raise ValueError("artifact.ssp.h_freq: exceeds Nyquist")
    projectors, events = [], []
    # no_proj keeps projectors already in raw out of the reviewed candidate list.
    common = dict(
        n_mag=0,
        n_grad=0,
        n_eeg=settings.n_eeg,
        l_freq=settings.l_freq,
        h_freq=settings.h_freq,
        tmin=settings.tmin,
        tmax=settings.tmax,
        reject=settings.reject,
        no_proj=True,
    )
    for channel in settings.eog_channels:
        found, detected = mne.preprocessing.compute_proj_eog(raw, ch_name=channel, **common)
        if not len(detected):
            raise ValueError("artifact.ssp: no EOG training events")
        projectors.extend(found)
        events.extend(detected.tolist())
    if settings.ecg_channel is not None:
        found, detected = mne.preprocessing.compute_proj_ecg(
            raw, ch_name=settings.ecg_channel, **common
        )
        if not len(detected):
            raise ValueError("artifact.ssp: no ECG training events")
        projectors.extend(found)
        events.extend(detected.tolist())
    # matrix_rank over-reports the rank of average-referenced data; MNE's estimator does not.
    rank = int(mne.compute_rank(raw, proj=False)["eeg"])
    if not projectors or len(projectors) >= rank:
        raise ValueError(
            "artifact.ssp.n_eeg: total projectors must be positive and below usable rank"
        )
    fit_id = identity({"data": fingerprint(raw), "settings": settings, "method": "ssp"})
    return ArtifactModel("ssp", projectors, fit_id, channel_identity(raw), {"events": events})


def fit_eog_regression(raw: Any, settings: RegressionSettings) -> ArtifactModel:
    if not raw.info["custom_ref_applied"]:
        raise ValueError("artifact.reference: regression requires explicitly referenced EEG")
    require_names(raw, list(settings.eog_channels), "artifact.regression.eog_channels")
    events = mne.make_fixed_length_events(raw, duration=settings.tstep)
    epochs = mne.Epochs(
        raw,
        events,
        {"training": 1},
        tmin=0,
        tmax=(round(settings.tstep * raw.info["sfreq"]) - 1) / raw.info["sfreq"],
        baseline=None,
        proj=False,
        preload=True,
        reject=settings.reject,
        flat=settings.flat,
        picks=raw.ch_names,
        reject_by_annotation=True,
    )
    if not len(epochs):
        raise ValueError("artifact.regression: no usable training epochs")
    model = mne.preprocessing.EOGRegression(
        picks=good_eeg_names(raw), picks_artifact=list(settings.eog_channels), proj=False
    ).fit(epochs)
    fit_id = identity({"data": fingerprint(raw), "settings": settings, "method": "regression"})
    return ArtifactModel(
        "regression", model, fit_id, channel_identity(raw), {"training_epochs": len(epochs)}
    )


def review_artifact(model: ArtifactModel, decision: dict[str, Any]) -> ReviewedArtifact:
    field_name = {"ica": "exclude", "ssp": "include", "regression": "apply"}[model.method]
    mapping(decision, "review.artifact", {"fit_id", field_name})
    if decision.get("fit_id") != model.fit_id:
        raise ValueError("review.artifact.fit_id: decision refers to a different fit")
    selection = decision.get(field_name)
    if model.method == "regression":
        if type(selection) is not bool:
            raise ValueError("review.artifact.apply: explicit true or false required")
    else:
        if not isinstance(selection, list):
            raise ValueError(
                f"review.artifact.{field_name}: explicit list required; [] keeps all data"
            )
        count = model.model.n_components_ if model.method == "ica" else len(model.model)
        for index in selection:
            integer(index, f"review.artifact.{field_name}")
            if index >= count:
                raise ValueError(
                    f"review.artifact.{field_name}: index {index} outside 0..{count - 1}"
                )
        if len(set(selection)) != len(selection):
            raise ValueError(f"review.artifact.{field_name}: duplicate indices")
    return ReviewedArtifact(model, dict(decision))


def apply_artifact(inst: Any, reviewed: ReviewedArtifact) -> Any:
    artifact = reviewed.artifact
    review_artifact(artifact, reviewed.decision)
    if identity(channel_identity(inst)) != identity(artifact.compatibility):
        raise ValueError(
            "artifact: incompatible channel order, bad labels, reference or sample rate"
        )
    result = inst.copy().load_data()
    # An empty choice leaves the samples exactly untouched instead of a PCA round trip.
    if artifact.method == "ica":
        if reviewed.decision["exclude"]:
            artifact.model.copy().apply(result, exclude=reviewed.decision["exclude"])
    elif artifact.method == "ssp":
        selected = [artifact.model[index] for index in reviewed.decision["include"]]
        if selected:
            result.add_proj(selected).apply_proj()
    elif reviewed.decision["apply"]:
        result = artifact.model.apply(result, copy=True)
    return result
