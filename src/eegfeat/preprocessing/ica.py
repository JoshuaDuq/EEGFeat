"""Seeded ICA fitting on a separate high-pass training copy."""

from __future__ import annotations

from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np

from ._deps import require
from .artifacts import ArtifactModel
from .config import ICLABEL_CLASSES, FilterSettings, ICASettings
from .provenance import channel_identity, fingerprint, identity
from .raw import filter_raw, good_eeg_names, require_names, validate_geometry

# ICLabel expects extended infomax; picard reproduces it with ortho off and extended on.
METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "fastica": ("fastica", {}),
    "infomax": ("infomax", {"extended": True}),
    "picard": ("picard", {"ortho": False, "extended": True}),
}


def fit_ica(raw: Any, settings: ICASettings) -> ArtifactModel:
    require("sklearn", "preprocessing")
    if settings.method == "picard":
        require("picard", "preprocessing-auto")
    if settings.iclabel is not None:
        require("mne_icalabel", "preprocessing-auto")
    validate_geometry(raw)
    picks = good_eeg_names(raw)
    channels = [*settings.eog_channels, *([settings.ecg_channel] if settings.ecg_channel else [])]
    require_names(raw, channels, "artifact.ica")
    # Train on a separate high-pass copy; only add the missing high-pass, never lower one.
    training = (
        raw.copy()
        if raw.info["highpass"] >= settings.l_freq
        else filter_raw(raw, FilterSettings(l_freq=settings.l_freq))
    )
    rank = int(mne.compute_rank(training, proj=False)["eeg"])
    components = rank if settings.n_components is None else settings.n_components
    if rank < 2 or components > rank:
        raise ValueError(f"artifact.ica.n_components: requested {components}, usable rank {rank}")
    method, fit_params = METHODS[settings.method]
    model = mne.preprocessing.ICA(
        n_components=components,
        method=method,
        fit_params=fit_params,
        rng=settings.random_state,
        max_iter=settings.max_iter,
    )
    model.fit(
        training,
        picks=picks,
        reject=settings.reject,
        flat=settings.flat,
        tstep=settings.tstep,
        reject_by_annotation=True,
    )
    if model.n_iter_ >= settings.max_iter:
        raise ValueError("artifact.ica.max_iter: ICA did not converge")
    # Scores and detector verdicts are evidence for review; nothing is excluded here.
    scores: dict[str, Any] = {}
    suggested: dict[str, Any] = {}
    for channel in settings.eog_channels:
        found, scores[channel] = model.find_bads_eog(training, ch_name=channel)
        suggested[channel] = [int(index) for index in found]
    if settings.ecg_channel is not None:
        found, scores[settings.ecg_channel] = model.find_bads_ecg(
            training, ch_name=settings.ecg_channel, method="correlation"
        )
        suggested[settings.ecg_channel] = [int(index) for index in found]
    evidence: dict[str, Any] = {"rank": rank, "scores": scores}
    if settings.iclabel is not None:
        evidence["iclabel"] = _label_components(training, model, settings)
        suggested["iclabel"] = evidence["iclabel"]["suggested"]
    evidence["suggested"] = suggested
    evidence["suggested_exclude"] = sorted({int(i) for found in suggested.values() for i in found})
    fit_id = identity(
        {
            "data": fingerprint(raw),
            "training": fingerprint(training),
            "settings": settings,
            "method": "ica",
        }
    )
    return ArtifactModel("ica", model, fit_id, channel_identity(raw), evidence)


def _label_components(training: Any, model: Any, settings: ICASettings) -> dict[str, Any]:
    from mne_icalabel.iclabel import (  # type: ignore[import-untyped]
        iclabel_label_components,
    )

    assert settings.iclabel is not None
    # inplace=False keeps the classifier's verdict out of the saved ICA object.
    probabilities = np.asarray(
        iclabel_label_components(training, model.copy(), inplace=False), dtype=float
    )
    if probabilities.shape != (model.n_components_, len(ICLABEL_CLASSES)):
        raise ValueError("artifact.ica.iclabel: unexpected classifier output shape")
    winners = probabilities.argmax(axis=1)
    labels = [ICLABEL_CLASSES[index] for index in winners]
    confidence = probabilities[np.arange(len(winners)), winners]
    suggested = [
        int(index)
        for index, (label, score) in enumerate(zip(labels, confidence, strict=True))
        if label not in settings.iclabel.keep and score >= settings.iclabel.threshold
    ]
    return {
        "classes": list(ICLABEL_CLASSES),
        "labels": labels,
        "probabilities": probabilities,
        "suggested": suggested,
    }
