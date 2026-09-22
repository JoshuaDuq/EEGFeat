"""Fixed scientific dependency graph and configuration ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AutoRejectSettings, ProcessingSettings, WorkflowSettings
from .provenance import serializable


@dataclass(frozen=True)
class Stage:
    name: str
    parents: tuple[str, ...]
    fields: tuple[str, ...]
    output: str
    review: bool = False


STAGES = (
    Stage("load", (), (), "Raw"),
    Stage("prepare", ("load",), ("channels",), "Raw"),
    Stage(
        "events",
        ("prepare",),
        (
            "epochs.events",
            "epochs.kind",
            "epochs.duration",
            "epochs.overlap",
            "epochs.start",
            "epochs.stop",
            "epochs.metadata",
        ),
        "EventData",
    ),
    Stage("crop-raw", ("events",), ("crop",), "Raw"),
    Stage("annotate", ("crop-raw",), ("annotations",), "QualityCandidates"),
    Stage("detect-bads", ("annotate",), ("bad_channels", "bridges"), "QualityCandidates"),
    Stage("review-raw", ("detect-bads",), (), "Raw", True),
    Stage("repair-stim", ("review-raw",), ("stimulation",), "Raw"),
    Stage("notch", ("repair-stim",), ("filter.notch_freqs",), "Raw"),
    Stage("filter", ("notch",), ("filter.l_freq", "filter.h_freq"), "Raw"),
    Stage("artifact-reference", ("filter",), ("artifact.reference",), "Raw"),
    Stage(
        "fit-artifact",
        ("artifact-reference",),
        ("artifact.method", "artifact.settings"),
        "ArtifactModel",
    ),
    Stage("review-artifact", ("fit-artifact",), (), "ReviewedArtifact", True),
    Stage(
        "epoch",
        ("artifact-reference",),
        (
            "epochs.kind",
            "epochs.duration",
            "epochs.tmin",
            "epochs.tmax",
            "epochs.padding",
            "rejection.tmin",
            "rejection.tmax",
        ),
        "Epochs",
    ),
    Stage("apply-artifact", ("epoch", "review-artifact"), (), "Epochs"),
    Stage("fit-rejection", ("apply-artifact",), ("rejection",), "RejectionModel"),
    Stage("reject", ("fit-rejection",), ("rejection",), "Epochs"),
    Stage("review-epochs", ("reject",), (), "Epochs", True),
    Stage("interpolate", ("review-epochs",), ("channels.interpolate_bads",), "Epochs"),
    Stage("reference", ("interpolate",), ("reference",), "Epochs"),
    Stage("resample", ("reference",), ("sampling", "filter.h_freq"), "Epochs"),
    Stage(
        "crop-epochs", ("resample",), ("epochs.tmin", "epochs.tmax", "epochs.duration"), "Epochs"
    ),
    Stage("detrend", ("crop-epochs",), ("epochs.detrend",), "Epochs"),
    Stage("baseline", ("detrend",), ("epochs.baseline",), "Epochs"),
    Stage("report", ("baseline",), (), "Report"),
    Stage("export", ("report",), (), "Bundle"),
)
REGISTRY = {stage.name: stage for stage in STAGES}


def get_stage(name: str) -> Stage:
    if name not in REGISTRY:
        raise ValueError(f'stage: unknown {name!r}; choose from {", ".join(REGISTRY)}')
    return REGISTRY[name]


def enabled(name: str, settings: ProcessingSettings, workflow: WorkflowSettings) -> bool:
    get_stage(name)
    choices = {
        "crop-raw": settings.crop is not None,
        "detect-bads": settings.bad_channels is not None or settings.bridges,
        "review-raw": workflow.raw_review in ("required", "suggested"),
        "repair-stim": settings.stimulation is not None,
        "notch": bool(settings.filter.notch_freqs),
        "filter": settings.filter.l_freq is not None or settings.filter.h_freq is not None,
        "artifact-reference": settings.artifact is not None
        and settings.artifact.reference is not None,
        "fit-artifact": settings.artifact is not None,
        "review-artifact": settings.artifact is not None,
        "apply-artifact": settings.artifact is not None,
        "fit-rejection": isinstance(settings.rejection, AutoRejectSettings),
        "reject": settings.rejection is not None,
        "review-epochs": workflow.epoch_review == "required",
        "interpolate": settings.channels.interpolate_bads,
        "reference": settings.reference.channels is not None,
        "resample": settings.sampling is not None,
        "detrend": settings.epochs.detrend is not None,
        "baseline": settings.epochs.baseline is not None,
    }
    return choices.get(name, True)


def stage_settings(stage: Stage, settings: ProcessingSettings) -> dict[str, Any]:
    data = serializable(settings)
    result = {}
    for path in stage.fields:
        value = data
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        result[path] = value
    if stage.name == "prepare":
        # interpolate_bads belongs to the interpolate stage, not channel preparation.
        result["channels"] = {
            key: value for key, value in result["channels"].items() if key != "interpolate_bads"
        }
    return result
