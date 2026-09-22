"""One numerical stage path shared by in-memory and persisted workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    ArtifactModel,
    ReviewedArtifact,
    apply_artifact,
    fit_eog_regression,
    fit_ssp,
    reference_artifact_data,
    review_artifact,
)
from .checks import validate_processing
from .config import (
    AutoRejectSettings,
    BadSpan,
    EventEpochSettings,
    ICASettings,
    ProcessingSettings,
    RegressionSettings,
    SSPSettings,
    ThresholdSettings,
    WorkflowSettings,
)
from .epochs import (
    analysis_bounds,
    baseline_epochs,
    detrend_epochs,
    interpolate_channels,
    make_epochs,
    reference_epochs,
    validate_epochs,
)
from .events import EventData, resolve_events
from .ica import fit_ica
from .provenance import fingerprint, serializable
from .quality import (
    apply_raw_review,
    detect_annotations,
    detect_bad_channels,
    detect_bridges,
    repair_stimulation,
    stimulation_intervals,
)
from .raw import (
    acquisition_spans,
    annotate_raw,
    crop_raw,
    filter_raw,
    notch_raw,
    prepare_channels,
    validate_raw,
)
from .rejection import (
    RejectionModel,
    apply_epoch_review,
    apply_rejection,
    fit_rejection,
    reject_epochs,
)
from .sampling import crop_epochs, resample_epochs
from .stages import STAGES, enabled, get_stage, stage_settings


@dataclass(frozen=True)
class StageData:
    raw: Any  # the continuous recording until the epoch stage, then None
    events: EventData | None = None
    epochs: Any = None
    artifact: ArtifactModel | None = None
    reviewed: ReviewedArtifact | None = None
    rejection: RejectionModel | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingResult:
    epochs: Any
    events: pd.DataFrame
    provenance: dict[str, Any]
    repairs: pd.DataFrame | None = None


def _repair_ledger(repair: dict[str, Any]) -> pd.DataFrame:
    # autoreject labels: 0 good, 1 bad, 2 interpolated; long form keyed by original event row.
    labels = np.asarray(repair["labels"])
    rows, columns = np.nonzero(labels)
    return pd.DataFrame(
        {
            "original_row": np.asarray(repair["original_rows"])[rows],
            "channel": np.asarray(repair["channels"])[columns],
            "label": np.where(labels[rows, columns] == 2, "interpolated", "bad"),
        }
    )


def result_from_state(state: StageData) -> PreprocessingResult:
    validate_epochs(state.epochs)
    if state.events is None:
        raise ValueError("report: missing original events")
    events = state.events
    retained = {int(original): index for index, original in enumerate(state.epochs.selection)}
    labels = {code: name for name, code in events.event_id.items()}
    ledger = pd.DataFrame(
        {
            "original_row": events.original_row,
            "original_sample": events.original_samples,
            "event_sample": events.events[:, 0],
            "event_code": events.events[:, 2],
            "label": [labels.get(int(code), "") for code in events.events[:, 2]],
            "event_sample_sfreq": events.original_sfreq,
            "retained": [index in retained for index in range(len(events.events))],
            "epoch_row": pd.array(
                [retained.get(index) for index in range(len(events.events))], dtype="Int64"
            ),
            "drop_reason": [";".join(reasons) for reasons in state.epochs.drop_log],
        }
    )
    repair = state.provenance.get("repair")
    provenance = {
        **state.provenance,
        "final_sfreq": state.epochs.info["sfreq"],
        "final_bads": state.epochs.info["bads"],
        "final_times": [state.epochs.tmin, state.epochs.tmax],
        "retained": len(state.epochs),
        "original_events": len(events.events),
    }
    if repair is not None:
        # The per-epoch label matrix lives in the repair ledger, not the manifest.
        provenance["repair"] = {key: value for key, value in repair.items() if key != "labels"}
    return PreprocessingResult(
        state.epochs, ledger, provenance, None if repair is None else _repair_ledger(repair)
    )


Operation = Callable[["StageData", ProcessingSettings, Any, int], "StageData"]


def _stage_load(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    raw = state.raw
    validate_raw(raw)
    return replace(
        state,
        raw=raw.copy(),
        provenance={
            **state.provenance,
            "input_hash": fingerprint(raw),
            "original_first_samp": raw.first_samp,
            "original_n_times": raw.n_times,
            "original_sfreq": raw.info["sfreq"],
            "original_description": raw.info["description"],
        },
    )


def _stage_prepare(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    prepared = prepare_channels(state.raw, settings.channels)
    validate_processing(prepared, settings)
    return replace(state, raw=prepared)


def _stage_events(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    epoch_settings = settings.epochs
    metadata = (
        pd.read_csv(epoch_settings.metadata, sep="\t")
        if isinstance(epoch_settings, EventEpochSettings) and epoch_settings.metadata
        else None
    )
    source = (
        epoch_settings.events if isinstance(epoch_settings, EventEpochSettings) else epoch_settings
    )
    return replace(state, events=resolve_events(state.raw, source, metadata=metadata))


def _stage_crop_raw(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.crop is not None
    return replace(state, raw=crop_raw(state.raw, settings.crop))


def _acquisition(state: StageData) -> dict[str, int]:
    # Span onsets are seconds from the first acquired sample, even after crop-raw.
    return {
        "acquisition_first_samp": state.provenance["original_first_samp"],
        "acquisition_n_times": state.provenance["original_n_times"],
    }


def _require_decision(stage: str, decision: Any) -> dict[str, Any]:
    if decision is None:
        raise ValueError(f"{stage}: explicit decision required")
    return dict(decision)


def _stage_annotate(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    raw, events = state.raw, state.events
    annotated = annotate_raw(raw, settings.annotations, **_acquisition(state))
    suggestions = detect_annotations(
        annotated, settings.annotations, None if events is None else events.events
    )
    first_samp = state.provenance["original_first_samp"]
    spans = [
        span
        for annotations in suggestions.annotations
        for span in acquisition_spans(annotated, annotations, first_samp)
    ]
    return replace(
        state,
        raw=annotated,
        candidates={
            "bads": list(suggestions.bads),
            "spans": spans,
            "evidence": serializable(suggestions.evidence),
        },
    )


def _stage_detect_bads(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    candidates = dict(state.candidates)
    if settings.bad_channels is not None:
        found = detect_bad_channels(state.raw, settings.bad_channels)
        candidates["bads"] = list(dict.fromkeys([*candidates.get("bads", []), *found.bads]))
        candidates["bad_channels"] = serializable(found.evidence)
    if settings.bridges:
        candidates["bridges"] = serializable(detect_bridges(state.raw).evidence)
    return replace(state, candidates=candidates)


def _stage_review_raw(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    decision = _require_decision("review-raw", decision)
    spans = tuple(BadSpan(**span) for span in decision["spans"])
    reviewed = apply_raw_review(state.raw, tuple(decision["bads"]), spans, **_acquisition(state))
    return replace(state, raw=reviewed, provenance={**state.provenance, "raw_decision": decision})


def _stage_repair_stim(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.stimulation is not None and state.events is not None
    events = state.events.events
    intervals = stimulation_intervals(state.raw, events, settings.stimulation)
    return replace(
        state,
        raw=repair_stimulation(state.raw, events, settings.stimulation),
        provenance={**state.provenance, "repaired_samples": intervals},
    )


def _stage_notch(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, raw=notch_raw(state.raw, settings.filter, n_jobs=n_jobs))


def _stage_filter(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, raw=filter_raw(state.raw, settings.filter, n_jobs=n_jobs))


def _stage_artifact_reference(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.artifact is not None and settings.artifact.reference is not None
    return replace(state, raw=reference_artifact_data(state.raw, settings.artifact.reference))


def _stage_fit_artifact(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.artifact is not None
    model_settings = settings.artifact.settings
    if isinstance(model_settings, ICASettings):
        model = fit_ica(state.raw, model_settings)
    elif isinstance(model_settings, SSPSettings):
        model = fit_ssp(state.raw, model_settings)
    else:
        assert isinstance(model_settings, RegressionSettings)
        model = fit_eog_regression(state.raw, model_settings)
    record = {"method": model.method, "fit_id": model.fit_id, "evidence": model.evidence}
    return replace(state, artifact=model, provenance={**state.provenance, "artifact": record})


def _stage_review_artifact(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.artifact is not None
    decision = _require_decision("review-artifact", decision)
    return replace(
        state,
        reviewed=review_artifact(state.artifact, decision),
        provenance={**state.provenance, "artifact_decision": decision},
    )


def _stage_epoch(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.events is not None
    thresholds = settings.rejection if isinstance(settings.rejection, ThresholdSettings) else None
    epochs = make_epochs(state.raw, state.events, settings.epochs, thresholds)
    # Nothing downstream reads the continuous data; carrying it would copy it into every
    # later checkpoint.
    return replace(state, raw=None, epochs=epochs)


def _stage_apply_artifact(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.reviewed is not None
    return replace(state, epochs=apply_artifact(state.epochs, state.reviewed))


def _stage_fit_rejection(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert isinstance(settings.rejection, AutoRejectSettings) and state.events is not None
    tmin, tmax = analysis_bounds(settings.epochs, state.events.original_sfreq)
    return replace(state, rejection=fit_rejection(state.epochs, settings.rejection, tmin, tmax))


def _stage_reject(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    if isinstance(settings.rejection, ThresholdSettings):
        return replace(state, epochs=reject_epochs(state.epochs, settings.rejection))
    assert state.rejection is not None
    cleaned, log = apply_rejection(state.epochs, state.rejection)
    # autoreject labels channels it did not fit with NaN; keep only fitted channels.
    fitted = np.isfinite(log.labels).all(axis=0)
    repair = {
        "original_rows": state.epochs.selection,
        "channels": [name for name, keep in zip(log.ch_names, fitted, strict=True) if keep],
        "bad_epochs": log.bad_epochs,
        "labels": log.labels[:, fitted],
    }
    return replace(state, epochs=cleaned, provenance={**state.provenance, "repair": repair})


def _stage_review_epochs(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    decision = _require_decision("review-epochs", decision)
    return replace(
        state,
        epochs=apply_epoch_review(state.epochs, tuple(decision["exclude"])),
        provenance={**state.provenance, "epoch_decision": decision},
    )


def _stage_interpolate(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, epochs=interpolate_channels(state.epochs))


def _stage_reference(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    return replace(state, epochs=reference_epochs(state.epochs, settings.reference))


def _stage_resample(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.sampling is not None
    epochs = resample_epochs(state.epochs, settings.sampling, settings.filter, n_jobs=n_jobs)
    return replace(state, epochs=epochs)


def _stage_crop_epochs(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert state.events is not None
    bounds = analysis_bounds(settings.epochs, state.events.original_sfreq)
    return replace(state, epochs=crop_epochs(state.epochs, *bounds))


def _stage_detrend(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.epochs.detrend is not None
    return replace(state, epochs=detrend_epochs(state.epochs, settings.epochs.detrend))


def _stage_baseline(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    assert settings.epochs.baseline is not None
    return replace(state, epochs=baseline_epochs(state.epochs, settings.epochs.baseline))


def _stage_report(
    state: StageData, settings: ProcessingSettings, decision: Any, n_jobs: int
) -> StageData:
    result_from_state(state)
    return state


OPERATIONS: dict[str, Operation] = {
    "load": _stage_load,
    "prepare": _stage_prepare,
    "events": _stage_events,
    "crop-raw": _stage_crop_raw,
    "annotate": _stage_annotate,
    "detect-bads": _stage_detect_bads,
    "review-raw": _stage_review_raw,
    "repair-stim": _stage_repair_stim,
    "notch": _stage_notch,
    "filter": _stage_filter,
    "artifact-reference": _stage_artifact_reference,
    "fit-artifact": _stage_fit_artifact,
    "review-artifact": _stage_review_artifact,
    "epoch": _stage_epoch,
    "apply-artifact": _stage_apply_artifact,
    "fit-rejection": _stage_fit_rejection,
    "reject": _stage_reject,
    "review-epochs": _stage_review_epochs,
    "interpolate": _stage_interpolate,
    "reference": _stage_reference,
    "resample": _stage_resample,
    "crop-epochs": _stage_crop_epochs,
    "detrend": _stage_detrend,
    "baseline": _stage_baseline,
    "report": _stage_report,
    "export": _stage_report,
}


def execute_numeric(
    stage: str,
    state: StageData,
    settings: ProcessingSettings,
    decision: dict[str, Any] | None = None,
    *,
    n_jobs: int = 1,
) -> StageData:
    definition = get_stage(stage)
    provenance = {
        **state.provenance,
        "stages": [*state.provenance.get("stages", []), stage],
        "settings": {
            **state.provenance.get("settings", {}),
            **stage_settings(definition, settings),
        },
    }
    return OPERATIONS[stage](replace(state, provenance=provenance), settings, decision, n_jobs)


def preprocess(
    raw: Any,
    settings: ProcessingSettings,
    *,
    decisions: dict[str, dict[str, Any]] | None = None,
    n_jobs: int = 1,
) -> PreprocessingResult:
    # Same catalog as the checkpointed path; a review stage runs only with its decision.
    reviews = {} if decisions is None else decisions
    workflow = WorkflowSettings(
        raw_review="required" if "review-raw" in reviews else "disabled",
        epoch_review="required" if "review-epochs" in reviews else "disabled",
    )
    state = StageData(
        raw,
        provenance={
            "settings": serializable(settings),
            "raw_review": workflow.raw_review,
            "artifact_review": workflow.artifact_review,
        },
    )
    for stage in STAGES:
        if enabled(stage.name, settings, workflow) and stage.name not in ("report", "export"):
            state = execute_numeric(
                stage.name, state, settings, reviews.get(stage.name), n_jobs=n_jobs
            )
    return result_from_state(state)
