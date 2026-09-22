"""Identity-bound headless review and disposable native MNE viewers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._deps import require
from ._validation import mapping
from .checkpoints import write_pointer
from .config import BadSpan, read_yaml
from .execution import (
    Workflow,
    decision_path,
    enabled_parents,
    pending_path,
    read_decision,
    read_stage,
    source_identity,
    stage_identities,
    suggested_decision,
)
from .pipeline import execute_numeric
from .provenance import identity
from .quality import apply_raw_review
from .stages import enabled


class ReviewCancelled(ValueError):
    pass


def _confirm_choices(title: str, labels: list[str]) -> list[int]:
    require("PyQt6", "preprocessing-gui")
    from PyQt6.QtWidgets import (  # type: ignore[import-not-found]
        QApplication,
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QVBoxLayout,
    )

    application = QApplication.instance() or QApplication([])
    dialog = QDialog()
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    boxes = [QCheckBox(label) for label in labels]
    for box in boxes:
        layout.addWidget(box)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        raise ReviewCancelled("review: cancelled; no decision saved")
    application.processEvents()
    return [index for index, box in enumerate(boxes) if box.isChecked()]


def new_bad_spans(before: Any, after: Any, acquisition_first_samp: int) -> list[dict[str, Any]]:
    # Only spans the reviewer added are the decision; the ones already on the record would be
    # appended twice. Onsets are acquisition seconds, the same clock as annotations.bad_spans.
    def rows(inst: Any) -> list[tuple[float, float, str]]:
        annotations = inst.annotations
        return [
            (float(onset), float(duration), str(description))
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description, strict=True
            )
        ]

    origin = acquisition_first_samp / after.info["sfreq"]
    existing = set(rows(before))
    return [
        {"onset": onset - origin, "duration": duration, "description": description}
        for onset, duration, description in rows(after)
        if description.upper().startswith("BAD") and (onset, duration, description) not in existing
    ]


def viewer_decision(stage: str, state: Any) -> dict[str, Any]:
    # Viewers get disposable copies; a decision exists only after explicit confirmation.
    require("mne_qt_browser", "preprocessing-gui")
    require("PyQt6", "preprocessing-gui")
    import mne  # type: ignore[import-untyped]

    with mne.viz.use_browser_backend("qt"):
        if stage == "review-raw":
            # The reviewer starts from the detectors' verdict and edits it in the viewer.
            suggested = suggested_decision(stage, state)
            first_samp = state.provenance["original_first_samp"]
            raw = apply_raw_review(
                state.raw,
                tuple(suggested["bads"]),
                tuple(BadSpan(**span) for span in suggested["spans"]),
                acquisition_first_samp=first_samp,
                acquisition_n_times=state.provenance["original_n_times"],
            )
            raw.plot(block=True)
            _confirm_choices("Save raw review?", [])
            return {
                "bads": list(raw.info["bads"]),
                "spans": new_bad_spans(state.raw, raw, first_samp),
            }
        if stage == "review-epochs":
            epochs = state.epochs.copy()
            original = set(epochs.selection)
            epochs.plot(block=True)
            _confirm_choices("Save epoch review?", [])
            return {"exclude": sorted(int(value) for value in original - set(epochs.selection))}
        artifact = state.artifact
        if artifact.method == "ica":
            model = artifact.model.copy()
            model.plot_components(picks=list(range(model.n_components_)))
            model.plot_sources(state.raw.copy(), block=True)
            selected = _confirm_choices(
                "Exclude ICA components", [str(index) for index in range(model.n_components_)]
            )
            return {"fit_id": artifact.fit_id, "exclude": selected}
        if artifact.method == "ssp":
            mne.viz.plot_projs_topomap(artifact.model, info=state.raw.info)
            selected = _confirm_choices(
                "Apply SSP projectors", [str(index) for index in range(len(artifact.model))]
            )
            return {"fit_id": artifact.fit_id, "include": selected}
        artifact.model.plot()
        selected = _confirm_choices("Apply EOG regression", ["Apply reviewed regression operator"])
        return {"fit_id": artifact.fit_id, "apply": bool(selected)}


def save_review(
    workflow: Workflow, target: str, decisions: Path | None = None, *, suggested: bool = False
) -> Path:
    if target not in ("raw", "artifact", "epochs"):
        raise ValueError("review: expected raw, artifact, or epochs")
    if suggested and decisions is not None:
        raise ValueError("review: --suggested and --decisions are exclusive")
    stage = f"review-{target}"
    if not enabled(stage, workflow.config.processing, workflow.config.workflow):
        raise ValueError(f"{stage}: disabled by the workflow policy; no decision applies")
    identities = stage_identities(workflow, source_identity(workflow) or "pending")
    parents = {
        name: read_stage(workflow, name, identities) for name in enabled_parents(workflow, stage)
    }
    parent_ids = {name: checkpoint.artifact_id for name, checkpoint in parents.items()}
    state = next(iter(parents.values())).state
    pending = pending_path(workflow, stage)
    if suggested:
        decision = {"parent_id": identity(parent_ids), **suggested_decision(stage, state)}
    elif decisions is not None:
        decision = read_yaml(decisions)
    else:
        # A filled-in pending file is the headless review; the viewer is for the rest.
        decision = read_yaml(pending) if pending.exists() else {}
        if not decision or None in decision.values():
            try:
                decision = {"parent_id": identity(parent_ids), **viewer_decision(stage, state)}
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(f"{exc}; or fill in {pending}") from exc
    if decision.get("parent_id") != identity(parent_ids):
        raise ValueError("review.parent_id: stale or missing reviewed checkpoint identity")
    fields = {
        "raw": {"bads", "spans"},
        "epochs": {"exclude"},
        "artifact": {"fit_id", "exclude", "include", "apply"},
    }
    mapping(decision, f"review.{target}", fields[target] | {"parent_id"})
    payload = {key: value for key, value in decision.items() if key != "parent_id"}
    if any(value is None for value in payload.values()):
        raise ValueError("review: pending null choices must be replaced by explicit decisions")
    execute_numeric(stage, state, workflow.config.processing, payload)
    filelock = require("filelock", "preprocessing")
    with filelock.FileLock(workflow.workspace / ".writer.lock", timeout=0):
        path = decision_path(workflow, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"{path}: decision already exists; reset --from {stage} to replace"
            )
        write_pointer(path, decision)
        read_decision(workflow, stage, parent_ids)
    return path
