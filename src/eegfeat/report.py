"""Portable cohort quality reports built with MNE's report interface."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne  # type: ignore[import-untyped]
import pandas as pd

from eegfeat.io import _read_sidecar, _table_from_sidecar
from eegfeat.provenance import identity, implementation_hash, serializable, software_versions
from eegfeat.quality import (
    QualityPolicy,
    _validate_descriptor_labels,
    apply_quality,
    cohort_quality,
    feature_quality,
)
from eegfeat.table import FeatureTable


def _epoch_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer count.")
    return value


def _decision_record(provenance: dict[str, Any], name: str) -> str | None:
    decision = provenance.get(name)
    if decision is None:
        return None
    if not isinstance(decision, dict):
        raise ValueError(f"Preprocessing {name} must contain a decision object.")
    return json.dumps(decision, sort_keys=True, allow_nan=False)


def _recording_evidence(source: Path) -> dict[str, Any]:
    sidecar = _read_sidecar(source)
    table = _table_from_sidecar(source, sidecar)
    provenance = sidecar.get("provenance")
    if not isinstance(provenance, dict) or not {"recording", "n_epochs", "upstream"} <= set(
        provenance
    ):
        raise ValueError(f"{source.name} lacks recording and preprocessing provenance.")
    recording = provenance["recording"]
    _validate_descriptor_labels(pd.DataFrame({"recording": [recording]}), ("recording",))
    if not isinstance(recording, str):
        raise ValueError("Recording provenance requires a string recording identity.")
    if table.row_ids is not None and any(row[0] != recording for row in table.row_ids):
        raise ValueError(f"{source.name} recording provenance disagrees with canonical row_ids.")
    record: dict[str, Any] = {
        "recording": recording,
        "input_epochs": _epoch_count(provenance["n_epochs"], "Input epochs"),
        "upstream_original_events": None,
        "upstream_retained": None,
        "upstream_rejected": None,
        "artifact_method": None,
        "proposed_artifact_exclusions": None,
        "artifact_decision": None,
        "epoch_decision": None,
        "raw_decision": None,
    }
    upstream = provenance["upstream"]
    if upstream is None:
        return record
    if not isinstance(upstream, dict) or not isinstance(upstream.get("provenance"), dict):
        raise ValueError("Upstream preprocessing provenance must contain an object.")
    original = upstream["provenance"]
    if identity(original) != upstream.get("identity"):
        raise ValueError("Upstream preprocessing provenance identity mismatch.")
    for source_name, column in (
        ("original_events", "upstream_original_events"),
        ("retained", "upstream_retained"),
    ):
        if source_name in original:
            record[column] = _epoch_count(original[source_name], source_name)
    original_count, retained_count = record["upstream_original_events"], record["upstream_retained"]
    if original_count is not None and retained_count is not None:
        if retained_count > original_count:
            raise ValueError("Upstream retained epochs exceed original event count.")
        record["upstream_rejected"] = original_count - retained_count
    artifact = original.get("artifact")
    if artifact is not None:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("method"), str):
            raise ValueError("Upstream artifact evidence must declare its method.")
        record["artifact_method"] = artifact["method"]
        evidence = artifact.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("Upstream artifact evidence must contain an object.")
        proposed = evidence.get("suggested_exclude")
        if proposed is not None:
            if not isinstance(proposed, list):
                raise ValueError("Suggested artifact exclusions must be a list.")
            record["proposed_artifact_exclusions"] = len(proposed)
    for name in ("artifact_decision", "epoch_decision", "raw_decision"):
        record[name] = _decision_record(original, name)
    return record


def recording_quality(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Restore recording counts and saved preprocessing decisions from checked bundles.

    Missing upstream evidence remains missing. Proposed artifact exclusions are
    the number explicitly saved by preprocessing, not an artifact score or an
    estimate of the number of rejected epochs. Repeated bundles for one recording
    must carry identical evidence and contribute one summary row.
    """
    if not paths:
        raise ValueError("recording_quality requires at least one feature bundle.")
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = _recording_evidence(Path(path))
        recording = record["recording"]
        if recording in records and records[recording] != record:
            raise ValueError(f"Recording {recording!r} has conflicting preprocessing evidence.")
        records[recording] = record
    return pd.DataFrame(records.values())


def write_quality_report(
    table: FeatureTable,
    descriptors: pd.DataFrame,
    path: str | Path,
    *,
    by: tuple[str, ...],
    quality: QualityPolicy | None = None,
    recording_summary: pd.DataFrame | None = None,
) -> Path:
    """Write cohort support, missingness, flags, exclusions and feature definitions.

    Descriptor rows must already be aligned to the table. Coverage counts finite
    numerical input; interpret it alongside preprocessing retention and artifact
    flags, rather than as an artifact-free-data score.
    ``recording_summary`` is the observed evidence from :func:`recording_quality`.
    """
    destination = Path(path)
    if destination.suffix != ".html":
        raise ValueError("Quality reports require an .html destination.")
    report = mne.Report(title="EEGFeat cohort quality", verbose="error")
    report.add_html(
        cohort_quality(table, descriptors, by=by).to_html(index=False), title="Cohort quality"
    )
    report.add_html(feature_quality(table).to_html(index=False), title="Feature quality")
    if recording_summary is not None:
        _validate_descriptor_labels(recording_summary, ("recording",))
        if recording_summary.recording.duplicated().any():
            raise ValueError("Recording quality requires one summary row per recording.")
        report.add_html(
            recording_summary.to_html(index=False), title="Recording and preprocessing quality"
        )
    if quality is not None:
        result = apply_quality(table, quality)
        report.add_html(
            cohort_quality(result.table, descriptors, by=by).to_html(index=False),
            title="After quality exclusions",
        )
        ledger = result.ledger.join(descriptors.reset_index(drop=True), on="row", rsuffix="_label")
        report.add_html(ledger.to_html(index=False), title="Quality exclusions")
    definitions = [serializable(meta.record()) for meta in table.meta]
    report.add_html(
        "<pre>" + html.escape(json.dumps(definitions, indent=2)) + "</pre>",
        title="Feature definitions",
    )
    provenance = {
        "software": software_versions(),
        "code_sha256": implementation_hash(),
        "quality": serializable(quality),
        "grouping": by,
        "coverage_definition": "fraction of numerically finite input",
    }
    report.add_html(
        "<pre>" + html.escape(json.dumps(provenance, indent=2)) + "</pre>",
        title="Report provenance",
    )
    report.save(destination, open_browser=False, overwrite=False, verbose="error")
    return destination
