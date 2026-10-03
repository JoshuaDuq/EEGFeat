"""Cohort assembly for estimates whose samples are trial groups, not epochs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegtable.io import _read_descriptor_frame, _read_sidecar, _table_from_sidecar
from eegtable.quality import QualityPolicy, _validate_descriptor_labels, apply_quality
from eegtable.table import FeatureMeta, FeatureTable, _stack_union


@dataclass(frozen=True)
class GroupDataset:
    table: FeatureTable
    targets: pd.DataFrame


@dataclass(frozen=True)
class GroupDesign:
    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    groups: npt.NDArray[np.object_]
    sample_ids: tuple[tuple[str, str], ...]
    column_names: tuple[str, ...]
    feature_columns: npt.NDArray[np.intp]
    covariate_columns: npt.NDArray[np.intp]
    coverage: npt.NDArray[np.float64]
    flags: dict[str, npt.NDArray[np.bool_]]
    meta: tuple[FeatureMeta, ...]
    quality_ledger: pd.DataFrame

    @property
    def n_covariates(self) -> int:
        return 0


def read_group_dataset(
    paths: Sequence[str | Path], *, columns: Literal["identical", "union"] = "union"
) -> GroupDataset:
    """Read group samples with explicit recording identities and trial counts."""
    if not paths or columns not in ("identical", "union"):
        raise ValueError("Provide table paths and columns='identical' or 'union'.")
    tables: list[FeatureTable] = []
    descriptors: list[pd.DataFrame] = []
    identities: list[tuple[str, str]] = []
    for path in paths:
        source = Path(path)
        sidecar = _read_sidecar(source)
        table = _table_from_sidecar(source, sidecar)
        if table.row_labels is None or table.row_ids is not None:
            raise ValueError("read_group_dataset requires group-row tables.")
        frame = _read_descriptor_frame(source, sidecar)
        if not {"recording", "group", "n_trials"}.issubset(frame.columns):
            raise ValueError("Group descriptors require recording, group and n_trials.")
        if frame["group"].tolist() != list(table.row_labels):
            raise ValueError("Serialized group labels disagree with the feature sidecar.")
        _validate_descriptor_labels(frame, ("recording", "group"))
        counts = pd.to_numeric(frame["n_trials"], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(counts) & (counts > 0) & (counts == np.floor(counts))):
            raise ValueError("n_trials must contain positive integer trial counts.")
        frame["n_trials"] = counts.astype(np.int64)
        identities.extend(zip(frame.recording, frame.group, strict=True))
        tables.append(table)
        descriptors.append(frame)
    if len(set(identities)) != len(identities):
        raise ValueError("Found duplicate (recording, group) samples.")
    if columns == "identical" and any(table.meta != tables[0].meta for table in tables):
        raise ValueError("Group tables require identical ordered feature metadata.")
    labels = tuple(json.dumps(pair, separators=(",", ":")) for pair in identities)
    return GroupDataset(
        _stack_union(tables, row_labels=labels), pd.concat(descriptors, ignore_index=True)
    )


def build_group_design(
    dataset: GroupDataset,
    *,
    target: str,
    groups: str = "subject_id",
    selection: object = None,
    quality: QualityPolicy | None = None,
) -> GroupDesign:
    """Build a grouped-CV design without broadcasting estimates into epoch rows."""
    from eegtable.model.design import Selection, _validate_precomputed_features, select

    table, descriptors = dataset.table, dataset.targets
    if table.row_labels is None or len(descriptors) != table.n_rows:
        raise ValueError("GroupDataset must contain aligned group samples.")
    _validate_descriptor_labels(descriptors, ("recording", "group", groups))
    sample_ids = tuple(zip(descriptors.recording, descriptors.group, strict=True))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("GroupDataset contains duplicate canonical samples.")
    labels = tuple(json.dumps(pair, separators=(",", ":")) for pair in sample_ids)
    if labels != table.row_labels:
        raise ValueError("Group descriptors disagree with canonical sample identities.")
    if selection is not None:
        if not isinstance(selection, Selection):
            raise TypeError("selection must be a model.Selection.")
        table = select(table, selection)
    _validate_precomputed_features(table)
    for name in (target, groups):
        if name not in descriptors:
            raise ValueError(f"Group descriptors missing {name!r}.")
        if descriptors[name].isna().any():
            raise ValueError(f"Group descriptors {name!r} contain missing values.")
    y = pd.to_numeric(descriptors[target], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Group targets must be finite.")
    ledger = pd.DataFrame(columns=["row", "feature", "reason", "coverage"])
    if quality is not None:
        result = apply_quality(table, quality)
        table, ledger = result.table, result.ledger
    return GroupDesign(
        table.values,
        y,
        descriptors[groups].to_numpy(dtype=object),
        sample_ids,
        tuple(table.names),
        np.arange(len(table.meta), dtype=np.intp),
        np.empty(0, dtype=np.intp),
        table.coverage,
        dict(table.flags),
        table.meta,
        ledger,
    )
