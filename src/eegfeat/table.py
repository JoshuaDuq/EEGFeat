from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegfeat.bands import Band
from eegfeat.naming import feature_name

SpaceKind = Literal["channel", "roi", "global", "pair", "state"]
Normalization = Literal["raw", "log10", "log_ratio", "db", "percent"]
RowId = tuple[str, int, str]
"""Immutable ``(recording, epoch, event)`` identity for one epoch row."""


@dataclass(frozen=True)
class ComputationSpec:
    """Canonical, immutable description of an algorithm and its parameters."""

    method: str
    parameters_json: str

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("computation method must be non-empty.")
        parsed = json.loads(self.parameters_json)
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical != self.parameters_json:
            raise ValueError("parameters_json must use canonical JSON serialization.")

    @classmethod
    def create(cls, method: str, **parameters: object) -> ComputationSpec:
        canonical = json.dumps(
            _json_value(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(method=method, parameters_json=canonical)

    @property
    def parameters(self) -> Mapping[str, object]:
        """Decoded parameters."""
        parsed = json.loads(self.parameters_json)
        if not isinstance(parsed, dict):
            raise TypeError("canonical computation parameters must be a JSON object.")
        return parsed

    @property
    def parameter_hash(self) -> str:
        """Stable SHA-256 digest of the complete computation specification."""
        payload = f"{self.method}\n{self.parameters_json}".encode()
        return hashlib.sha256(payload).hexdigest()

    def record(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {"method": self.method, "parameters": dict(self.parameters)}


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"computation parameter {value!r} is not JSON-serializable.")


@dataclass(frozen=True)
class FeatureMeta:
    """Structured description of one feature column.

    Every field is constant across epochs. Per-epoch facts belong in
    :attr:`FeatureTable.flags`, not here.

    Parameters
    ----------
    measure : str
        Measure label, e.g. ``"power"``, ``"peak_freq"``, ``"slope"``.
    band : Band or None
        The band this column was computed over, or None for measures spanning
        a fitted range rather than a band.
    space : str
        Channel name, ROI name, or ``"global"``.
    space_kind : {"channel", "roi", "global", "pair", "state"}
        Which of those ``space`` is. ``"pair"`` marks a derived unit relating two
        nodes, such as an asymmetry or a connection; ``"state"`` marks a
        microstate class, which is a spatial mode rather than a location.
    window : str or None
        Time window name, or None when the spectrum spans the whole segment.
    normalization : {"raw", "log10", "log_ratio", "db"}
        Normalization applied to the value.
    unit : str
        Physical unit, or a description of the normalized scale.
    source : str
        Provenance of the spectra, e.g. ``"morlet"`` or ``"multitaper"``.
    freq_resolution_hz : float or None
        Median spacing of the frequency bins inside ``band``. Reported so a
        caller can judge whether the grid supported the measure; it never
        gates anything.
    """

    measure: str
    band: Band | None
    space: str
    space_kind: SpaceKind
    window: str | None
    normalization: Normalization
    unit: str
    source: str
    window_bounds: tuple[float, float] | None
    computation: ComputationSpec
    freq_resolution_hz: float | None = None
    phase_band: Band | None = None
    amplitude_band: Band | None = None
    nodes: tuple[str, str] | None = None

    @property
    def parameter_hash(self) -> str:
        """Stable digest of every field that defines this feature column."""
        band = None
        if self.band is not None:
            band = {"name": self.band.name, "fmin": self.band.fmin, "fmax": self.band.fmax}
        phase_band = None
        if self.phase_band is not None:
            phase_band = {
                "name": self.phase_band.name,
                "fmin": self.phase_band.fmin,
                "fmax": self.phase_band.fmax,
            }
        amplitude_band = None
        if self.amplitude_band is not None:
            amplitude_band = {
                "name": self.amplitude_band.name,
                "fmin": self.amplitude_band.fmin,
                "fmax": self.amplitude_band.fmax,
            }
        record = {
            "measure": self.measure,
            "band": band,
            "space": self.space,
            "space_kind": self.space_kind,
            "window": self.window,
            "window_bounds": self.window_bounds,
            "normalization": self.normalization,
            "unit": self.unit,
            "source": self.source,
            "freq_resolution_hz": self.freq_resolution_hz,
            "computation": self.computation.record(),
            "phase_band": phase_band,
            "amplitude_band": amplitude_band,
            "nodes": self.nodes,
        }
        canonical = json.dumps(
            _json_value(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def name(self) -> str:
        """The canonical feature name for this column."""
        readable = feature_name(
            measure=self.measure,
            band=(
                f"phase-{self.phase_band.name}--amp-{self.amplitude_band.name}"
                if self.phase_band is not None and self.amplitude_band is not None
                else self.band.name if self.band is not None else None
            ),
            space=self.space,
            window=self.window,
            normalization=self.normalization,
        )
        return f"{readable}_p{self.parameter_hash[:12]}"


@dataclass(frozen=True, eq=False)
class FeatureTable:
    """Feature values with one metadata record per column.

    Parameters
    ----------
    values : ndarray, shape (n_epochs, n_features)
        Feature values. NaN marks a value withheld because of a data
        condition; see ``coverage`` for how much valid input it had.
    coverage : ndarray, shape (n_epochs, n_features)
        Fraction of numerically finite input that produced each value, in
        ``[0, 1]``. This is not an artifact-free-data score. For Morlet input,
        :attr:`Spectra.support` separately records the fraction of the requested
        window with complete wavelet support.
    meta : tuple of FeatureMeta
        One record per column.
    flags : mapping of str to ndarray, optional
        Per-cell boolean annotations, each shaped like ``values``.
    row_labels : tuple of str, optional
        Names for the rows when they are **not** epochs. None, the default, means
        one row per epoch. A measure estimated across trials, such as inter-trial
        phase coherence, has one row per trial group and names them here, so a
        table of group rows cannot be silently joined to a table of epoch rows.
    row_ids : tuple of (str, int, str), optional
        Recording, original epoch index and event identity for every epoch row.
        Required when concatenating per-epoch tables. Group-row tables use
        ``row_labels`` instead, and cannot also carry ``row_ids``.
    """

    values: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]
    meta: tuple[FeatureMeta, ...]
    flags: Mapping[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    row_labels: tuple[str, ...] | None = None
    row_ids: tuple[RowId, ...] | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D (n_epochs, n_features), got {self.values.shape}.")
        if self.coverage.shape != self.values.shape:
            raise ValueError(
                f"coverage shape {self.coverage.shape} does not match values {self.values.shape}."
            )
        if len(self.meta) != self.values.shape[1]:
            raise ValueError(
                f"meta has {len(self.meta)} records but values has {self.values.shape[1]} columns."
            )
        for key, array in self.flags.items():
            if array.shape != self.values.shape:
                raise ValueError(
                    f"flag {key!r} shape {array.shape} does not match values {self.values.shape}."
                )
        if self.row_labels is not None and len(self.row_labels) != self.values.shape[0]:
            raise ValueError(
                f"row_labels has {len(self.row_labels)} entries but values has "
                f"{self.values.shape[0]} rows."
            )
        if self.row_ids is not None and len(self.row_ids) != self.values.shape[0]:
            raise ValueError(
                f"row_ids has {len(self.row_ids)} entries but values has "
                f"{self.values.shape[0]} rows."
            )
        if self.row_labels is not None and self.row_ids is not None:
            raise ValueError("group row_labels and epoch row_ids are mutually exclusive.")
        if self.row_ids is not None:
            for row_id in self.row_ids:
                if (
                    len(row_id) != 3
                    or not isinstance(row_id[0], str)
                    or not isinstance(row_id[1], int)
                    or not isinstance(row_id[2], str)
                ):
                    raise ValueError(
                        "each row_id must be (recording: str, epoch: int, event: str)."
                    )
        names = self.names
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate feature names: {duplicates}")

    @property
    def names(self) -> list[str]:
        """Canonical name of every column, in order."""
        return [m.name for m in self.meta]

    @property
    def n_rows(self) -> int:
        """Number of rows: epochs, or trial groups when ``row_labels`` is set."""
        return int(self.values.shape[0])

    def to_dataframe(self) -> pd.DataFrame:
        """Render the values as a DataFrame with canonical column names.

        The index is the row labels when the rows are trial groups, and a plain
        range when they are epochs.
        """
        return pd.DataFrame(self.values, columns=self.names, index=self.row_labels)

    def select(self, **conditions: object) -> FeatureTable:
        """Return the columns whose metadata matches every given field.

        Parameters
        ----------
        **conditions
            Field name to required value, e.g. ``select(space="C4")``.

        Returns
        -------
        FeatureTable
            A new table holding only the matching columns.
        """
        known = {f.name for f in fields(FeatureMeta)}
        unknown = set(conditions) - known
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} is not a FeatureMeta field; known fields: {sorted(known)}"
            )
        keep = [
            i
            for i, m in enumerate(self.meta)
            if all(getattr(m, key) == value for key, value in conditions.items())
        ]
        index = np.asarray(keep, dtype=int)
        return FeatureTable(
            values=self.values[:, index],
            coverage=self.coverage[:, index],
            meta=tuple(self.meta[i] for i in keep),
            flags={k: v[:, index] for k, v in self.flags.items()},
            row_labels=self.row_labels,
            row_ids=self.row_ids,
        )


def concat(tables: Sequence[FeatureTable]) -> FeatureTable:
    """Join feature tables column-wise.

    Every table must have the same number of rows and the same row semantics.
    Rows are never aligned or reindexed; a mismatch is an error rather than
    something to repair.

    Parameters
    ----------
    tables : sequence of FeatureTable
        Tables to join, in order.

    Returns
    -------
    FeatureTable
        One table holding every column. A flag present in some inputs and
        absent in others is False where it was absent.
    """
    if not tables:
        raise ValueError("concat requires at least one table.")
    n_rows = tables[0].n_rows
    row_labels = tables[0].row_labels
    row_ids = tables[0].row_ids
    for table in tables:
        # Row semantics first: it explains a count mismatch too, and is the more
        # useful error when a cross-trial estimate meets a per-epoch one.
        if table.row_labels != row_labels:
            raise ValueError(
                "concat requires matching row semantics: one table has rows "
                f"{row_labels!r} and another {table.row_labels!r}. A measure estimated "
                "across trials cannot be joined to one estimated per epoch without "
                "deciding how to broadcast it."
            )
        if row_labels is None and table.row_ids != row_ids:
            raise ValueError(
                "concat requires exact row identities in the same order; the recording, "
                "epoch or event identity differs."
            )
        if table.n_rows != n_rows:
            raise ValueError(f"concat requires matching n_rows; got {n_rows} and {table.n_rows}.")
    if row_labels is None and row_ids is None:
        raise ValueError(
            "concat requires row_ids for per-epoch tables; row counts alone cannot prove "
            "that recordings, epochs and events are aligned."
        )
    flag_keys = sorted({key for table in tables for key in table.flags})
    flags = {
        key: np.concatenate(
            [table.flags.get(key, np.zeros(table.values.shape, dtype=bool)) for table in tables],
            axis=1,
        )
        for key in flag_keys
    }
    return FeatureTable(
        values=np.concatenate([t.values for t in tables], axis=1),
        coverage=np.concatenate([t.coverage for t in tables], axis=1),
        meta=tuple(m for t in tables for m in t.meta),
        flags=flags,
        row_labels=row_labels,
        row_ids=row_ids,
    )
