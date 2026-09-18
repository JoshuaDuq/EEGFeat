from __future__ import annotations

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
    freq_resolution_hz: float | None = None

    @property
    def name(self) -> str:
        """The canonical feature name for this column."""
        return feature_name(
            measure=self.measure,
            band=self.band.name if self.band is not None else None,
            space=self.space,
            window=self.window,
            normalization=self.normalization,
        )


@dataclass(frozen=True, eq=False)
class FeatureTable:
    """Feature values with one metadata record per column.

    Parameters
    ----------
    values : ndarray, shape (n_epochs, n_features)
        Feature values. NaN marks a value withheld because of a data
        condition; see ``coverage`` for how much valid input it had.
    coverage : ndarray, shape (n_epochs, n_features)
        Fraction of valid input that produced each value, in ``[0, 1]``.
    meta : tuple of FeatureMeta
        One record per column.
    flags : mapping of str to ndarray, optional
        Per-cell boolean annotations, each shaped like ``values``.
    row_labels : tuple of str, optional
        Names for the rows when they are **not** epochs. None, the default, means
        one row per epoch. A measure estimated across trials, such as inter-trial
        phase coherence, has one row per trial group and names them here, so a
        table of group rows cannot be silently joined to a table of epoch rows.
    """

    values: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]
    meta: tuple[FeatureMeta, ...]
    flags: Mapping[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    row_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D (n_epochs, n_features), got {self.values.shape}.")
        if self.coverage.shape != self.values.shape:
            raise ValueError(
                f"coverage shape {self.coverage.shape} does not match values {self.values.shape}."
            )
        if len(self.meta) != self.values.shape[1]:
            raise ValueError(
                f"meta has {len(self.meta)} records but values has "
                f"{self.values.shape[1]} columns."
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
        if table.n_rows != n_rows:
            raise ValueError(f"concat requires matching n_rows; got {n_rows} and {table.n_rows}.")
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
    )
