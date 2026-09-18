"""Read and write a :class:`~eegfeat.FeatureTable` without losing its metadata.

A table is stored as three files beside one another:

- ``<name>.tsv``, the values, one row per epoch or trial group, NaN as ``n/a``;
- ``<name>_coverage.tsv``, the coverage matrix in the same layout;
- ``<name>.json``, a sidecar holding the :class:`~eegfeat.FeatureMeta` of every
  column, the row semantics and any per-cell flags.

The TSV files open in anything that reads tabular data. The sidecar is what lets
:func:`read_table` rebuild the table exactly, so columns can still be selected by
band, space or window rather than by parsing their names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegfeat.bands import Band
from eegfeat.table import FeatureMeta, FeatureTable

_NA = "n/a"
"""Missing-value marker, following the BIDS convention for tabular files."""

_EPOCH_KEY = "epoch"
_GROUP_KEY = "group"


def write_table(
    table: FeatureTable,
    path: str | os.PathLike[str],
    *,
    rows: pd.DataFrame | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write a feature table as TSV files and a JSON sidecar.

    Parameters
    ----------
    table : FeatureTable
        The table to write.
    path : path-like
        Destination of the values file. Must end in ``.tsv``; the coverage file
        and sidecar are named after it.
    rows : DataFrame, optional
        Descriptive columns written between the row key and the features, one
        row per table row, e.g. the event and metadata of each epoch. They are
        for joining the values to other data and are not read back.
    provenance : mapping, optional
        JSON-serializable record of how the table was produced, stored in the
        sidecar as given.

    Returns
    -------
    tuple of Path
        The values file, the coverage file and the sidecar, in that order.
    """
    target = Path(path)
    if target.suffix != ".tsv":
        raise ValueError(f"write_table writes a .tsv file; got {target.name!r}.")

    key, key_values = _row_key(table)
    descriptors = _descriptors(rows, table, key)
    names = table.names

    values_frame = pd.concat(
        [
            pd.DataFrame({key: key_values}),
            descriptors,
            pd.DataFrame(table.values, columns=names),
        ],
        axis=1,
    )
    coverage_frame = pd.concat(
        [pd.DataFrame({key: key_values}), pd.DataFrame(table.coverage, columns=names)], axis=1
    )

    coverage_path = target.with_name(f"{target.stem}_coverage.tsv")
    sidecar_path = target.with_suffix(".json")
    sidecar = _sidecar(table, key, list(descriptors.columns), coverage_path.name, provenance)

    _write_tsv(values_frame, target)
    _write_tsv(coverage_frame, coverage_path)
    _write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n", sidecar_path)
    return target, coverage_path, sidecar_path


def read_table(path: str | os.PathLike[str]) -> FeatureTable:
    """Read a feature table written by :func:`write_table`.

    Parameters
    ----------
    path : path-like
        The values ``.tsv`` file. Its sidecar and coverage file are found beside it.

    Returns
    -------
    FeatureTable
        Values, coverage, metadata, flags and row labels as they were written.
    """
    source = Path(path)
    sidecar = json.loads(source.with_suffix(".json").read_text())
    meta = tuple(_meta_from_record(record) for record in sidecar["columns"])
    names = [m.name for m in meta]

    values = _read_matrix(source, names)
    coverage = _read_matrix(source.with_name(sidecar["coverage"]), names)
    column_index = {name: i for i, name in enumerate(names)}
    flags: dict[str, npt.NDArray[np.bool_]] = {}
    for key, cells in sidecar["flags"].items():
        flag = np.zeros(values.shape, dtype=bool)
        for name, row_indices in cells.items():
            flag[np.asarray(row_indices, dtype=int), column_index[name]] = True
        flags[key] = flag

    labels = sidecar["row_labels"]
    return FeatureTable(
        values=values,
        coverage=coverage,
        meta=meta,
        flags=flags,
        row_labels=None if labels is None else tuple(str(label) for label in labels),
    )


def _row_key(table: FeatureTable) -> tuple[str, list[int] | list[str]]:
    if table.row_labels is None:
        return _EPOCH_KEY, list(range(table.n_rows))
    return _GROUP_KEY, list(table.row_labels)


def _descriptors(rows: pd.DataFrame | None, table: FeatureTable, key: str) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame(index=range(table.n_rows))
    if len(rows) != table.n_rows:
        raise ValueError(f"rows has {len(rows)} entries but the table has {table.n_rows} rows.")
    reserved = {key, *table.names}
    clashes = sorted(str(column) for column in rows.columns if column in reserved)
    if clashes:
        raise ValueError(
            f"descriptor columns {clashes} would shadow the row key {key!r} or a feature."
        )
    return rows.reset_index(drop=True)


def _sidecar(
    table: FeatureTable,
    key: str,
    descriptor_columns: list[str],
    coverage_name: str,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from eegfeat import __version__

    names = table.names
    flags = {
        flag: {
            names[column]: np.flatnonzero(cells[:, column]).tolist()
            for column in range(cells.shape[1])
            if cells[:, column].any()
        }
        for flag, cells in table.flags.items()
    }
    sidecar: dict[str, Any] = {
        "eegfeat_version": __version__,
        "rows": "epochs" if table.row_labels is None else "groups",
        "row_labels": None if table.row_labels is None else list(table.row_labels),
        "row_columns": [key, *descriptor_columns],
        "coverage": coverage_name,
        "columns": [_meta_record(m) for m in table.meta],
        "flags": flags,
    }
    if provenance is not None:
        sidecar["provenance"] = dict(provenance)
    return sidecar


def _meta_record(meta: FeatureMeta) -> dict[str, Any]:
    band = (
        None
        if meta.band is None
        else {
            "name": meta.band.name,
            "fmin": meta.band.fmin,
            "fmax": meta.band.fmax,
        }
    )
    return {
        "name": meta.name,
        "measure": meta.measure,
        "band": band,
        "space": meta.space,
        "space_kind": meta.space_kind,
        "window": meta.window,
        "normalization": meta.normalization,
        "unit": meta.unit,
        "source": meta.source,
        "freq_resolution_hz": meta.freq_resolution_hz,
    }


def _meta_from_record(record: Mapping[str, Any]) -> FeatureMeta:
    band = record["band"]
    meta = FeatureMeta(
        measure=record["measure"],
        band=None if band is None else Band(band["name"], band["fmin"], band["fmax"]),
        space=record["space"],
        space_kind=record["space_kind"],
        window=record["window"],
        normalization=record["normalization"],
        unit=record["unit"],
        source=record["source"],
        freq_resolution_hz=record["freq_resolution_hz"],
    )
    if meta.name != record["name"]:
        raise ValueError(
            f"sidecar column {record['name']!r} does not match its own metadata, "
            f"which names it {meta.name!r}."
        )
    return meta


def _read_matrix(path: Path, names: list[str]) -> npt.NDArray[np.float64]:
    frame = pd.read_csv(path, sep="\t", na_values=[_NA], keep_default_na=False)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks columns its sidecar describes: {missing}")
    return np.asarray(frame[names].to_numpy(dtype=float), dtype=np.float64)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, sep="\t", index=False, na_rep=_NA)
    os.replace(partial, path)


def _write_text(text: str, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)
