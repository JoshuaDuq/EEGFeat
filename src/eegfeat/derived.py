from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from eegfeat.table import FeatureMeta, FeatureTable

_LOGARITHMIC = ("log10", "log_ratio", "db")


def band_ratio(table: FeatureTable, numerator: str, denominator: str) -> FeatureTable:
    """Ratio between two bands, computed per spatial unit and window.

    When the input is already logarithmic the ratio is a difference, and this
    subtracts rather than divides.

    Parameters
    ----------
    table : FeatureTable
        Band power, with both bands present for every spatial unit and window.
    numerator, denominator : str
        Band names.

    Returns
    -------
    FeatureTable
        One column per spatial unit and window, measure
        ``"ratio_{numerator}_{denominator}"``.
    """
    top = _index_by_position(table, numerator)
    bottom = _index_by_position(table, denominator)
    missing = sorted(set(top) ^ set(bottom))
    if missing:
        raise ValueError(
            f"bands {numerator!r} and {denominator!r} do not cover the same spatial units "
            f"and windows; unmatched: {missing}"
        )

    keys = sorted(top)
    values = np.stack([_combine(table, top[k], bottom[k], ratio=True) for k in keys], axis=1)
    coverage = np.stack(
        [np.minimum(table.coverage[:, top[k]], table.coverage[:, bottom[k]]) for k in keys],
        axis=1,
    )
    meta = tuple(
        replace(
            table.meta[top[k]],
            measure=f"ratio_{numerator}_{denominator}",
            band=None,
            unit="ratio" if table.meta[top[k]].normalization not in _LOGARITHMIC else "log ratio",
        )
        for k in keys
    )
    return FeatureTable(
        values=values,
        coverage=coverage,
        meta=meta,
        row_labels=table.row_labels,
        row_ids=table.row_ids,
    )


def asymmetry(table: FeatureTable, pairs: Sequence[tuple[str, str]]) -> FeatureTable:
    """Hemispheric asymmetry between channel pairs.

    For raw power this is ``(right - left) / (right + left)``. For logarithmic
    input the normalized difference is already a plain difference, so it is
    ``right - left``.

    Parameters
    ----------
    table : FeatureTable
        Band power containing both members of every pair.
    pairs : sequence of (str, str)
        ``(left, right)`` channel names.

    Returns
    -------
    FeatureTable
        One column per pair, band and window, with ``space_kind="pair"``.
    """
    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for left, right in pairs:
        left_index = _index_by_space(table, left)
        right_index = _index_by_space(table, right)
        for key, i in sorted(left_index.items()):
            if key not in right_index:
                continue
            j = right_index[key]
            columns.append(
                (
                    replace(
                        table.meta[i],
                        measure="asymmetry",
                        space=f"{left}-{right}",
                        space_kind="pair",
                        unit="a.u.",
                    ),
                    _combine(table, j, i, ratio=False),
                    np.minimum(table.coverage[:, i], table.coverage[:, j]),
                )
            )
    if not columns:
        raise ValueError("no pair matched a band and window present in the table.")
    return FeatureTable(
        values=np.stack([v for _, v, _ in columns], axis=1),
        coverage=np.stack([c for _, _, c in columns], axis=1),
        meta=tuple(m for m, _, _ in columns),
        row_labels=table.row_labels,
        row_ids=table.row_ids,
    )


def _combine(table: FeatureTable, top: int, bottom: int, *, ratio: bool) -> npt.NDArray[np.float64]:
    a, b = table.values[:, top], table.values[:, bottom]
    if table.meta[top].normalization in _LOGARITHMIC:
        return a - b
    if ratio:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(b != 0.0, a / b, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = a + b
        return np.where(total != 0.0, (a - b) / total, np.nan)


def _index_by_position(table: FeatureTable, band_name: str) -> dict[tuple[str, str | None], int]:
    found = {
        (m.space, m.window): i
        for i, m in enumerate(table.meta)
        if m.band is not None and m.band.name == band_name
    }
    if not found:
        raise ValueError(f"band {band_name!r} is not present in the table.")
    return found


def _index_by_space(table: FeatureTable, channel: str) -> dict[tuple[str, str | None], int]:
    found = {
        (m.band.name if m.band else "", m.window): i
        for i, m in enumerate(table.meta)
        if m.space == channel
    }
    if not found:
        raise KeyError(f"channel {channel!r} is not present in the table.")
    return found
