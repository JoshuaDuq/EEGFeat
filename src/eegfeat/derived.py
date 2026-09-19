from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable, Normalization

_LOG_UNITS: dict[Normalization, str] = {
    "log10": "log10 ratio",
    "log_ratio": "log10 ratio",
    "db": "dB",
}
"""Unit of a difference taken on each logarithmic scale.

Subtracting two values already in dB leaves a dB difference, not a bare log
ratio: the factor of ten is still in there.
"""

_LOGARITHMIC = tuple(_LOG_UNITS)


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

    operands = [(top[k], bottom[k]) for k in sorted(top)]
    measure = f"ratio_{numerator}_{denominator}"
    meta = tuple(
        replace(
            table.meta[i],
            measure=measure,
            band=None,
            unit=_LOG_UNITS.get(table.meta[i].normalization, "ratio"),
            computation=_derived_spec(
                measure, table, (i, "numerator"), (j, "denominator"), ratio=True
            ),
        )
        for i, j in operands
    )
    return FeatureTable(
        values=np.stack([_combine(table, i, j, ratio=True) for i, j in operands], axis=1),
        coverage=np.stack(
            [np.minimum(table.coverage[:, i], table.coverage[:, j]) for i, j in operands], axis=1
        ),
        meta=meta,
        flags=_merge_flags(table, operands),
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
    operands: list[tuple[int, int]] = []
    meta: list[FeatureMeta] = []
    for left, right in pairs:
        left_index = _index_by_space(table, left)
        right_index = _index_by_space(table, right)
        for key, on_left in sorted(left_index.items()):
            if key not in right_index:
                continue
            on_right = right_index[key]
            operands.append((on_right, on_left))
            meta.append(
                replace(
                    table.meta[on_left],
                    measure="asymmetry",
                    space=f"{left}-{right}",
                    space_kind="pair",
                    unit=_LOG_UNITS.get(table.meta[on_left].normalization, "a.u."),
                    computation=_derived_spec(
                        "asymmetry", table, (on_right, "right"), (on_left, "left"), ratio=False
                    ),
                )
            )
    if not operands:
        raise ValueError("no pair matched a band and window present in the table.")
    return FeatureTable(
        values=np.stack([_combine(table, i, j, ratio=False) for i, j in operands], axis=1),
        coverage=np.stack(
            [np.minimum(table.coverage[:, i], table.coverage[:, j]) for i, j in operands], axis=1
        ),
        meta=tuple(meta),
        flags=_merge_flags(table, operands),
        row_labels=table.row_labels,
        row_ids=table.row_ids,
    )


def _derived_spec(
    measure: str,
    table: FeatureTable,
    first: tuple[int, str],
    second: tuple[int, str],
    *,
    ratio: bool,
) -> ComputationSpec:
    """Describe the derivation itself, not the measurement it started from.

    Inheriting the input's spec would name the numerator's algorithm alone, so a
    theta/beta ratio and a theta/alpha ratio would hash identically. Each operand
    is recorded whole: its band bounds, scale, window and own computation.
    """
    first_index, first_role = first
    second_index, second_role = second
    if table.meta[first_index].normalization in _LOGARITHMIC:
        operation = "difference"
    else:
        operation = "quotient" if ratio else "normalized_difference"
    return ComputationSpec.create(
        measure,
        operation=operation,
        **{
            first_role: table.meta[first_index].record(),
            second_role: table.meta[second_index].record(),
        },
    )


def _merge_flags(
    table: FeatureTable, operands: Sequence[tuple[int, int]]
) -> Mapping[str, npt.NDArray[np.bool_]]:
    """Carry a flag on either operand onto the derived column.

    A ratio built from a flagged input is itself suspect, and dropping the flag
    would hide that.
    """
    return {
        key: np.stack([array[:, i] | array[:, j] for i, j in operands], axis=1)
        for key, array in table.flags.items()
    }


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
