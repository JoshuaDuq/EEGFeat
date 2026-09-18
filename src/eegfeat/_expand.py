from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from eegfeat.bands import Band
from eegfeat.baseline import normalize
from eegfeat.groups import SpatialUnit, aggregate
from eegfeat.qc import band_coverage
from eegfeat.signal import TimeSeries
from eegfeat.spectra import Spectra, Window, gradient_weights, trapezoid_weights
from eegfeat.table import FeatureMeta, FeatureTable, Normalization

Kernel = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.bool_]]],
]


@dataclass(frozen=True, eq=False)
class _Column:
    meta: FeatureMeta
    values: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]


def _collect(
    units: Sequence[SpatialUnit],
    windows: Sequence[Window],
    flags: Mapping[str, npt.NDArray[np.bool_]],
    make_meta: Callable[[SpatialUnit, Window], FeatureMeta],
    *,
    skip_window: int | None,
) -> tuple[list[_Column], dict[str, list[npt.NDArray[np.bool_]]]]:
    columns: list[_Column] = []
    flag_columns: dict[str, list[npt.NDArray[np.bool_]]] = {}
    for spatial in units:
        for w_index, window in enumerate(windows):
            if w_index == skip_window:
                continue
            columns.append(
                _Column(
                    make_meta(spatial, window),
                    spatial.values[:, w_index],
                    spatial.coverage[:, w_index],
                )
            )
            for key, array in flags.items():
                flag_col: npt.NDArray[np.bool_] = np.asarray(
                    array[:, list(spatial.picks), w_index].any(axis=1), dtype=np.bool_
                )
                flag_columns.setdefault(key, []).append(flag_col)
    return columns, flag_columns


def _assemble(
    columns: Sequence[_Column],
    flag_columns: Mapping[str, Sequence[npt.NDArray[np.bool_]]],
    row_labels: tuple[str, ...] | None = None,
) -> FeatureTable:
    return FeatureTable(
        values=np.stack([c.values for c in columns], axis=1),
        coverage=np.stack([c.coverage for c in columns], axis=1),
        meta=tuple(c.meta for c in columns),
        flags={key: np.stack(list(arrays), axis=1) for key, arrays in flag_columns.items()},
        row_labels=row_labels,
    )


def expand(
    spectra: Spectra,
    kernel: Kernel,
    *,
    measure: str,
    unit: str,
    bands: Sequence[Band] | None,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    baseline: str | None,
    mode: Normalization,
    min_bins: int,
    weighting: Literal["trapezoid", "gradient"] = "trapezoid",
) -> FeatureTable:
    baseline_index = _baseline_index(spectra, baseline)
    columns: list[_Column] = []
    flag_columns: dict[str, list[npt.NDArray[np.bool_]]] = {}

    for band in bands if bands is not None else (None,):
        mask = band.mask(spectra.freqs) if band is not None else np.ones(spectra.freqs.size, bool)
        _check_band(band, mask, min_bins, spectra.freqs)

        sub_freqs = spectra.freqs[mask]
        weights = (
            trapezoid_weights(sub_freqs)
            if weighting == "trapezoid"
            else gradient_weights(sub_freqs)
        )
        values, flags = kernel(spectra.data[:, :, :, mask], sub_freqs, weights)
        coverage = band_coverage(spectra.coverage[:, :, :, mask], weights)

        base = values[:, :, baseline_index] if baseline_index is not None else None
        values = normalize(values, baseline=base, mode=mode)

        units = aggregate(values, coverage, spectra.ch_names, groups, include_global)
        resolution = float(np.median(np.diff(sub_freqs))) if sub_freqs.size > 1 else None

        def make_meta(
            spatial: SpatialUnit,
            window: Window,
            _b: Band | None = band,
            _r: float | None = resolution,
        ) -> FeatureMeta:
            return FeatureMeta(
                measure=measure,
                band=_b,
                space=spatial.space,
                space_kind=spatial.space_kind,
                window=window.name,
                normalization=mode,
                unit=unit,
                source=spectra.source,
                freq_resolution_hz=_r,
            )

        band_columns, band_flags = _collect(
            units, spectra.windows, flags, make_meta, skip_window=baseline_index
        )
        columns.extend(band_columns)
        for key, arrays in band_flags.items():
            flag_columns.setdefault(key, []).extend(arrays)

    return _assemble(columns, flag_columns)


def _baseline_index(spectra: Spectra, baseline: str | None) -> int | None:
    if baseline is None:
        return None
    names = [w.name for w in spectra.windows]
    if baseline not in names:
        raise ValueError(f"baseline window {baseline!r} is not among {names}.")
    if len(names) < 2:
        raise ValueError("baseline normalization needs at least one non-baseline window.")
    return names.index(baseline)


def _check_band(
    band: Band | None,
    mask: npt.NDArray[np.bool_],
    min_bins: int,
    freqs: npt.NDArray[np.float64],
) -> None:
    label = band.name if band is not None else "the fitted range"
    n_bins = int(mask.sum())
    if n_bins == 0:
        raise ValueError(
            f"band {label!r} contains no frequencies of the axis spanning "
            f"({freqs[0]}, {freqs[-1]})."
        )
    if n_bins < min_bins:
        raise ValueError(
            f"band {label!r} holds {n_bins} frequency bins but this measure needs at least "
            f"{min_bins} bins. Compute the input on a finer frequency grid."
        )


Series = TypeVar("Series", bound=TimeSeries)

SignalKernel: TypeAlias = Callable[
    [Series, npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
]
"""Measure one window of a series.

Generic in the series type, so a kernel written against :class:`BandSignal` stays
typed as such while :func:`expand_signal` also accepts a plain :class:`Signal`.
"""


def expand_signal(
    signals: Sequence[Series],
    *,
    trace_of: Callable[[Series], npt.NDArray[np.float64]],
    kernel: SignalKernel[Series],
    units: Mapping[str, str],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
    mode: Normalization,
    row_groups: npt.NDArray[np.int_] | None = None,
    row_labels: tuple[str, ...] | None = None,
) -> FeatureTable:
    """Apply a kernel across bands, windows and spatial groups.

    ``row_groups`` maps each epoch to an output row, for measures estimated across
    trials rather than within one. The kernel then returns one value per row
    instead of per epoch, and coverage is averaged over each group's epochs.
    """
    _check_signals(signals, windows)
    if (row_groups is None) != (row_labels is None):
        raise ValueError("row_groups and row_labels must be given together.")
    columns: list[_Column] = []
    flag_columns: dict[str, list[npt.NDArray[np.bool_]]] = {}

    for signal in signals:
        trace = trace_of(signal)
        by_measure: dict[str, list[npt.NDArray[np.float64]]] = {}
        coverages: list[npt.NDArray[np.float64]] = []
        for window in windows:
            mask = window_mask(signal.times, window)
            measured = kernel(signal, trace[:, :, mask], signal.times[mask])
            for name, values in measured.items():
                by_measure.setdefault(name, []).append(values)
            per_epoch = signal.coverage[:, :, mask].mean(axis=2)
            coverages.append(
                per_epoch
                if row_groups is None
                else _reduce_rows(per_epoch, row_groups, len(row_labels or ()))
            )
        coverage = np.stack(coverages, axis=2)

        for measure, per_window in by_measure.items():
            if measure not in units:
                raise ValueError(f"kernel returned measure {measure!r} with no unit declared.")
            stacked = np.stack(per_window, axis=2)
            spatial_units = aggregate(stacked, coverage, signal.ch_names, groups, include_global)

            def make_meta(
                spatial: SpatialUnit,
                window: Window,
                _m: str = measure,
                _s: Series = signal,
            ) -> FeatureMeta:
                return FeatureMeta(
                    measure=_m,
                    band=_s.band,
                    space=spatial.space,
                    space_kind=spatial.space_kind,
                    window=window.name,
                    normalization=mode,
                    unit=units[_m],
                    source=_s.source,
                    freq_resolution_hz=None,
                )

            new_columns, new_flags = _collect(
                spatial_units, windows, {}, make_meta, skip_window=None
            )
            columns.extend(new_columns)
            for key, arrays in new_flags.items():
                flag_columns.setdefault(key, []).extend(arrays)

    return _assemble(columns, flag_columns, row_labels)


def _reduce_rows(
    per_epoch: npt.NDArray[np.float64],
    row_groups: npt.NDArray[np.int_],
    n_rows: int,
) -> npt.NDArray[np.float64]:
    return np.stack([per_epoch[row_groups == row].mean(axis=0) for row in range(n_rows)])


def _check_signals(signals: Sequence[TimeSeries], windows: Sequence[Window]) -> None:
    if not signals:
        raise ValueError("expand_signal requires at least one BandSignal.")
    if not windows:
        raise ValueError("expand_signal requires at least one window.")
    first = signals[0]
    for signal in signals[1:]:
        if signal.ch_names != first.ch_names:
            raise ValueError(
                "all signals must share the same channels; got "
                f"{first.ch_names} and {signal.ch_names}."
            )
        if signal.times.shape != first.times.shape or not np.allclose(signal.times, first.times):
            raise ValueError("all signals must share the same time axis.")


def window_mask(times: npt.NDArray[np.float64], window: Window) -> npt.NDArray[np.bool_]:
    """Boolean mask of the samples a window covers, inclusive of both bounds."""
    mask = (times >= window.tmin) & (times <= window.tmax)
    if not mask.any():
        raise ValueError(
            f"window {window.name!r} ({window.tmin}, {window.tmax}) selects no samples "
            f"from a time axis spanning ({times[0]}, {times[-1]})."
        )
    return mask
