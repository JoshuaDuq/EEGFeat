"""Compute a recipe's features for one recording.

The runner is where the choices the library leaves to its caller get made: how
spectra are estimated, how band signals are filtered, how trials are grouped.
Each of those inputs is built at most once per recording, and only if some entry
in the recipe needs it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from mne.time_frequency import (  # type: ignore[import-untyped]
    psd_array_multitaper,
    psd_array_welch,
)

from eegfeat._expand import window_mask
from eegfeat.bands import Band
from eegfeat.derived import asymmetry, band_ratio
from eegfeat.identity import epoch_row_ids
from eegfeat.microstates import MicrostateSegmentation, segment
from eegfeat.runner.measures import GRAPH, MEASURES, Measure
from eegfeat.runner.recipe import WHOLE_EPOCH, FeatureSpec, Recipe
from eegfeat.signal import BandSignal, Signal
from eegfeat.spectra import Spectra, Window
from eegfeat.table import ComputationSpec, FeatureTable, concat

OnStep = Callable[[str, int, int], None]
"""Called before each entry with its measure, its position and the entry count."""

_WELCH_SEGMENT_SEC = 2.0
"""Default Welch segment length, capped by the shortest window it must fit."""

_WELCH_TAPER = "hann"


@dataclass(frozen=True, eq=False)
class RecordingFeatures:
    """One recording's features, split by what their rows are.

    Parameters
    ----------
    epochs : FeatureTable or None
        Measures estimated within each epoch, one row per epoch.
    crosstrial : FeatureTable or None
        Measures estimated across trials, one row per trial group.
    """

    epochs: FeatureTable | None
    crosstrial: FeatureTable | None


def compute_features(
    epochs: Any,
    recipe: Recipe,
    *,
    recording: str,
    n_jobs: int = 1,
    on_step: OnStep | None = None,
) -> RecordingFeatures:
    """Compute every entry of a recipe for one recording.

    Parameters
    ----------
    epochs : mne.Epochs
        Preloaded epochs, already reduced to the channels to analyse.
    recipe : Recipe
        What to compute.
    n_jobs : int, default 1
        Passed to MNE's filtering and spectral estimation.
    on_step : callable, optional
        Progress callback, see :data:`OnStep`.

    Returns
    -------
    RecordingFeatures
    """
    inputs = RecordingInputs(epochs, recipe, recording=recording, n_jobs=n_jobs)
    per_epoch: list[FeatureTable] = []
    crosstrial: list[FeatureTable] = []
    for position, spec in enumerate(recipe.features, start=1):
        if on_step is not None:
            on_step(spec.measure, position, len(recipe.features))
        try:
            table = _compute(spec, inputs)
        except Exception as exc:
            # Keep the exception's type for callers; the note names the entry the way
            # recipe problems do, so it can be found in the file.
            exc.add_note(f"features[{position - 1}] ({spec.measure})")
            raise
        (per_epoch if table.row_labels is None else crosstrial).append(table)
    return RecordingFeatures(
        epochs=concat(per_epoch) if per_epoch else None,
        crosstrial=concat(crosstrial) if crosstrial else None,
    )


def event_names(epochs: Any) -> list[str]:
    """The event name of each epoch."""
    names = {code: name for name, code in epochs.event_id.items()}
    return [names[int(code)] for code in epochs.events[:, 2]]


class RecordingInputs:
    """The inputs a recording's measures draw on, each built on first use."""

    def __init__(self, epochs: Any, recipe: Recipe, *, recording: str, n_jobs: int) -> None:
        self.epochs = epochs
        self.recipe = recipe
        self.recording = recording
        self.n_jobs = n_jobs
        self.times: npt.NDArray[np.float64] = np.asarray(epochs.times, dtype=float)
        self.sfreq = float(epochs.info["sfreq"])
        self._data: npt.NDArray[np.float64] | None = None
        self._signal: Signal | None = None
        self._band_signals: dict[Band, BandSignal] = {}
        self._psd: dict[Window, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
        self._tfr: tuple[Any, npt.NDArray[np.float64]] | None = None
        self._segmentation: MicrostateSegmentation | None = None

    def window(self, window: Window) -> Window:
        """The window as this recording measures it: finite and inside its epochs."""
        first, last = float(self.times[0]), float(self.times[-1])
        if math.isinf(window.tmin) and math.isinf(window.tmax):
            return Window(window.name, first, last)
        tolerance = 0.5 / self.sfreq
        if window.tmin < first - tolerance or window.tmax > last + tolerance:
            raise ValueError(
                f"window {window.name!r} ({window.tmin}, {window.tmax}) s reaches outside this "
                f"recording's epochs, which span ({first:.3f}, {last:.3f}) s."
            )
        return window

    def windows(self, windows: Sequence[Window]) -> tuple[Window, ...]:
        """:meth:`window` applied to each."""
        return tuple(self.window(window) for window in windows)

    def signal(self) -> Signal:
        """The broadband epochs."""
        if self._signal is None:
            self._signal = Signal.from_epochs(self.epochs, recording=self.recording)
        return self._signal

    def band_signal(self, band: Band) -> BandSignal:
        """The epochs bandpassed to ``band``, with their analytic signal."""
        if band not in self._band_signals:
            settings = self.recipe.band_signal
            self._band_signals[band] = BandSignal.from_epochs(
                self.epochs,
                band,
                recording=self.recording,
                pad_sec=settings.pad_sec,
                pad_cycles=settings.pad_cycles,
                n_jobs=self.n_jobs,
            )
        return self._band_signals[band]

    def spectra(self, windows: Sequence[Window]) -> Spectra:
        """Power spectra over the given windows, by the recipe's method."""
        finite = self.windows(windows)
        if self.recipe.spectra.method == "morlet":
            tfr, n_cycles = self._morlet()
            return Spectra.from_tfr(tfr, finite, recording=self.recording, n_cycles=n_cycles)

        estimates = [self._window_psd(window) for window in finite]
        freqs = estimates[0][1]
        for (_, other), window in zip(estimates[1:], finite[1:], strict=True):
            if not np.array_equal(other, freqs):
                raise ValueError(
                    f"window {window.name!r} yields a different frequency grid from "
                    f"{finite[0].name!r}. Multitaper grids follow the window length, so windows "
                    "measured together must be equally long; or use welch or morlet."
                )
        data = np.stack([psd for psd, _ in estimates], axis=2)
        return Spectra(
            data=data,
            freqs=freqs,
            ch_names=tuple(self.epochs.ch_names),
            windows=finite,
            coverage=np.isfinite(data).astype(float),
            source=self.recipe.spectra.method,
            representation="psd",
            support=np.ones(data.shape, dtype=float),
            row_ids=epoch_row_ids(self.epochs, self.recording, data.shape[0]),
            computation=ComputationSpec.create(
                self.recipe.spectra.method,
                normalization="full" if self.recipe.spectra.method == "multitaper" else "density",
                settings={
                    key: value
                    for key, value in self.recipe.spectra.__dict__.items()
                    if value is not None
                },
            ),
        )

    def trials(self) -> tuple[str, ...] | None:
        """One group label per epoch for cross-trial measures, or None for one group."""
        grouping = self.recipe.trials
        if grouping.by == "all":
            return None
        if grouping.by == "event":
            return tuple(event_names(self.epochs))
        metadata = self.epochs.metadata
        if metadata is None or grouping.column not in metadata.columns:
            raise ValueError(
                f"trials are grouped by the metadata column {grouping.column!r}, "
                "which this recording does not have."
            )
        labels = metadata[grouping.column]
        missing = int(labels.isna().sum())
        if missing:
            raise ValueError(
                f"the metadata column {grouping.column!r} is empty for {missing} epochs, "
                "and every epoch needs a trial group."
            )
        return tuple(str(label) for label in labels)

    def segmentation(self) -> MicrostateSegmentation:
        """Microstate templates fitted to this recording, shared by its measures."""
        if self._segmentation is None:
            self._segmentation = segment(self.signal(), **self.recipe.microstates)  # type: ignore[arg-type]
        return self._segmentation

    def _array(self) -> npt.NDArray[np.float64]:
        if self._data is None:
            self._data = np.asarray(self.epochs.get_data(), dtype=float)
        return self._data

    def _window_psd(
        self, window: Window
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        cached = self._psd.get(window)
        if cached is not None:
            return cached
        settings = self.recipe.spectra
        data = self._array()[:, :, window_mask(self.times, window)]
        if settings.method == "welch":
            n_fft = self._n_fft()
            if n_fft > data.shape[-1]:
                raise ValueError(
                    f"n_fft = {n_fft} is longer than window {window.name!r}, which holds "
                    f"{data.shape[-1]} samples; lower n_fft or widen the window."
                )
            resolution = self.sfreq / n_fft
            psd, freqs = psd_array_welch(
                data,
                self.sfreq,
                fmin=max(0.0, settings.fmin - resolution),
                fmax=min(self.sfreq / 2.0, settings.fmax + resolution),
                n_fft=n_fft,
                n_overlap=n_fft // 2 if settings.n_overlap is None else settings.n_overlap,
                window=_WELCH_TAPER,
                n_jobs=self.n_jobs,
                verbose=False,
            )
        else:
            resolution = self.sfreq / data.shape[-1]
            psd, freqs = psd_array_multitaper(
                data,
                self.sfreq,
                fmin=max(0.0, settings.fmin - resolution),
                fmax=min(self.sfreq / 2.0, settings.fmax + resolution),
                bandwidth=settings.bandwidth,
                normalization="full",
                n_jobs=self.n_jobs,
                verbose=False,
            )
        estimate = (np.asarray(psd, dtype=float), np.asarray(freqs, dtype=float))
        self._psd[window] = estimate
        return estimate

    def _n_fft(self) -> int:
        """Segment length shared by every Welch window, so their grids align."""
        if self.recipe.spectra.n_fft is not None:
            return self.recipe.spectra.n_fft
        spectral = {
            window
            for spec in self.recipe.features
            if MEASURES[spec.measure].kind == "spectra"
            for window in (*spec.windows, *((spec.baseline,) if spec.baseline else ()))
        }
        shortest = min(
            int(window_mask(self.times, self.window(window)).sum())
            for window in (spectral or {WHOLE_EPOCH})
        )
        return min(int(round(_WELCH_SEGMENT_SEC * self.sfreq)), shortest)

    def _morlet(self) -> tuple[Any, npt.NDArray[np.float64]]:
        if self._tfr is None:
            settings = self.recipe.spectra
            # geomspace rather than logspace of the log bounds: only geomspace returns
            # fmax itself, and fmax is usually a band edge that then has to be integrated.
            freqs = (
                np.geomspace(settings.fmin, settings.fmax, settings.n_freqs)
                if settings.spacing == "log"
                else np.linspace(settings.fmin, settings.fmax, settings.n_freqs)
            )
            n_cycles = np.clip(
                freqs / settings.n_cycles_factor, settings.min_cycles, settings.max_cycles
            )
            tfr = self.epochs.compute_tfr(
                "morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                picks="all",
                decim=settings.decim,
                output="power",
                average=False,
                return_itc=False,
                n_jobs=self.n_jobs,
                verbose=False,
            )
            self._tfr = (tfr, n_cycles)
        return self._tfr


def _compute(spec: FeatureSpec, inputs: RecordingInputs) -> FeatureTable:
    measure = MEASURES[spec.measure]
    function = measure.function
    params: dict[str, Any] = dict(spec.params)
    windows = inputs.windows(spec.windows)
    rois = dict(inputs.recipe.rois)
    has_global = measure.takes("include_global")

    if measure.kind == "spectra":
        return _spectral(spec, measure, inputs, params, rois)

    if measure.kind == "series":
        series = [inputs.signal() if b is None else inputs.band_signal(b) for b in spec.series]
        return _over_space(
            spec.spatial,
            rois,
            lambda **space: function(series, windows=windows, **params, **space),
            has_global=has_global,
        )

    if measure.kind == "signals":
        signals = [inputs.band_signal(band) for band in spec.bands]
        if spec.baseline is not None:
            params["baseline"] = inputs.window(spec.baseline)
        if measure.takes("trials"):
            params["trials"] = inputs.trials()
        table = _over_space(
            spec.spatial,
            rois,
            lambda **space: function(signals, windows=windows, **params, **space),
            has_global=has_global,
        )
        return _with_graph(spec, table)

    if measure.kind == "pac":
        return concat(
            [
                _over_space(
                    spec.spatial,
                    rois,
                    lambda _p=phase, _a=amplitude, **space: function(
                        inputs.band_signal(_p),
                        inputs.band_signal(_a),
                        windows=windows,
                        **params,
                        **space,
                    ),
                    has_global=has_global,
                )
                for phase, amplitude in spec.pairs
            ]
        )

    if measure.kind == "wpli":
        params["trials"] = inputs.trials()
        table = _over_space(
            spec.spatial,
            rois,
            lambda **space: function(
                inputs.signal(), bands=spec.bands, windows=windows, **params, **space
            ),
            has_global=has_global,
        )
        return _with_graph(spec, table)

    segmented: FeatureTable = function(inputs.segmentation(), windows=windows)
    return segmented


def _spectral(
    spec: FeatureSpec,
    measure: Measure,
    inputs: RecordingInputs,
    params: dict[str, Any],
    rois: Mapping[str, tuple[str, ...]],
) -> FeatureTable:
    function = measure.function
    # Power reductions consume their baseline window, so spectra must include it.
    spectra = inputs.spectra((*spec.windows, *((spec.baseline,) if spec.baseline else ())))

    if measure.takes("bands"):
        if spec.baseline is not None:
            params["baseline"] = spec.baseline.name
        power = _over_space(
            spec.spatial,
            rois,
            lambda **space: function(spectra, bands=spec.bands, **params, **space),
            has_global=True,
        )
        derived = [
            band_ratio(power, numerator, denominator) for numerator, denominator in spec.ratios
        ]
        if spec.asymmetry:
            derived.append(asymmetry(power, spec.asymmetry))
        return concat([power, *derived])

    if measure.takes("band"):
        return concat(
            [
                _over_space(
                    spec.spatial,
                    rois,
                    lambda _b=band, **space: function(spectra, band=_b, **params, **space),
                    has_global=True,
                )
                for band in spec.bands
            ]
        )

    return _over_space(
        spec.spatial,
        rois,
        lambda **space: function(spectra, **params, **space),
        has_global=True,
    )


def _over_space(
    levels: tuple[str, ...],
    rois: Mapping[str, tuple[str, ...]],
    call: Callable[..., FeatureTable],
    *,
    has_global: bool,
) -> FeatureTable:
    """Call a measure once per spatial level and join the results.

    A library call yields either channels or ROIs, plus the global mean on
    request, so each level is its own call and the global mean is asked for once.
    """
    if not has_global:
        return concat([call(groups=None if level == "channels" else rois) for level in levels])
    wants_global = "global" in levels
    tables = []
    if "channels" in levels:
        tables.append(call(groups=None, include_global=wants_global))
    if "rois" in levels:
        tables.append(call(groups=rois, include_global=wants_global and "channels" not in levels))
    if set(levels) == {"global"}:
        tables.append(call(groups=None, include_global=True).select(space_kind="global"))
    return concat(tables)


def _with_graph(spec: FeatureSpec, table: FeatureTable) -> FeatureTable:
    summaries = [
        (
            GRAPH[name](table, threshold=spec.clustering_threshold)
            if name == "clustering_coefficient"
            else GRAPH[name](table)
        )
        for name in spec.graph
    ]
    return concat([table, *summaries]) if summaries else table
