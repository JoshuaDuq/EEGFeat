"""Run a recipe over every recording it selects.

Each recording is processed independently: one that fails is recorded and the
run moves on, so a batch finishes even when some inputs are unusable. Results are
written per recording, mirroring the input tree under the output root, and a run
log in the output root records what happened to each.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from eegfeat.io import write_table
from eegfeat.runner.compute import RecordingFeatures, compute_features, event_names
from eegfeat.runner.progress import NullReporter, Reporter
from eegfeat.runner.recipe import Inputs, Recipe

RUN_LOG = "eegfeat_run.json"
"""Name of the run log written to the output root."""

_EPOCH_SUFFIXES = ("_epo", "-epo")
_FIF_SUFFIXES = (".fif.gz", ".fif")
_TABLES = ("features", "crosstrial")


class RunError(Exception):
    """A run that cannot start: no inputs, clashing outputs, or results in the way."""


class TrialError(Exception):
    """The first recording failed when :func:`check` tried a recipe on it."""

    def __init__(self, recording: Recording, cause: Exception) -> None:
        self.recording = recording
        super().__init__(f"{recording.label}: {type(cause).__name__}: {cause}")


@dataclass(frozen=True)
class Recording:
    """One input file and where its results go.

    Parameters
    ----------
    source : Path
        The epochs file.
    label : str
        Short unique name, used in progress reports.
    base : Path
        Output path prefix; tables are written as ``<base>_features.tsv`` and so on.
    """

    source: Path
    label: str
    base: Path

    @property
    def features_path(self) -> Path:
        """Where the per-epoch table goes."""
        return self._table("features")

    @property
    def crosstrial_path(self) -> Path:
        """Where the cross-trial table goes."""
        return self._table("crosstrial")

    def files(self) -> tuple[Path, ...]:
        """Every file a run could write for this recording."""
        return tuple(
            path
            for name in _TABLES
            for path in (
                self._table(name),
                self._table(name).with_suffix(".json"),
                self.base.with_name(f"{self.base.name}_{name}_coverage.tsv"),
            )
        )

    def _table(self, name: str) -> Path:
        return self.base.with_name(f"{self.base.name}_{name}.tsv")


@dataclass(frozen=True)
class RecordingResult:
    """What happened to one recording."""

    recording: Recording
    success: bool
    seconds: float
    outputs: tuple[Path, ...] = ()
    summary: str = ""
    error: str | None = None
    traceback: str | None = None

    @property
    def label(self) -> str:
        """The recording's label."""
        return self.recording.label


@dataclass(frozen=True)
class RunResult:
    """What happened to every recording in a run."""

    recordings: tuple[RecordingResult, ...]
    seconds: float

    @property
    def ok(self) -> bool:
        """Whether every recording succeeded."""
        return all(result.success for result in self.recordings)

    @property
    def failed(self) -> tuple[RecordingResult, ...]:
        """The recordings that did not."""
        return tuple(result for result in self.recordings if not result.success)


@dataclass(frozen=True, eq=False)
class CheckReport:
    """A recipe tried on its first recording, without writing anything."""

    recordings: tuple[Recording, ...]
    existing: tuple[Path, ...]
    trial: Recording
    n_epochs: int
    channels: tuple[str, ...]
    features: RecordingFeatures


def discover(recipe: Recipe) -> tuple[Recording, ...]:
    """Find the recordings a recipe selects, in sorted order.

    Raises
    ------
    RunError
        When the input root is missing, nothing matches, or two inputs would
        write to the same outputs.
    """
    root, pattern = recipe.inputs.root, recipe.inputs.pattern
    if not root.is_dir():
        raise RunError(f"inputs.root {root} is not a directory.")
    # Hidden paths are skipped: macOS leaves "._" AppleDouble files beside every file
    # on an external drive, and pathlib's glob matches them where a shell would not.
    sources = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    if not sources:
        raise RunError(f"no files match {pattern!r} under {root}.")

    bases = [
        recipe.output.root / source.parent.relative_to(root) / _stem(source) for source in sources
    ]
    by_base: dict[Path, list[Path]] = {}
    for source, base in zip(sources, bases, strict=True):
        by_base.setdefault(base, []).append(source)
    clashes = {base: found for base, found in by_base.items() if len(found) > 1}
    if clashes:
        base, found = next(iter(clashes.items()))
        raise RunError(
            f"{len(found)} inputs would write to the same results {base}: "
            f"{', '.join(path.name for path in found)}."
        )

    stems = [base.name for base in bases]
    unique = len(set(stems)) == len(stems)
    return tuple(
        Recording(
            source=source,
            label=base.name if unique else base.relative_to(recipe.output.root).as_posix(),
            base=base,
        )
        for source, base in zip(sources, bases, strict=True)
    )


def load_epochs(path: Path, inputs: Inputs) -> Any:
    """Read an epochs file and keep the channels the recipe picks."""
    epochs = mne.read_epochs(path, preload=True, verbose="error")
    picks = list(inputs.picks) if isinstance(inputs.picks, tuple) else inputs.picks
    epochs.pick(picks, exclude="bads" if inputs.exclude_bads else ())
    return epochs


def run(
    recipe: Recipe,
    *,
    overwrite: bool = False,
    n_jobs: int = 1,
    reporter: Reporter | None = None,
) -> RunResult:
    """Compute a recipe's features for every recording it selects.

    Parameters
    ----------
    recipe : Recipe
        What to compute, and for which files.
    overwrite : bool, default False
        Replace results from an earlier run. Without it, finding any is an error,
        raised before anything is computed.
    n_jobs : int, default 1
        Passed to MNE's filtering and spectral estimation.
    reporter : Reporter, optional
        Receives progress. Nothing is reported by default.

    Returns
    -------
    RunResult
        One result per recording, failed ones included.

    Raises
    ------
    RunError
        When the run cannot start.
    """
    report: Reporter = NullReporter() if reporter is None else reporter
    recordings = discover(recipe)
    existing = _existing(recordings)
    if existing and not overwrite:
        raise RunError(
            f"{len(existing)} result files from an earlier run are in the way, e.g. "
            f"{existing[0]}; rerun with --overwrite (overwrite=True) to replace them."
        )

    started = datetime.now(UTC)
    clock = time.perf_counter()
    report.start([r.label for r in recordings], recipe.output.root)
    results = []
    for position, recording in enumerate(recordings, start=1):
        report.recording_start(recording.label, position, len(recordings))
        result = _process(recording, recipe, n_jobs, report, overwrite=overwrite)
        results.append(result)
        report.recording_done(recording.label, result.success, _summary(result))

    outcome = RunResult(tuple(results), time.perf_counter() - clock)
    _write_log(recipe, outcome, started)
    report.complete(outcome.ok, outcome.seconds, _run_summary(outcome), [str(recipe.output.root)])
    return outcome


def check(recipe: Recipe, *, n_jobs: int = 1) -> CheckReport:
    """Compute a recipe's features for its first recording, writing nothing.

    Catches what loading a recipe cannot: channels an ROI names but the data
    lacks, windows outside the epochs, spectra the data cannot support.

    Raises
    ------
    RunError
        When the run could not start.
    TrialError
        When computing the first recording fails.
    """
    recordings = discover(recipe)
    trial = recordings[0]
    try:
        epochs = load_epochs(trial.source, recipe.inputs)
        features = compute_features(epochs, recipe, n_jobs=n_jobs)
        if features.epochs is not None:
            epoch_rows(epochs, recipe.output.epoch_metadata)
    except Exception as exc:  # noqa: BLE001 - reported with the recording it came from
        raise TrialError(trial, exc) from exc
    return CheckReport(
        recordings=recordings,
        existing=_existing(recordings),
        trial=trial,
        n_epochs=len(epochs),
        channels=tuple(epochs.ch_names),
        features=features,
    )


def epoch_rows(epochs: Any, include_metadata: bool) -> pd.DataFrame:
    """Descriptive columns joining each epoch to its event and metadata."""
    rows = pd.DataFrame(
        {"selection": np.asarray(epochs.selection, dtype=int), "event": event_names(epochs)}
    )
    metadata = epochs.metadata
    if not include_metadata or metadata is None:
        return rows
    clashes = sorted({"epoch", *rows.columns} & set(metadata.columns))
    if clashes:
        raise ValueError(
            f"epoch metadata has columns {clashes}, which the feature table uses itself; "
            "rename them or set output.epoch_metadata = false."
        )
    return pd.concat([rows, pd.DataFrame(metadata).reset_index(drop=True)], axis=1)


def _process(
    recording: Recording, recipe: Recipe, n_jobs: int, report: Reporter, *, overwrite: bool
) -> RecordingResult:
    label = recording.label
    clock = time.perf_counter()
    total = len(recipe.features) + 2
    try:
        if overwrite:
            for path in recording.files():
                path.unlink(missing_ok=True)
        report.step(label, "read", 1, total)
        epochs = load_epochs(recording.source, recipe.inputs)
        features = compute_features(
            epochs,
            recipe,
            n_jobs=n_jobs,
            on_step=lambda measure, current, _: report.step(label, measure, current + 1, total),
        )
        report.step(label, "write", total, total)
        outputs = _write(recording, features, epochs, recipe)
    except Exception as exc:  # noqa: BLE001 - one bad recording must not end the batch
        return RecordingResult(
            recording,
            success=False,
            seconds=time.perf_counter() - clock,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    seconds = time.perf_counter() - clock
    return RecordingResult(
        recording,
        success=True,
        seconds=seconds,
        outputs=outputs,
        summary=_describe(len(epochs), features, seconds),
    )


def _write(
    recording: Recording, features: RecordingFeatures, epochs: Any, recipe: Recipe
) -> tuple[Path, ...]:
    recording.base.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "input": str(recording.source),
        "recipe": str(recipe.path),
        "recipe_sha256": _sha256(recipe.text),
        "channels": list(epochs.ch_names),
        "n_epochs": len(epochs),
        "sfreq": float(epochs.info["sfreq"]),
        "mne_version": mne.__version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    written: list[Path] = []
    if features.epochs is not None:
        rows = epoch_rows(epochs, recipe.output.epoch_metadata)
        written += write_table(
            features.epochs, recording.features_path, rows=rows, provenance=provenance
        )
    if features.crosstrial is not None:
        written += write_table(
            features.crosstrial, recording.crosstrial_path, provenance=provenance
        )
    return tuple(written)


def _existing(recordings: tuple[Recording, ...]) -> tuple[Path, ...]:
    return tuple(path for recording in recordings for path in recording.files() if path.exists())


def _write_log(recipe: Recipe, outcome: RunResult, started: datetime) -> None:
    from eegfeat import __version__

    recipe.output.root.mkdir(parents=True, exist_ok=True)
    log = {
        "eegfeat_version": __version__,
        "mne_version": mne.__version__,
        "python_version": platform.python_version(),
        "recipe": str(recipe.path),
        "recipe_sha256": _sha256(recipe.text),
        "recipe_text": recipe.text,
        "started": started.isoformat(timespec="seconds"),
        "seconds": round(outcome.seconds, 3),
        "recordings": [
            {
                "label": result.label,
                "input": str(result.recording.source),
                "success": result.success,
                "seconds": round(result.seconds, 3),
                "outputs": [str(path) for path in result.outputs],
                "error": result.error,
                "traceback": result.traceback,
            }
            for result in outcome.recordings
        ],
    }
    (recipe.output.root / RUN_LOG).write_text(json.dumps(log, indent=2) + "\n")


def _summary(result: RecordingResult) -> str:
    return result.summary if result.success else result.error or "failed"


def _describe(n_epochs: int, features: RecordingFeatures, seconds: float) -> str:
    parts = [f"{n_epochs} epochs"]
    if features.epochs is not None:
        parts.append(f"{len(features.epochs.meta)} features")
    if features.crosstrial is not None:
        table = features.crosstrial
        parts.append(f"{len(table.meta)} cross-trial features × {table.n_rows} groups")
    parts.append(f"{seconds:.1f} s")
    return " · ".join(parts)


def _run_summary(outcome: RunResult) -> str:
    done = len(outcome.recordings) - len(outcome.failed)
    line = f"{done} of {len(outcome.recordings)} recordings succeeded in {outcome.seconds:.1f} s"
    if outcome.failed:
        line += "; failed: " + ", ".join(result.label for result in outcome.failed)
    return line


def _stem(source: Path) -> str:
    name = source.name
    for suffix in _FIF_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for marker in _EPOCH_SUFFIXES:
        if name.endswith(marker) and len(name) > len(marker):
            return name[: -len(marker)]
    return name


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
