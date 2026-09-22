"""Run a recipe over every recording it selects.

Each recording is processed independently: one that fails is recorded and the
run moves on, so a batch finishes even when some inputs are unusable. Recordings
can be computed in parallel worker processes. Results are written per recording,
mirroring the input tree under the output root; a failure is written beside the
results it did not produce, and a run log in the output root, rewritten after every
recording, records what happened to each. :func:`status` reads them back, so a run
can resume where an earlier one stopped.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import queue
import time
import tomllib
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from eegfeat.io import write_table
from eegfeat.runner.compute import RecordingFeatures, compute_features, event_names
from eegfeat.runner.progress import NullReporter, Reporter
from eegfeat.runner.recipe import Inputs, Recipe, RoiPattern

RUN_LOG = "eegfeat_run.json"
"""Name of the run log written to the output root."""

_EPOCH_SUFFIXES = ("_epo", "-epo")
_FIF_SUFFIXES = (".fif.gz", ".fif")
_TABLES = ("features", "crosstrial")
_FAILURE = "failed"
# Spawn rather than fork: forking a process that has loaded MNE, BLAS and their
# thread pools can deadlock the child.
_WORKER_CONTEXT = "spawn"
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_POLL_SECONDS = 0.2
QUICK_EPOCHS = 4
"""How many epochs ``check(quick=True)`` computes."""
# Where a recipe reads and writes, as opposed to what it computes.
_LOCATION_KEYS = (("inputs", "root"), ("inputs", "pattern"), ("output", "root"))

RecordingState = Literal["done", "missing", "failed", "stale", "partial"]
STATES: tuple[RecordingState, ...] = ("done", "missing", "failed", "stale", "partial")
"""Every state :func:`status` reports, in the order it counts them."""


class RunError(Exception):
    """A run that cannot start: no inputs, clashing outputs, or results in the way."""


class TrialError(Exception):
    """The first recording failed when :func:`check` tried a recipe on it."""

    def __init__(self, recording: Recording, cause: Exception) -> None:
        self.recording = recording
        super().__init__(f"{recording.label}: {_describe_error(cause)}")


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
        return tuple(path for name in _TABLES for path in self._bundle(name))

    @property
    def failure_path(self) -> Path:
        """Where a failure to compute this recording is recorded."""
        return self.base.with_name(f"{self.base.name}_{_FAILURE}.json")

    def _bundle(self, name: str) -> tuple[Path, Path, Path]:
        table = self._table(name)
        return (
            table,
            table.with_suffix(".json"),
            self.base.with_name(f"{self.base.name}_{name}_coverage.tsv"),
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
    """What happened to every recording in a run.

    ``skipped`` holds the recordings a resumed run left alone because their
    results were already up to date.
    """

    recordings: tuple[RecordingResult, ...]
    seconds: float
    skipped: tuple[Recording, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every recording succeeded."""
        return all(result.success for result in self.recordings)

    @property
    def failed(self) -> tuple[RecordingResult, ...]:
        """The recordings that did not."""
        return tuple(result for result in self.recordings if not result.success)


@dataclass(frozen=True)
class RecordingStatus:
    """Where one recording's results stand against a recipe.

    ``done``: complete, and written by this computation from the current input.
    ``missing``: no results. ``failed``: no results, and the last run failed on
    it. ``stale``: results of a different computation, or older than the input.
    ``partial``: files of a run are gone. ``reason`` says why, in a sentence.
    """

    recording: Recording
    state: RecordingState
    reason: str
    outputs: tuple[Path, ...] = ()

    @property
    def label(self) -> str:
        """The recording's label."""
        return self.recording.label


@dataclass(frozen=True, eq=False)
class Trial:
    """What computing the first recording produced, and how long it took.

    ``n_epochs`` were computed out of the recording's ``epochs_total``; fewer only
    in a quick check. ``seconds`` covers reading and computing, ``read_seconds`` the
    reading alone.
    """

    recording: Recording
    n_epochs: int
    channels: tuple[str, ...]
    features: RecordingFeatures
    seconds: float = 0.0
    epochs_total: int = 0
    read_seconds: float = 0.0


@dataclass(frozen=True, eq=False)
class CheckReport:
    """A recipe tried on its first recording, without writing anything.

    ``missing_channels`` maps the label of each recording a run would fail to the
    channels the recipe names that it will not have once picked. ``trial`` is None
    when the first recording is one of them: it would fail for a reason already
    reported, from inside whichever measure named the channel first.
    """

    recordings: tuple[Recording, ...]
    existing: tuple[Path, ...]
    trial: Trial | None
    missing_channels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


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
    return _pick(mne.read_epochs(path, preload=True, verbose="error"), inputs)


def _pick(epochs: Any, inputs: Inputs) -> Any:
    picks = list(inputs.picks) if isinstance(inputs.picks, tuple) else inputs.picks
    epochs.pick(picks, exclude="bads" if inputs.exclude_bads else ())
    # MNE retains explicitly named bad channels regardless of pick's exclude argument.
    if inputs.exclude_bads and epochs.info["bads"]:
        epochs.drop_channels(epochs.info["bads"])
    return epochs


def run(
    recipe: Recipe,
    *,
    overwrite: bool = False,
    resume: bool = False,
    n_jobs: int = 1,
    workers: int = 1,
    reporter: Reporter | None = None,
) -> RunResult:
    """Compute a recipe's features for every recording it selects.

    Parameters
    ----------
    recipe : Recipe
        What to compute, and for which files.
    overwrite : bool, default False
        Replace results from an earlier run. Without it, finding any results the
        run would replace is an error, raised before anything is computed.
    resume : bool, default False
        Compute only the recordings whose :func:`status` is not ``done``. Stale
        and partial results are still replaced only with ``overwrite``.
    n_jobs : int, default 1
        Passed to MNE's filtering and spectral estimation.
    workers : int, default 1
        Recordings computed at once, each in its own process.
    reporter : Reporter, optional
        Receives progress. Nothing is reported by default.

    Returns
    -------
    RunResult
        One result per recording in input order, failed ones included.

    Raises
    ------
    RunError
        When the run cannot start.
    """
    report: Reporter = NullReporter() if reporter is None else reporter
    if workers < 1:
        raise RunError(f"workers must be at least 1, got {workers}.")
    recordings = discover(recipe)
    skipped: tuple[Recording, ...] = ()
    if resume:
        statuses = _statuses(recipe, recordings)
        skipped = tuple(entry.recording for entry in statuses if entry.state == "done")
        pending = [entry for entry in statuses if entry.state != "done"]
        recordings = tuple(entry.recording for entry in pending)
        in_the_way = [entry for entry in pending if entry.outputs]
        if in_the_way and not overwrite:
            first, n = in_the_way[0], len(in_the_way)
            raise RunError(
                f"{n} recording{'s' if n != 1 else ''} to compute already "
                f"{'have' if n != 1 else 'has'} results, e.g. {first.label} ({first.state}: "
                f"{first.reason}); add --overwrite (overwrite=True) to replace them."
            )
    else:
        existing = _existing(recordings)
        if existing and not overwrite:
            raise RunError(
                f"{len(existing)} result files from an earlier run are in the way, e.g. "
                f"{existing[0]}; rerun with --resume (resume=True) to compute only what "
                "is not up to date, or --overwrite (overwrite=True) to replace them."
            )

    started = datetime.now(UTC)
    clock = time.perf_counter()
    report.start([r.label for r in recordings], recipe.output.root)
    finished: dict[int, RecordingResult] = {}

    def record(position: int, result: RecordingResult) -> None:
        finished[position] = result
        _record_failure(result, recipe)
        # Rewritten after every recording, so a run that is killed still says what it did.
        so_far = RunResult(tuple(finished[k] for k in sorted(finished)), 0.0, skipped)
        _write_log(recipe, so_far, started, time.perf_counter() - clock, finished=False)
        report.recording_done(result.label, result.success, _summary(result))

    if workers == 1 or len(recordings) == 1:
        for position, recording in enumerate(recordings, start=1):
            report.recording_start(recording.label, position, len(recordings))
            record(position, _process(recording, recipe, n_jobs, report, overwrite=overwrite))
    else:
        _run_parallel(recordings, recipe, n_jobs, workers, overwrite, report, record)

    results = tuple(finished[k] for k in sorted(finished))
    outcome = RunResult(results, time.perf_counter() - clock, skipped)
    # A run with nothing to compute keeps the log of the run that did the work.
    if results:
        _write_log(recipe, outcome, started, outcome.seconds, finished=True)
    report.complete(outcome.ok, outcome.seconds, _run_summary(outcome), [str(recipe.output.root)])
    return outcome


def _run_parallel(
    recordings: tuple[Recording, ...],
    recipe: Recipe,
    n_jobs: int,
    workers: int,
    overwrite: bool,
    report: Reporter,
    record: Callable[[int, RecordingResult], None],
) -> None:
    context = multiprocessing.get_context(_WORKER_CONTEXT)
    steps = context.Queue()
    announced: set[int] = set()

    def pool_round(waiting: deque[tuple[int, Recording]], size: int) -> list[tuple[int, Recording]]:
        """Compute what is waiting; if a worker dies, return what was in flight."""
        with ProcessPoolExecutor(
            max_workers=size, mp_context=context, initializer=_start_worker, initargs=(steps,)
        ) as pool:
            running: dict[Future[RecordingResult], tuple[int, Recording]] = {}
            while waiting or running:
                while waiting and len(running) < size:
                    position, recording = waiting[0]
                    try:
                        future = pool.submit(_work, recording, recipe, n_jobs, overwrite)
                    except BrokenProcessPool:
                        # A worker died since the last poll; this one never started.
                        return list(running.values())
                    waiting.popleft()
                    running[future] = (position, recording)
                    if position not in announced:
                        announced.add(position)
                        report.recording_start(recording.label, position, len(recordings))
                done, _ = wait(running, timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
                _forward_steps(steps, report)
                broken = False
                for future in done:
                    # Only a dead worker raises here: _process catches everything else.
                    if future.exception() is not None:
                        broken = True
                        continue
                    position, _ = running.pop(future)
                    record(position, future.result())
                if broken:
                    return list(running.values())
        return []

    waiting = deque(enumerate(recordings, start=1))
    with _thread_limits(workers):
        while waiting:
            # A dying worker breaks the whole pool and fails every future in it, so the
            # recordings in flight are rerun one at a time: whichever breaks a pool of
            # its own was the cause.
            for suspect in pool_round(waiting, min(workers, len(waiting))):
                if pool_round(deque([suspect]), 1):
                    position, recording = suspect
                    record(
                        position,
                        RecordingResult(
                            recording,
                            success=False,
                            seconds=0.0,
                            error="the worker process computing it exited unexpectedly "
                            "(out of memory, or a crash in compiled code)",
                        ),
                    )
    _forward_steps(steps, report)


@contextmanager
def _thread_limits(workers: int) -> Iterator[None]:
    # Spawned workers read these as they start, before numpy sizes its thread pools.
    limits = _worker_threads(workers, os.cpu_count() or 1, os.environ)
    saved = {name: os.environ.get(name) for name in limits}
    os.environ.update(limits)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_threads(workers: int, cpu_count: int, environ: Mapping[str, str]) -> dict[str, str]:
    """Thread-pool sizes that split the cores between workers, leaving any the user set."""
    share = str(max(1, cpu_count // workers))
    return {name: share for name in _THREAD_VARIABLES if name not in environ}


_worker_steps: Any = None


def _start_worker(steps: Any) -> None:
    global _worker_steps
    _worker_steps = steps


def _work(recording: Recording, recipe: Recipe, n_jobs: int, overwrite: bool) -> RecordingResult:
    return _process(recording, recipe, n_jobs, _QueueReporter(_worker_steps), overwrite=overwrite)


class _QueueReporter(NullReporter):
    """Sends a worker's steps to the parent, which reports them."""

    def __init__(self, steps: Any) -> None:
        self.steps = steps

    def step(self, label: str, step: str, current: int, total: int) -> None:
        self.steps.put((label, step, current, total))


def _forward_steps(steps: Any, report: Reporter) -> None:
    while True:
        try:
            label, step, current, total = steps.get_nowait()
        except queue.Empty:
            return
        report.step(label, step, current, total)


def _record_failure(result: RecordingResult, recipe: Recipe) -> None:
    path = result.recording.failure_path
    if result.success:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    failure = {
        "input": str(result.recording.source),
        "recipe": str(recipe.path),
        "settings_sha256": settings_sha256(recipe),
        "error": result.error,
        "traceback": result.traceback,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(path, failure)


def status(recipe: Recipe) -> tuple[RecordingStatus, ...]:
    """Say where each recording a recipe selects stands, without computing anything.

    Results count as current when their sidecars record the same computation,
    whatever the recipe says about where its inputs and outputs live: repointing
    ``inputs.root`` after moving the data leaves them current. Results older than
    their input file are stale. Failures come from the run log in the output root.

    Raises
    ------
    RunError
        When the input root is missing or nothing matches.
    """
    return _statuses(recipe, discover(recipe))


def settings_sha256(recipe: Recipe) -> str:
    """Fingerprint what a recipe computes, leaving out where it reads and writes.

    Comments, formatting and key order do not change it; ``inputs.root``,
    ``inputs.pattern`` and ``output.root`` are left out.
    """
    data = tomllib.loads(recipe.text)
    for section, key in _LOCATION_KEYS:
        if isinstance(data.get(section), dict):
            data[section].pop(key, None)
    return _sha256(json.dumps(data, sort_keys=True, default=str))


def _statuses(recipe: Recipe, recordings: tuple[Recording, ...]) -> tuple[RecordingStatus, ...]:
    logged = _last_failures(recipe.output.root)
    current = {settings_sha256(recipe), _sha256(recipe.text)}
    return tuple(
        _status_of(recording, current, _failure_of(recording) or logged.get(recording.label))
        for recording in recordings
    )


def _failure_of(recording: Recording) -> str | None:
    try:
        failure = json.loads(recording.failure_path.read_text())
    except (OSError, ValueError):
        return None
    return str(failure.get("error") or "failed") if isinstance(failure, dict) else None


def _status_of(recording: Recording, current: set[str], failure: str | None) -> RecordingStatus:
    outputs = tuple(path for path in recording.files() if path.exists())
    if not outputs:
        if failure is not None:
            return RecordingStatus(recording, "failed", f"the last run failed: {failure}")
        return RecordingStatus(recording, "missing", "no results yet")

    state, reason = _judge(recording, current)
    if state != "done" and failure is not None:
        reason += f"; the last run failed: {failure}"
    return RecordingStatus(recording, state, reason, outputs)


def _judge(recording: Recording, current: set[str]) -> tuple[RecordingState, str]:
    present: dict[str, dict[str, Any]] = {}
    for name in _TABLES:
        bundle = recording._bundle(name)
        gone = [path.name for path in bundle if not path.exists()]
        if len(gone) == len(bundle):
            continue
        if gone:
            return "partial", f"the {name} table lacks {', '.join(gone)}"
        try:
            present[name] = json.loads(bundle[1].read_text()).get("provenance") or {}
        except (OSError, ValueError, AttributeError) as exc:
            return "partial", f"the {name} sidecar is unreadable ({exc})"

    for provenance in present.values():
        # Sidecars from before runs recorded their tables name none, so only the
        # bundles on disk are checked for them.
        expected = provenance.get("tables", ())
        lost = [name for name in expected if name not in present]
        if lost:
            return "partial", f"the {', '.join(lost)} table is missing"

    for name, provenance in present.items():
        fingerprints = {provenance.get("settings_sha256"), provenance.get("recipe_sha256")}
        if not fingerprints & current:
            return "stale", f"the {name} table was computed by a different recipe"

    oldest = min(recording._table(name).stat().st_mtime for name in present)
    if recording.source.stat().st_mtime > oldest:
        return "stale", "the input file changed after these results were written"
    return "done", "up to date"


def _last_failures(output_root: Path) -> dict[str, str]:
    try:
        log = json.loads((output_root / RUN_LOG).read_text())
        return {
            entry["label"]: entry.get("error") or "failed"
            for entry in log["recordings"]
            if not entry["success"]
        }
    except (OSError, ValueError, KeyError, TypeError):
        # No log, or one this code cannot read: failures simply read as missing.
        return {}


def check(recipe: Recipe, *, n_jobs: int = 1, quick: bool = False) -> CheckReport:
    """Compute a recipe's features for its first recording, writing nothing.

    Catches what loading a recipe cannot: channels an ROI names but the data
    lacks, windows outside the epochs, spectra the data cannot support. Only the
    first recording is computed, but every recording's header is read for the
    channels that ROIs and asymmetry pairs name, since bad channels differ between
    recordings. The trial is timed, entry by entry. ``quick`` computes only its
    first :data:`QUICK_EPOCHS` epochs.

    Raises
    ------
    RunError
        When the run could not start.
    TrialError
        When computing the first recording fails.
    """
    recordings = discover(recipe)
    missing = _missing_channels(recipe, recordings)
    report = CheckReport(
        recordings=recordings,
        existing=_existing(recordings),
        trial=None,
        missing_channels=missing,
    )
    first = recordings[0]
    if first.label in missing:
        return report
    clock = time.perf_counter()
    try:
        epochs = load_epochs(first.source, recipe.inputs)
        read_seconds = time.perf_counter() - clock
        total = len(epochs)
        if quick and total > QUICK_EPOCHS:
            epochs = epochs[:QUICK_EPOCHS]
        features = compute_features(
            epochs, recipe, recording=first.source.as_posix(), n_jobs=n_jobs
        )
        if features.epochs is not None:
            epoch_rows(epochs, recipe.output.epoch_metadata)
    except Exception as exc:  # noqa: BLE001 - reported with the recording it came from
        raise TrialError(first, exc) from exc
    return replace(
        report,
        trial=Trial(
            recording=first,
            n_epochs=len(epochs),
            channels=tuple(epochs.ch_names),
            features=features,
            seconds=time.perf_counter() - clock,
            epochs_total=total,
            read_seconds=read_seconds,
        ),
    )


def _missing_channels(
    recipe: Recipe, recordings: tuple[Recording, ...]
) -> dict[str, tuple[str, ...]]:
    uses: dict[str, list[str]] = {}
    patterns: dict[str, RoiPattern] = {}
    if any("rois" in spec.spatial for spec in recipe.features):
        for roi, members in recipe.rois.items():
            if isinstance(members, RoiPattern):
                patterns[roi] = members
                continue
            for channel in members:
                uses.setdefault(channel, []).append(f"ROI {roi!r}")
    for spec in recipe.features:
        for left, right in spec.asymmetry:
            for channel in (left, right):
                uses.setdefault(channel, []).append(f"asymmetry pair {left}/{right}")
    if not uses and not patterns:
        return {}

    missing: dict[str, tuple[str, ...]] = {}
    for recording in recordings:
        # Picking needs loaded data, so pick on a one-sample stand-in with the
        # recording's own info: the same selection load_epochs makes, without reading data.
        info = mne.io.read_info(recording.source, verbose="error")
        stand_in = mne.EpochsArray(np.zeros((1, info["nchan"], 1)), info, verbose="error")
        kept = set(_pick(stand_in, recipe.inputs).ch_names)
        problems = []
        for channel, named_by in uses.items():
            if channel in kept:
                continue
            if channel in info["bads"] and recipe.inputs.exclude_bads:
                reason = "is marked bad, and inputs.exclude_bads leaves it out"
            elif channel in info["ch_names"]:
                reason = "is not among the channels inputs.picks keeps"
            else:
                reason = "is not in the recording"
            problems.append(f"{channel} ({', '.join(dict.fromkeys(named_by))}) {reason}")
        for roi, pattern in patterns.items():
            if not pattern.resolve(sorted(kept, key=info["ch_names"].index)):
                problems.append(
                    f"ROI {roi!r} matches none of the channels kept, with {list(pattern.patterns)}"
                )
        if problems:
            missing[recording.label] = tuple(problems)
    return missing


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
        report.step(label, "read", 1, total)
        epochs = load_epochs(recording.source, recipe.inputs)
        features = compute_features(
            epochs,
            recipe,
            recording=recording.source.as_posix(),
            n_jobs=n_jobs,
            on_step=lambda measure, current, _: report.step(label, measure, current + 1, total),
        )
        report.step(label, "write", total, total)
        outputs = _stage_and_publish(recording, features, epochs, recipe, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 - one bad recording must not end the batch
        return RecordingResult(
            recording,
            success=False,
            seconds=time.perf_counter() - clock,
            error=_describe_error(exc),
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


def _stage_and_publish(
    recording: Recording,
    features: RecordingFeatures,
    epochs: Any,
    recipe: Recipe,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    destination = recording.base.parent
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".eegfeat-", dir=destination) as temporary:
        staging = Path(temporary)
        staged = Recording(
            source=recording.source,
            label=recording.label,
            base=staging / recording.base.name,
        )
        staged_outputs = _write(staged, features, epochs, recipe)
        _validate_staged_tables(staged_outputs)
        final_outputs = tuple(destination / path.name for path in staged_outputs)
        _publish(
            staged_outputs,
            final_outputs,
            existing=recording.files() if overwrite else (),
            backup_root=staging / "backup",
        )
    return final_outputs


def _validate_staged_tables(outputs: tuple[Path, ...]) -> None:
    from eegfeat.io import read_table

    for path in outputs:
        if path.suffix == ".tsv" and not path.stem.endswith("_coverage"):
            read_table(path)


def _publish(
    staged: tuple[Path, ...],
    final: tuple[Path, ...],
    *,
    existing: tuple[Path, ...],
    backup_root: Path,
) -> None:
    backup_root.mkdir()
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for path in existing:
            if path.exists():
                backup = backup_root / path.name
                os.replace(path, backup)
                backups.append((backup, path))
        for source, target in zip(staged, final, strict=True):
            os.replace(source, target)
            published.append(target)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        for backup, target in backups:
            os.replace(backup, target)
        raise


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
        "settings_sha256": settings_sha256(recipe),
        "tables": [
            name
            for name, table in zip(_TABLES, (features.epochs, features.crosstrial), strict=True)
            if table is not None
        ],
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


def _write_log(
    recipe: Recipe, outcome: RunResult, started: datetime, seconds: float, *, finished: bool
) -> None:
    from eegfeat import __version__

    recipe.output.root.mkdir(parents=True, exist_ok=True)
    log = {
        "eegfeat_version": __version__,
        "mne_version": mne.__version__,
        "python_version": platform.python_version(),
        "recipe": str(recipe.path),
        "recipe_sha256": _sha256(recipe.text),
        "recipe_text": recipe.text,
        "settings_sha256": settings_sha256(recipe),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished,
        "seconds": round(seconds, 3),
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
        "skipped": [recording.label for recording in outcome.skipped],
    }
    _write_json(recipe.output.root / RUN_LOG, log)


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    # Replaced whole, so a reader never sees a half-written file.
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(content, indent=2) + "\n")
    os.replace(staging, path)


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


def _describe_error(exc: Exception) -> str:
    # compute_features notes which recipe entry failed; lead with it.
    context = "".join(f"{note}: " for note in getattr(exc, "__notes__", ()))
    return f"{context}{type(exc).__name__}: {exc}"


def _run_summary(outcome: RunResult) -> str:
    if not outcome.recordings:
        return f"all {len(outcome.skipped)} recordings are up to date; nothing to compute"
    done = len(outcome.recordings) - len(outcome.failed)
    line = f"{done} of {len(outcome.recordings)} recordings succeeded in {outcome.seconds:.1f} s"
    if outcome.skipped:
        line += f"; {len(outcome.skipped)} up to date"
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
