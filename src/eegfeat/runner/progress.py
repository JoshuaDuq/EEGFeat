"""Progress reporting for a run: plain text for people, JSON lines for programs.

The JSON events follow the protocol of the EEG_fMRI_Pipeline terminal UI, one
object per line: ``start``, ``subject_start``, ``progress``, ``log``,
``subject_done``, ``complete`` and ``error``. A recording plays the part of a
subject, so a front end written for that protocol can follow a run unchanged.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO

CHECK = "✓"
CROSS = "✗"
_INDENT = "      "


class Reporter(Protocol):
    """Receives a run's progress."""

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        """The run is about to process these recordings."""

    def recording_start(self, label: str, position: int, total: int) -> None:
        """A recording is starting."""

    def step(self, label: str, step: str, current: int, total: int) -> None:
        """A recording has reached a step."""

    def recording_done(self, label: str, success: bool, message: str) -> None:
        """A recording has finished, with a one-line summary or error."""

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        """The run has finished."""


class NullReporter:
    """Reports nothing."""

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        del labels, output_root

    def recording_start(self, label: str, position: int, total: int) -> None:
        del label, position, total

    def step(self, label: str, step: str, current: int, total: int) -> None:
        del label, step, current, total

    def recording_done(self, label: str, success: bool, message: str) -> None:
        del label, success, message

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        del success, seconds, message, outputs


class TextReporter:
    """One line per recording, for a person watching a terminal."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        noun = "recording" if len(labels) == 1 else "recordings"
        self._write(f"eegfeat · {len(labels)} {noun} → {output_root}")

    def recording_start(self, label: str, position: int, total: int) -> None:
        self._write(f"[{position}/{total}] {label}")

    def step(self, label: str, step: str, current: int, total: int) -> None:
        del label, step, current, total

    def recording_done(self, label: str, success: bool, message: str) -> None:
        del label
        self._write(f"{_INDENT}{CHECK if success else CROSS} {message}")

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        del success, seconds, outputs
        self._write(message)

    def _write(self, line: str) -> None:
        print(line, file=self.stream, flush=True)


class JsonReporter:
    """One JSON object per line, for a program following the run."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        self.emit(
            event="start",
            operation="eegfeat run",
            subjects=list(labels),
            total_subjects=len(labels),
            output_root=str(output_root),
        )

    def recording_start(self, label: str, position: int, total: int) -> None:
        del position, total
        self.emit(event="subject_start", subject=label)

    def step(self, label: str, step: str, current: int, total: int) -> None:
        self.emit(
            event="progress",
            subject=label,
            step=step,
            current=current,
            total=total,
            pct=round(100 * current / total) if total else 0,
        )

    def recording_done(self, label: str, success: bool, message: str) -> None:
        self.emit(event="log", level="info" if success else "error", message=message, subject=label)
        self.emit(event="subject_done", subject=label, success=success)

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        self.emit(event="log", level="info" if success else "warning", message=message)
        self.emit(
            event="complete", success=success, duration=round(seconds, 3), outputs=list(outputs)
        )

    def error(self, code: str, message: str) -> None:
        """A problem that stopped the run before or instead of processing."""
        self.emit(event="error", code=code, message=message)

    def emit(self, **event: Any) -> None:
        """Write one event."""
        print(json.dumps(event), file=self.stream, flush=True)
