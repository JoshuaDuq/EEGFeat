"""What a person and a program watching a run are told."""

import json
from io import StringIO
from pathlib import Path

from eegfeat.runner.progress import JsonReporter, TextReporter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_text_progress_estimates_the_time_left_from_the_pace_so_far() -> None:
    stream, clock = StringIO(), Clock()
    reporter = TextReporter(stream, clock=clock)
    reporter.start(["a", "b", "c", "d"], Path("out"))
    reporter.recording_start("a", 1, 4)
    clock.now = 120.0
    reporter.recording_done("a", True, "12 epochs")

    last = stream.getvalue().splitlines()[-1]
    assert "12 epochs" in last
    assert "1/4 done" in last and "2 min elapsed" in last and "about 6 min left" in last


def test_text_progress_names_a_recording_that_finishes_out_of_turn() -> None:
    # With workers, recordings finish in any order; a result line that does not follow
    # its own start line must say which recording it belongs to.
    stream, clock = StringIO(), Clock()
    reporter = TextReporter(stream, clock=clock)
    reporter.start(["a", "b"], Path("out"))
    reporter.recording_start("a", 1, 2)
    reporter.recording_start("b", 2, 2)
    clock.now = 10.0
    reporter.recording_done("a", True, "12 epochs")
    reporter.recording_done("b", False, "boom")

    a_line, b_line = stream.getvalue().splitlines()[-2:]
    assert "a · 12 epochs" in a_line
    assert "b · boom" in b_line and "done" not in b_line


def test_text_progress_keeps_the_plain_line_when_a_recording_follows_its_start() -> None:
    stream, clock = StringIO(), Clock()
    reporter = TextReporter(stream, clock=clock)
    reporter.start(["a"], Path("out"))
    reporter.recording_start("a", 1, 1)
    reporter.recording_done("a", True, "12 epochs")

    assert stream.getvalue().splitlines()[-1].strip().startswith("✓ 12 epochs")


def test_json_progress_carries_the_estimate_on_each_finished_recording() -> None:
    stream, clock = StringIO(), Clock()
    reporter = JsonReporter(stream, clock=clock)
    reporter.start(["a", "b"], Path("out"))
    clock.now = 30.0
    reporter.recording_done("a", True, "ok")

    done = [json.loads(line) for line in stream.getvalue().splitlines()][-1]
    assert done["event"] == "subject_done"
    assert done["elapsed"] == 30.0 and done["eta"] == 30.0


def test_durations_read_the_way_a_person_says_them() -> None:
    from eegfeat.runner.progress import human_duration

    assert [human_duration(s) for s in (0.34, 4.26, 45.0, 720.0, 7500.0)] == [
        "0.3 s",
        "4.3 s",
        "45 s",
        "12 min",
        "2 h 05 min",
    ]
