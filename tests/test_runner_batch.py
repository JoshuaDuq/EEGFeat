"""Running a recipe over a folder of recordings."""

import hashlib
import json
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from eegfeat.io import read_table
from eegfeat.runner import RunError, check, load_recipe, run
from eegfeat.runner.progress import JsonReporter
from synthetic import save_epochs

POWER = '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["global"]\n'
ITPC = (
    '[trials]\nby = "event"\n\n'
    '[[features]]\nmeasure = "itpc"\nbands = ["alpha"]\nspatial = ["global"]\n'
)


def _recipe(tmp_path: Path, body: str, output: str = "out"):
    path = tmp_path / "recipe.toml"
    path.write_text(f'[inputs]\nroot = "data"\n\n[output]\nroot = "{output}"\n\n{body}')
    return load_recipe(path)


def _two_recordings(tmp_path: Path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif")


def _features_path(tmp_path: Path, subject: str) -> Path:
    return tmp_path / f"out/{subject}/eeg/{subject}_task-rest_features.tsv"


# --- outputs ------------------------------------------------------------------


def test_each_recording_gets_its_own_table_in_a_mirrored_tree(tmp_path) -> None:
    _two_recordings(tmp_path)

    result = run(_recipe(tmp_path, POWER))

    assert result.ok
    for subject in ("sub-01", "sub-02"):
        assert read_table(_features_path(tmp_path, subject)).n_rows == 12


def test_feature_rows_carry_each_epochs_event_and_metadata(tmp_path) -> None:
    _two_recordings(tmp_path)

    run(_recipe(tmp_path, POWER))

    frame = pd.read_csv(_features_path(tmp_path, "sub-01"), sep="\t")
    assert list(frame.columns[:4]) == ["epoch", "selection", "event", "rating"]
    assert frame["event"].tolist()[:4] == ["left", "right", "left", "right"]


def test_epoch_metadata_can_be_left_out(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)
    path = recipe.path
    path.write_text(
        path.read_text().replace('root = "out"', 'root = "out"\nepoch_metadata = false')
    )

    run(load_recipe(path))

    frame = pd.read_csv(_features_path(tmp_path, "sub-01"), sep="\t")
    assert "rating" not in frame.columns


def test_cross_trial_measures_are_written_to_their_own_table(tmp_path) -> None:
    _two_recordings(tmp_path)

    run(_recipe(tmp_path, ITPC))

    table = read_table(tmp_path / "out/sub-01/eeg/sub-01_task-rest_crosstrial.tsv")
    assert table.row_labels == ("left", "right")
    assert not _features_path(tmp_path, "sub-01").exists()


def test_sidecar_records_where_the_features_came_from(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)

    run(recipe)

    sidecar = json.loads(_features_path(tmp_path, "sub-01").with_suffix(".json").read_text())
    provenance = sidecar["provenance"]
    assert provenance["input"] == str(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    assert provenance["recipe_sha256"] == hashlib.sha256(recipe.text.encode()).hexdigest()
    assert provenance["channels"] == ["Fz", "F3", "F4", "Cz", "Pz"]


def test_channels_marked_bad_are_left_out_by_default(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif", bads=["Fz"])

    run(_recipe(tmp_path, POWER))

    sidecar = json.loads(_features_path(tmp_path, "sub-01").with_suffix(".json").read_text())
    assert sidecar["provenance"]["channels"] == ["F3", "F4", "Cz", "Pz"]


def test_run_log_records_every_recording(tmp_path) -> None:
    _two_recordings(tmp_path)
    recipe = _recipe(tmp_path, POWER)

    run(recipe)

    log = json.loads((tmp_path / "out/eegfeat_run.json").read_text())
    assert log["recipe_text"] == recipe.text
    assert [(r["label"], r["success"]) for r in log["recordings"]] == [
        ("sub-01_task-rest", True),
        ("sub-02_task-rest", True),
    ]


# --- failures -----------------------------------------------------------------


def test_a_failing_recording_does_not_stop_the_others(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    save_epochs(
        tmp_path / "data/sub-02/eeg/sub-02_task-rest_epo.fif", channels=["Fz", "F4", "Cz", "Pz"]
    )
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n',
    )

    result = run(recipe)

    first, second = result.recordings
    assert first.success and not second.success
    assert second.error is not None and "F3" in second.error
    assert not result.ok
    assert _features_path(tmp_path, "sub-01").exists()
    assert not _features_path(tmp_path, "sub-02").exists()
    log = json.loads((tmp_path / "out/eegfeat_run.json").read_text())
    assert "F3" in log["recordings"][1]["error"]


def test_existing_results_are_not_overwritten_by_default(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))

    with pytest.raises(RunError, match="overwrite"):
        run(_recipe(tmp_path, POWER))


def test_overwrite_replaces_results_and_clears_stale_ones(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER + "\n" + ITPC))
    stale = tmp_path / "out/sub-01/eeg/sub-01_task-rest_crosstrial.tsv"
    assert stale.exists()

    result = run(_recipe(tmp_path, POWER), overwrite=True)

    assert result.ok
    assert _features_path(tmp_path, "sub-01").exists()
    assert not stale.exists()
    assert not stale.with_suffix(".json").exists()


def test_failed_overwrite_preserves_the_previous_complete_result(tmp_path, monkeypatch) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    recipe = _recipe(tmp_path, POWER)
    first = run(recipe)
    assert first.ok
    values = _features_path(tmp_path, "sub-01")
    existing = {
        path: path.read_bytes()
        for path in (
            values,
            values.with_suffix(".json"),
            values.with_name(f"{values.stem}_coverage.tsv"),
        )
    }

    import eegfeat.runner.batch as batch

    def fail(*args, **kwargs):
        raise RuntimeError("intentional replacement failure")

    monkeypatch.setattr(batch, "compute_features", fail)
    result = run(recipe, overwrite=True)

    assert not result.ok
    assert all(path.read_bytes() == content for path, content in existing.items())


def test_hidden_files_are_not_recordings(tmp_path) -> None:
    # macOS writes "._" AppleDouble companions beside files on external drives.
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif")
    (tmp_path / "data/sub-01/eeg/._sub-01_task-rest_epo.fif").write_bytes(b"\0" * 4096)
    (tmp_path / "data/.cache").mkdir()
    save_epochs(tmp_path / "data/.cache/sub-09_task-rest_epo.fif")

    result = run(_recipe(tmp_path, POWER))

    assert [r.label for r in result.recordings] == ["sub-01_task-rest"]


def test_no_matching_file_is_an_error(tmp_path) -> None:
    (tmp_path / "data").mkdir()

    with pytest.raises(RunError, match="no files match"):
        run(_recipe(tmp_path, POWER))


def test_missing_input_root_is_an_error(tmp_path) -> None:
    with pytest.raises(RunError, match="data"):
        run(_recipe(tmp_path, POWER))


def test_inputs_that_would_share_outputs_are_rejected(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01_epo.fif")
    save_epochs(tmp_path / "data/sub-01-epo.fif")
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        '[inputs]\nroot = "data"\npattern = "*epo.fif"\n\n[output]\nroot = "out"\n\n' + POWER
    )

    with pytest.raises(RunError, match="sub-01"):
        run(load_recipe(recipe))


# --- progress -----------------------------------------------------------------


def test_progress_events_follow_the_tui_protocol(tmp_path) -> None:
    _two_recordings(tmp_path)
    stream = StringIO()

    run(_recipe(tmp_path, POWER), reporter=JsonReporter(stream))

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    one, two = "sub-01_task-rest", "sub-02_task-rest"
    assert [(e["event"], e.get("subject"), e.get("step")) for e in events] == [
        ("start", None, None),
        ("subject_start", one, None),
        ("progress", one, "read"),
        ("progress", one, "integrated_band_power"),
        ("progress", one, "write"),
        ("log", one, None),
        ("subject_done", one, None),
        ("subject_start", two, None),
        ("progress", two, "read"),
        ("progress", two, "integrated_band_power"),
        ("progress", two, "write"),
        ("log", two, None),
        ("subject_done", two, None),
        ("log", None, None),
        ("complete", None, None),
    ]
    assert events[0]["subjects"] == [one, two] and events[0]["total_subjects"] == 2
    steps = [(e["current"], e["total"]) for e in events if e["event"] == "progress"][:3]
    assert steps == [(1, 3), (2, 3), (3, 3)]
    assert [e["success"] for e in events if e["event"] == "subject_done"] == [True, True]
    assert events[-1]["success"] is True


def test_a_failed_recording_is_reported_as_such(tmp_path) -> None:
    save_epochs(tmp_path / "data/sub-01/eeg/sub-01_task-rest_epo.fif", channels=["Fz", "Cz"])
    stream = StringIO()
    recipe = _recipe(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )

    run(recipe, reporter=JsonReporter(stream))

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    done = next(e for e in events if e["event"] == "subject_done")
    error = next(e for e in events if e["event"] == "log" and e.get("level") == "error")
    assert done["success"] is False
    assert "F3" in error["message"]
    assert events[-1]["success"] is False


# --- check --------------------------------------------------------------------


def test_check_trials_the_first_recording_without_writing(tmp_path) -> None:
    _two_recordings(tmp_path)

    report = check(_recipe(tmp_path, POWER + "\n" + ITPC))

    assert [r.label for r in report.recordings] == ["sub-01_task-rest", "sub-02_task-rest"]
    assert report.trial.label == "sub-01_task-rest"
    assert report.features.epochs is not None and report.features.epochs.n_rows == 12
    assert report.features.crosstrial is not None
    assert not (tmp_path / "out").exists()


def test_check_lists_results_already_on_disk(tmp_path) -> None:
    _two_recordings(tmp_path)
    run(_recipe(tmp_path, POWER))

    report = check(_recipe(tmp_path, POWER))

    assert _features_path(tmp_path, "sub-01") in report.existing
