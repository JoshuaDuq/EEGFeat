"""The ``eegfeat`` command line: exit codes, output, and the progress stream."""

import json
import subprocess
import sys
from pathlib import Path

from eegfeat.runner import load_recipe
from eegfeat.runner.cli import main
from synthetic import save_epochs

POWER = '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["global"]\n'
FRONTAL_ROI = (
    '[rois]\nfront = ["Fz", "F3"]\n\n'
    '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["rois"]\n'
)

# A dB scale needs a baseline window: the recipe loads, and the entry fails when computed.
DB_WITHOUT_BASELINE = (
    '\n[[features]]\nmeasure = "mean_psd"\nbands = ["alpha"]\nspatial = ["global"]\n'
    'normalize = "db"\n'
)


def _recipe(tmp_path: Path, body: str = POWER) -> Path:
    path = tmp_path / "recipe.toml"
    path.write_text(f'[inputs]\nroot = "data"\n\n[output]\nroot = "out"\n\n{body}')
    return path


def _recording(tmp_path: Path, subject: str, **kwargs: object) -> None:
    save_epochs(tmp_path / f"data/{subject}/eeg/{subject}_task-rest_epo.fif", **kwargs)


def test_run_writes_results_and_exits_zero(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["run", str(_recipe(tmp_path))])

    out = capsys.readouterr().out
    assert code == 0
    assert "sub-01_task-rest" in out
    assert "12 epochs" in out and "1 features" in out
    assert (tmp_path / "out/sub-01/eeg/sub-01_task-rest_features.tsv").exists()


def test_progress_json_prints_one_json_event_per_line(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["run", str(_recipe(tmp_path)), "--progress-json"])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert events[0]["event"] == "start" and events[-1]["event"] == "complete"


def test_run_exits_one_when_a_recording_fails(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", channels=["Fz", "Cz"])

    code = main(["run", str(_recipe(tmp_path, FRONTAL_ROI))])

    assert code == 1
    assert "sub-02_task-rest" in capsys.readouterr().out


def test_invalid_recipe_exits_two_and_lists_its_problems(tmp_path, capsys) -> None:
    code = main(["run", str(_recipe(tmp_path, '[[features]]\nmeasure = "band_powr"\n'))])

    assert code == 2
    assert "band_powr" in capsys.readouterr().err


def test_invalid_recipe_is_an_error_event_in_the_progress_stream(tmp_path, capsys) -> None:
    body = '[[features]]\nmeasure = "band_powr"\n'

    code = main(["run", str(_recipe(tmp_path, body)), "--progress-json"])

    (event,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 2
    assert event["event"] == "error" and "band_powr" in event["message"]


def test_existing_results_exit_two_unless_overwritten(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    recipe = str(_recipe(tmp_path))
    assert main(["run", recipe]) == 0

    assert main(["run", recipe]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main(["run", recipe, "--overwrite"]) == 0


def test_check_exits_zero_and_writes_nothing(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path))])

    assert code == 0
    assert "sub-01_task-rest" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_check_exits_one_when_the_trial_recording_fails(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path, POWER + DB_WITHOUT_BASELINE))])

    assert code == 1
    assert "baseline" in capsys.readouterr().err


def test_check_finds_a_named_channel_that_a_later_recording_marks_bad(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["F3"])

    code = main(["check", str(_recipe(tmp_path, FRONTAL_ROI))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-02_task-rest" in out and "F3" in out and "marked bad" in out
    assert "Ready" not in out


def test_check_finds_a_named_channel_that_a_later_recording_lacks(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", channels=["Fz", "Cz"])

    code = main(["check", str(_recipe(tmp_path, FRONTAL_ROI))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-02_task-rest" in out and "F3" in out


def test_check_finds_an_asymmetry_channel_that_a_later_recording_marks_bad(
    tmp_path, capsys
) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["F4"])
    body = (
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\n'
        'spatial = ["channels"]\nasymmetry = [["F3", "F4"]]\n'
    )

    code = main(["check", str(_recipe(tmp_path, body))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-02_task-rest" in out and "F4" in out


def test_check_ignores_rois_that_no_entry_uses(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")
    _recording(tmp_path, "sub-02", bads=["F3"])

    code = main(["check", str(_recipe(tmp_path, '[rois]\nfront = ["Fz", "F3"]\n\n' + POWER))])

    assert code == 0
    assert "Ready" in capsys.readouterr().out


def test_a_failed_check_names_the_recipe_entry_it_came_from(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["check", str(_recipe(tmp_path, POWER + DB_WITHOUT_BASELINE))])

    err = capsys.readouterr().err
    assert code == 1
    assert "features[1] (mean_psd)" in err
    assert "requires a baseline" in err


def test_a_failed_recording_in_a_run_names_the_recipe_entry(tmp_path, capsys) -> None:
    _recording(tmp_path, "sub-01")

    code = main(["run", str(_recipe(tmp_path, POWER + DB_WITHOUT_BASELINE))])

    out = capsys.readouterr().out
    assert code == 1
    assert "features[1] (mean_psd)" in out
    assert "requires a baseline" in out


def test_init_writes_a_recipe_that_loads(tmp_path) -> None:
    path = tmp_path / "recipe.toml"

    assert main(["init", str(path)]) == 0
    assert load_recipe(path).features


def test_init_does_not_overwrite_an_existing_file(tmp_path, capsys) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text("# mine\n")

    assert main(["init", str(path)]) == 2
    assert path.read_text() == "# mine\n"


def test_the_package_runs_as_a_module() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "eegfeat", "--version"], capture_output=True, text=True
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("eegfeat ")


def test_check_reports_a_channel_the_first_recording_lacks_instead_of_computing_it(
    tmp_path, capsys
) -> None:
    # Surveying the cohort's channels is cheap and says why; computing a recording that
    # already fails the survey can only raise from inside a measure.
    _recording(tmp_path, "sub-01", bads=["F3"])
    _recording(tmp_path, "sub-02")

    code = main(["check", str(_recipe(tmp_path, FRONTAL_ROI))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sub-01_task-rest" in out and "F3" in out and "marked bad" in out
    assert "sub-02_task-rest" not in out
