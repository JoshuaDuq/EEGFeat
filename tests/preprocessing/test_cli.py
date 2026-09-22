import io
import json

import numpy as np
import pandas as pd
import pytest

from eegfeat.runner import load_recipe, run
from eegfeat.runner.cli import main


def test_cli_init_and_catalog(tmp_path, capsys):
    config = tmp_path / "preprocessing.yaml"
    assert main(["preprocess", "init", str(config), "--mode", "resting"]) == 0
    assert main(["preprocess", "steps", str(config)]) == 0
    output = capsys.readouterr().out
    assert "review-artifact" in output
    assert "export" in output
    assert not (tmp_path / "preprocessed").exists()


@pytest.mark.parametrize(
    "command,flags",
    [
        ("run", ["--until", "--recording", "--n-jobs", "--overwrite", "--progress-json"]),
        ("check", ["--recording"]),
        ("status", ["--recording", "--verify"]),
        ("step", ["--recording", "--n-jobs", "--overwrite"]),
        ("next", ["--recording", "--n-jobs", "--overwrite"]),
        ("inspect", ["--recording", "--report"]),
        ("review", ["--recording", "--decisions", "--suggested"]),
        ("reset", ["--from", "--recording"]),
        ("init", ["--mode"]),
    ],
)
def test_every_flag_is_documented(command, flags, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["preprocess", command, "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert all(flag in text for flag in flags)
    assert "recipe" in text.lower()


def _write_config(raw, tmp_path, raw_review):
    raw.save(tmp_path / "recording_raw.fif", fmt="double", verbose=False)
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "input: {path: recording_raw.fif}\noutput: {directory: preprocessed, name: recording}\n"
        f"workflow: {{raw_review: {raw_review}}}\nepochs: {{kind: fixed, duration: 2.0}}\n"
    )
    return config


def _write_cohort(raw, tmp_path, raw_review, *labels):
    for label in labels:
        (tmp_path / "raw" / label).mkdir(parents=True)
        raw.save(tmp_path / "raw" / label / f"{label}_raw.fif", fmt="double", verbose=False)
    config = tmp_path / "study.yaml"
    config.write_text(
        "input: {root: raw, pattern: '**/*_raw.fif'}\noutput: {directory: preprocessed}\n"
        f"workflow: {{raw_review: {raw_review}}}\nepochs: {{kind: fixed, duration: 2.0}}\n"
    )
    return config


def test_cli_return_codes_and_review_gate(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "required")
    assert main(["preprocess", "check", str(config)]) == 0
    assert not (tmp_path / "preprocessed").exists()
    assert main(["preprocess", "run", str(config)]) == 3
    assert f"eegfeat preprocess review {config} raw" in capsys.readouterr().out
    assert main(["preprocess", "status", str(config)]) == 0
    assert "review-raw: needs-review" in capsys.readouterr().out
    assert main(["preprocess", "step", str(config), "export"]) == 2
    assert main(["preprocess", "step", str(config), "notch"]) == 2
    assert main(["preprocess", "reset", str(config), "--from", "events"]) == 0


def test_export_feeds_feature_runner_and_hides_checkpoints(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config)]) == 0
    assert "eegfeat init" in capsys.readouterr().out
    assert main(["preprocess", "run", str(config)]) == 0
    assert main(["preprocess", "status", str(config), "--verify"]) == 0
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        '[inputs]\nroot = "preprocessed"\npattern = "**/*_epo.fif"\npicks = "eeg"\n'
        '[output]\nroot = "features"\n[bands]\nalpha = [8.0, 13.0]\n'
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nspatial = ["global"]\n'
    )
    result = run(load_recipe(recipe))
    tables = sorted(path.name for path in (tmp_path / "features").rglob("*_features.tsv"))
    assert tables == ["recording_features.tsv"]
    assert len(result.recordings) == 1
    features = pd.read_csv(tmp_path / "features" / "recording_features.tsv", sep="\t")
    assert len(features) == 15
    assert np.isfinite(features.select_dtypes("number").to_numpy()).all()


def test_cohort_run_mirrors_tree_and_isolates_failures(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    (tmp_path / "raw" / "sub-00").mkdir()
    (tmp_path / "raw" / "sub-00" / "sub-00_raw.fif").write_bytes(b"not a recording")
    assert main(["preprocess", "run", str(config)]) == 1
    out = capsys.readouterr().out
    assert "[1/3] sub-00" in out and "✗" in out
    assert "Opening raw data file" not in out
    assert (tmp_path / "preprocessed" / "sub-01" / "sub-01_epo.fif").exists()
    assert (tmp_path / "preprocessed" / "sub-02" / "sub-02_epo.fif").exists()
    assert "eegfeat init" not in out
    assert main(["preprocess", "run", str(config), "--recording", "sub-02"]) == 0
    assert "eegfeat init" in capsys.readouterr().out


def test_cohort_gate_prints_complete_commands_and_status_summary(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    out = capsys.readouterr().out
    assert f"eegfeat preprocess review {config} --recording sub-01 raw" in out
    assert main(["preprocess", "status", str(config)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("sub-01") and "awaiting review-raw" in line for line in lines)
    assert main(["preprocess", "status", str(config), "--recording", "sub-02"]) == 0
    assert "review-raw: needs-review" in capsys.readouterr().out


def test_cohort_commands_need_a_recording(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    assert main(["preprocess", "step", str(config), "load"]) == 2
    assert "--recording" in capsys.readouterr().err
    assert main(["preprocess", "run", str(config), "--recording", "sub-03"]) == 2
    assert "sub-01, sub-02" in capsys.readouterr().err


def test_cohort_flagless_review_walks_pending_recordings(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    for label in ("sub-01", "sub-02"):
        pending = tmp_path / "preprocessed" / label / ".preprocessing" / label / "decisions"
        pending = pending / "review-raw.pending.yaml"
        text = (
            pending.read_text()
            .replace("bads: null", "bads: []")
            .replace("spans: null", "spans: []")
        )
        pending.write_text(text)
    decisions = tmp_path / "one.yaml"
    decisions.write_text(pending.read_text())
    assert main(["preprocess", "review", str(config), "raw", "--decisions", str(decisions)]) == 2
    assert "--recording" in capsys.readouterr().err
    assert main(["preprocess", "review", str(config), "raw"]) == 0
    assert main(["preprocess", "run", str(config)]) == 0


def test_progress_json_uses_runner_events(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01")
    assert main(["preprocess", "run", str(config), "--progress-json"]) == 0
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events[0] == "start" and events[-1] == "complete"
    assert {"subject_start", "progress", "subject_done"} <= set(events)


def test_init_refuses_to_overwrite(tmp_path, capsys):
    config = tmp_path / "preprocessing.yaml"
    assert main(["preprocess", "init", str(config)]) == 0
    assert main(["preprocess", "init", str(config)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_check_applies_the_same_source_checks_as_run(raw, tmp_path, capsys):
    raw.save(tmp_path / "rec_epo.fif", fmt="double", verbose=False)
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "input: {path: rec_epo.fif}\noutput: {directory: ., name: rec}\n"
        "workflow: {raw_review: disabled}\nepochs: {kind: fixed, duration: 2.0}\n"
    )
    assert main(["preprocess", "check", str(config)]) == 1
    assert "collides" in capsys.readouterr().out


def test_stale_checkpoint_is_reported_as_a_command(raw, tmp_path, capsys):
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config)]) == 0
    config.write_text(config.read_text().replace("duration: 2.0", "duration: 1.0"))
    assert main(["preprocess", "run", str(config)]) == 1
    out = capsys.readouterr().out
    assert f"eegfeat preprocess reset {config} --from events" in out
    assert "CONFIG" not in out
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegfeat preprocess reset {config} --from events" in capsys.readouterr().out


def test_cohort_status_suggests_the_next_command(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegfeat preprocess run {config}" in capsys.readouterr().out
    assert main(["preprocess", "run", str(config)]) == 3
    capsys.readouterr()
    assert main(["preprocess", "status", str(config)]) == 0
    assert f"Next: eegfeat preprocess review {config} raw" in capsys.readouterr().out


def test_stage_arguments_are_checked_before_any_recording_runs(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "disabled", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config), "--until", "nope"]) == 2
    assert "nope" in capsys.readouterr().err
    assert not (tmp_path / "preprocessed").exists()


@pytest.mark.parametrize("error,traced", [(RuntimeError, True), (ValueError, False)])
def test_only_unexpected_failures_keep_a_traceback(
    raw, tmp_path, capsys, monkeypatch, error, traced
):
    from eegfeat.preprocessing import execution

    config = _write_cohort(raw, tmp_path, "disabled", "sub-01")

    def boom(*args, **kwargs):
        raise error("boom")

    monkeypatch.setattr(execution, "run_until", boom)
    assert main(["preprocess", "run", str(config)]) == 1
    captured = capsys.readouterr()
    assert "✗ boom" in captured.out
    assert ("Traceback" in captured.err) is traced


def test_run_summary_points_at_the_pending_review(raw, tmp_path, capsys):
    config = _write_cohort(raw, tmp_path, "required", "sub-01", "sub-02")
    assert main(["preprocess", "run", str(config)]) == 3
    assert f"Next: eegfeat preprocess review {config} raw" in capsys.readouterr().out


def test_stage_reporter_shows_progress_only_on_a_terminal():
    from eegfeat.preprocessing.cli import StageReporter

    class Tty(io.StringIO):
        def isatty(self):
            return True

    tty = Tty()
    reporter = StageReporter(tty)
    reporter.step("sub-01", "filter", 10, 26)
    reporter.recording_done("sub-01", True, "done")
    assert "· filter (10/26)" in tty.getvalue()
    assert tty.getvalue().endswith("✓ done\n")
    plain = io.StringIO()
    StageReporter(plain).step("sub-01", "filter", 10, 26)
    assert plain.getvalue() == ""


def test_inspect_report_fits_the_checkpoint(raw, tmp_path, monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda uri: opened.append(uri))
    config = _write_config(raw, tmp_path, "disabled")
    assert main(["preprocess", "run", str(config), "--until", "crop-epochs"]) == 0
    assert main(["preprocess", "inspect", str(config), "load", "--report"]) == 0
    assert main(["preprocess", "inspect", str(config), "crop-epochs", "--report"]) == 0
    assert len(opened) == 2 and all(uri.endswith("-inspection.html") for uri in opened)
