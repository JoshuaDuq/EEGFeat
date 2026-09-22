from dataclasses import replace

import pytest

from eegfeat.preprocessing import list_steps, open_workflow, run_step, run_until
from eegfeat.preprocessing.config import FilterSettings

from .test_execution import config_for


def test_baseline_change_does_not_invalidate_upstream(raw, tmp_path):
    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "crop-epochs")
    config = workflow.config
    settings = replace(
        config.processing, epochs=replace(config.processing.epochs, baseline=(0, 0.2))
    )
    changed = open_workflow(replace(config, processing=settings))
    statuses = {item.stage: item.state for item in list_steps(changed)}
    assert statuses["epoch"] == statuses["crop-epochs"] == "completed"
    assert statuses["baseline"] == "pending"


def test_source_change_requires_explicit_reset(raw, tmp_path):
    workflow = open_workflow(config_for(raw, tmp_path))
    run_step(workflow, "load")
    raw.apply_function(lambda values: values * 2, picks=["C3"])
    raw.save(workflow.config.input.path, fmt="double", overwrite=True)
    with pytest.raises(ValueError, match="reset"):
        run_step(workflow, "load")


def test_disabled_filter_change_does_not_invalidate_events(raw, tmp_path):
    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "epoch")
    changed = open_workflow(
        replace(
            workflow.config,
            processing=replace(workflow.config.processing, filter=FilterSettings(h_freq=40)),
        )
    )
    statuses = {item.stage: item.state for item in list_steps(changed)}
    assert statuses["events"] == "completed"
    assert statuses["epoch"] == "stale"


def test_reset_retires_descendants_and_keeps_payloads(raw, tmp_path):
    from eegfeat.preprocessing import reset_from

    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "epoch")
    payloads = sorted(workflow.workspace.glob("*/*/manifest.json"))
    assert reset_from(workflow, "events")[:2] == ("events", "crop-raw")
    statuses = {item.stage: item.state for item in list_steps(workflow)}
    assert statuses["prepare"] == "completed"
    assert statuses["events"] == statuses["epoch"] == "pending"
    assert sorted(workflow.workspace.glob("*/*/manifest.json")) == payloads
    assert run_until(workflow, "epoch").state == "completed"


def test_second_writer_is_refused(raw, tmp_path):
    import filelock

    workflow = open_workflow(config_for(raw, tmp_path))
    run_step(workflow, "load")
    with (
        filelock.FileLock(workflow.workspace / ".writer.lock", timeout=0),
        pytest.raises(filelock.Timeout),
    ):
        run_step(workflow, "prepare")


def test_hidden_files_beside_payloads_are_ignored(raw, tmp_path):
    from eegfeat.preprocessing import read_checkpoint

    workflow = open_workflow(config_for(raw, tmp_path))
    result = run_step(workflow, "load")
    # Finder drops .DS_Store into any browsed folder; exFAT drives add ._ AppleDouble files.
    (result.path / ".DS_Store").write_bytes(b"\0")
    (result.path / "._data_raw.fif").write_bytes(b"\0")
    read_checkpoint(workflow, "load")


def test_epoch_checkpoints_hold_no_raw(raw, tmp_path):
    from eegfeat.preprocessing import read_checkpoint

    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "crop-epochs")
    assert read_checkpoint(workflow, "events").state.raw is not None
    for stage in ("epoch", "crop-epochs"):
        checkpoint = read_checkpoint(workflow, stage)
        assert checkpoint.state.raw is None
        assert not (checkpoint.path / "data_raw.fif").exists()
