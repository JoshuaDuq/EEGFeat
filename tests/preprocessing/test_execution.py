import mne
import numpy as np
import pytest

from eegfeat.preprocessing.config import (
    FixedEpochSettings,
    InputSettings,
    OutputSettings,
    PreprocessingConfig,
    ProcessingSettings,
    WorkflowSettings,
    read_yaml,
)


def config_for(raw, tmp_path):
    path = tmp_path / "source_raw.fif"
    raw.save(path, fmt="double")
    return PreprocessingConfig(
        InputSettings(path),
        OutputSettings(tmp_path / "out", "subject"),
        ProcessingSettings(FixedEpochSettings(2)),
        WorkflowSettings(raw_review="disabled"),
    )


def test_step_has_no_hidden_parents(raw, tmp_path):
    from eegfeat.preprocessing import open_workflow, run_step

    workflow = open_workflow(config_for(raw, tmp_path))
    with pytest.raises(ValueError, match="load"):
        run_step(workflow, "prepare")
    assert not workflow.workspace.exists()


def test_sequential_matches_memory(raw, tmp_path):
    from eegfeat.preprocessing import open_workflow, preprocess, read_checkpoint, run_until

    config = config_for(raw, tmp_path)
    expected = preprocess(raw, config.processing)
    workflow = open_workflow(config)
    outcome = run_until(workflow, "baseline")
    assert outcome.state == "completed"
    actual = read_checkpoint(workflow, "crop-epochs").state.epochs
    np.testing.assert_allclose(actual.get_data(), expected.epochs.get_data(), atol=1e-15)


def test_tampered_payload_fails(raw, tmp_path):
    from eegfeat.preprocessing import open_workflow, read_checkpoint, run_step

    workflow = open_workflow(config_for(raw, tmp_path))
    result = run_step(workflow, "load")
    payload = result.path / "data_raw.fif"
    with payload.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        read_checkpoint(workflow, "load")


def test_export_failure_is_not_completed(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, list_steps, open_workflow, run_until

    workflow = open_workflow(config_for(raw, tmp_path))

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(execution, "write_result", fail)
    with pytest.raises(OSError, match="injected"):
        run_until(workflow)
    assert list_steps(workflow)[-1].state != "completed"


def test_headless_review(raw, tmp_path):
    import json
    from dataclasses import replace

    from eegfeat.preprocessing import open_workflow, run_until
    from eegfeat.preprocessing.review import save_review

    config = config_for(raw, tmp_path)
    workflow = open_workflow(replace(config, workflow=WorkflowSettings(raw_review="required")))
    pending = run_until(workflow, "review-raw")
    assert pending.state == "needs-review"
    template = read_yaml(pending.steps[-1].path)
    template.update(bads=[], spans=[])
    decision = tmp_path / "decision.yaml"
    decision.write_text(json.dumps(template))
    save_review(workflow, "raw", decision)
    assert run_until(workflow, "epoch").state == "completed"


def test_load_checkpoint_from_brainvision(tmp_path):
    from eegfeat.preprocessing import open_workflow, read_checkpoint, run_until

    pybv = pytest.importorskip("pybv")
    rng = np.random.default_rng(7)
    channels = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    data = rng.normal(scale=2e-5, size=(len(channels), 2500))
    # The pybv default: 0.1 µV resolution is a calibration FIF cannot store exactly.
    pybv.write_brainvision(
        data=data,
        sfreq=250.0,
        ch_names=channels,
        fname_base="source",
        folder_out=tmp_path,
        resolution=1e-7,
        unit="V",
        fmt="binary_float32",
    )
    config = PreprocessingConfig(
        InputSettings(tmp_path / "source.vhdr"),
        OutputSettings(tmp_path / "out", "subject"),
        ProcessingSettings(FixedEpochSettings(2)),
        WorkflowSettings(raw_review="disabled"),
    )
    outcome = run_until(open_workflow(config), "load")
    assert outcome.state == "completed"
    loaded = read_checkpoint(open_workflow(config), "load").state.raw
    source = mne.io.read_raw(tmp_path / "source.vhdr", preload=True)
    np.testing.assert_array_equal(loaded.get_data(), source.get_data())


def test_step_hashes_the_source_once(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, open_workflow, run_step

    calls = []
    original = execution.fingerprint
    monkeypatch.setattr(execution, "fingerprint", lambda raw: calls.append(1) or original(raw))
    run_step(open_workflow(config_for(raw, tmp_path)), "load")
    assert len(calls) == 1


def test_run_until_loads_and_hashes_the_source_once(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, open_workflow, run_until

    counts = {"load": 0, "hash": 0}
    load, digest = execution.load_source, execution.fingerprint

    def counting_load(workflow):
        counts["load"] += 1
        return load(workflow)

    def counting_hash(raw):
        counts["hash"] += 1
        return digest(raw)

    monkeypatch.setattr(execution, "load_source", counting_load)
    monkeypatch.setattr(execution, "fingerprint", counting_hash)
    assert run_until(open_workflow(config_for(raw, tmp_path))).state == "completed"
    assert counts == {"load": 1, "hash": 1}


def test_second_run_keeps_pending_edits(raw, tmp_path):
    from dataclasses import replace

    from eegfeat.preprocessing import open_workflow, run_until

    config = config_for(raw, tmp_path)
    workflow = open_workflow(replace(config, workflow=WorkflowSettings(raw_review="required")))
    pending = run_until(workflow).steps[-1].path
    pending.write_text(pending.read_text().replace("bads: null", "bads: [C3]"))
    assert run_until(workflow).state == "needs-review"
    assert "bads: [C3]" in pending.read_text()


def test_run_loads_the_source_once(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, open_workflow, run_until

    calls = []
    original = execution.load_source
    monkeypatch.setattr(execution, "load_source", lambda w: calls.append(w) or original(w))
    assert run_until(open_workflow(config_for(raw, tmp_path)), "epoch").state == "completed"
    assert len(calls) == 1


def test_read_checkpoint_does_not_reload_the_source(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, open_workflow, read_checkpoint, run_until

    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "events")

    def refuse(w):
        raise AssertionError("source reloaded")

    monkeypatch.setattr(execution, "load_source", refuse)
    assert read_checkpoint(workflow, "events").state.events is not None


def test_missing_bundle_is_republished(raw, tmp_path):
    from eegfeat.preprocessing import open_workflow, run_until

    workflow = open_workflow(config_for(raw, tmp_path))
    assert run_until(workflow).state == "completed"
    for path in workflow.config.output.directory.glob("subject_*"):
        path.unlink()
    assert run_until(workflow).state == "completed"
    assert (workflow.config.output.directory / "subject_epo.fif").exists()


def test_parent_checkpoints_are_deserialized_once_per_step(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, open_workflow, run_until

    loads = []
    original = execution.load_checkpoint
    monkeypatch.setattr(
        execution,
        "load_checkpoint",
        lambda path, id: loads.append(path.parent.name) or original(path, id),
    )
    assert run_until(open_workflow(config_for(raw, tmp_path)), "events").state == "completed"
    assert loads == ["load", "prepare"]


def test_rerun_of_a_finished_recording_deserializes_nothing(raw, tmp_path, monkeypatch):
    from eegfeat.preprocessing import execution, open_workflow, run_until

    workflow = open_workflow(config_for(raw, tmp_path))
    assert run_until(workflow).state == "completed"

    def refuse(path, id):
        raise AssertionError(f"{path.parent.name} deserialized")

    monkeypatch.setattr(execution, "load_checkpoint", refuse)
    assert run_until(workflow).state == "completed"


def test_missing_payload_names_the_reset(raw, tmp_path):
    import shutil

    from eegfeat.preprocessing import open_workflow, run_until

    workflow = open_workflow(config_for(raw, tmp_path))
    run_until(workflow, "prepare")
    shutil.rmtree(workflow.workspace / "prepare")
    with pytest.raises(ValueError, match="prepare: checkpoint payload missing; reset CONFIG"):
        run_until(workflow, "events")
