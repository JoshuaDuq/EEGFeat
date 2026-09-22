import json

import mne
import numpy as np
import pandas as pd
import pytest

from eegfeat.preprocessing import open_workflow, preprocess, read_checkpoint, run_until
from eegfeat.preprocessing.config import (
    ArtifactSettings,
    AutoRejectSettings,
    EventEpochSettings,
    EventSettings,
    FilterSettings,
    FixedEpochSettings,
    ICASettings,
    ProcessingSettings,
    read_yaml,
)
from eegfeat.preprocessing.review import save_review

from .test_execution import config_for


def test_no_stage_matches_direct_mne(raw):
    events = EventSettings("stim", {"stimulus": 1}, stim_channel="STI", shortest_event=1)
    result = preprocess(raw, ProcessingSettings(EventEpochSettings(events, -0.2, 0.8)))
    expected = mne.Epochs(
        raw,
        mne.find_events(raw, stim_channel="STI", shortest_event=1),
        {"stimulus": 1},
        tmin=-0.2,
        tmax=0.8,
        baseline=None,
        proj=False,
        preload=True,
        picks=raw.ch_names,
    )
    np.testing.assert_array_equal(result.epochs.events, expected.events)
    np.testing.assert_array_equal(result.epochs.get_data(), expected.get_data())
    assert result.epochs.drop_log == expected.drop_log
    # Disabled stages perform no operation at all.
    assert result.provenance["stages"] == [
        "load",
        "prepare",
        "events",
        "annotate",
        "epoch",
        "crop-epochs",
    ]
    assert result.events["retained"].all()
    np.testing.assert_array_equal(result.events["original_row"], np.arange(5))
    np.testing.assert_array_equal(result.events["original_sample"], result.events["event_sample"])


def test_checkpointed_ica_review_and_autoreject_export(mixture, tmp_path):
    config = config_for(mixture, tmp_path)
    processing = ProcessingSettings(
        FixedEpochSettings(2),
        filter=FilterSettings(l_freq=1.0),
        artifact=ArtifactSettings("ica", ICASettings(n_components=4, eog_channels=("VEOG",))),
        rejection=AutoRejectSettings((1,), (0.5,), 2),
    )
    workflow = open_workflow(
        config.__class__(config.input, config.output, processing, config.workflow)
    )
    pending = run_until(workflow)
    assert pending.state == "needs-review"
    template = read_yaml(pending.steps[-1].path)
    assert template["exclude"] is None
    template["exclude"] = [0]
    decision = tmp_path / "artifact.yaml"
    decision.write_text(json.dumps(template))
    save_review(workflow, "artifact", decision)
    assert run_until(workflow).state == "completed"

    # Application reuses the saved fit exactly; nothing is refitted.
    model = read_checkpoint(workflow, "fit-artifact").state.artifact.model
    uncorrected = read_checkpoint(workflow, "epoch").state.epochs
    corrected = read_checkpoint(workflow, "apply-artifact").state.epochs
    expected = model.copy().apply(uncorrected.copy(), exclude=[0])
    np.testing.assert_allclose(corrected.get_data(), expected.get_data(), rtol=0, atol=1e-18)

    bundle = config.output.directory
    manifest = json.loads((bundle / "subject_preprocessing.json").read_text())
    assert manifest["provenance"]["artifact_decision"]["exclude"] == [0]
    assert manifest["provenance"]["artifact"]["fit_id"] == template["fit_id"]
    assert set(manifest["provenance"]["repair"]) == {"original_rows", "channels", "bad_epochs"}
    repairs = pd.read_csv(bundle / "subject_repairs.tsv", sep="\t")
    assert set(repairs.columns) == {"original_row", "channel", "label"}
    assert set(repairs["channel"]) <= set(mixture.ch_names[:8])
    epochs = mne.read_epochs(bundle / "subject_epo.fif", preload=True, proj=False)
    assert "preprocessing=subject_preprocessing.json" in epochs.info["description"]


def test_candidate_and_review_spans_use_acquisition_time(raw):
    from eegfeat.preprocessing.config import (
        AmplitudeSettings,
        AnnotationSettings,
        CropSettings,
    )
    from eegfeat.preprocessing.pipeline import StageData, execute_numeric

    # A flat second at 12 s of the file, seen through a crop that removes the first 4 s.
    flat = slice(int(12 * raw.info["sfreq"]), int(13 * raw.info["sfreq"]))
    data = raw.get_data()
    data[:8, flat] = 0.0
    flat_raw = mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
    settings = ProcessingSettings(
        FixedEpochSettings(2),
        crop=CropSettings(4.0, 28.0),
        annotations=AnnotationSettings(amplitude=AmplitudeSettings(flat={"eeg": 1e-9})),
    )
    state = StageData(flat_raw)
    for stage in ("load", "prepare", "events", "crop-raw", "annotate"):
        state = execute_numeric(stage, state, settings)
    # Noise can be flat for a single sample; the planted second is the only long span.
    (candidate,) = [span for span in state.candidates["spans"] if span["duration"] > 0.5]
    assert candidate["onset"] == pytest.approx(12.0, abs=1 / raw.info["sfreq"])
    decision = {"bads": [], "spans": [{"onset": 20.0, "duration": 1.0, "description": "BAD_x"}]}
    reviewed = execute_numeric("review-raw", state, settings, decision).raw
    (onset,) = reviewed.annotations.onset[reviewed.annotations.description == "BAD_x"]
    assert onset == pytest.approx(20.0 + raw.first_time)


def test_missing_review_decision_is_refused(raw):
    from eegfeat.preprocessing.config import RegressionSettings

    artifact = ArtifactSettings("regression", RegressionSettings(("VEOG",)), "average")
    with pytest.raises(ValueError, match="review-artifact"):
        preprocess(raw, ProcessingSettings(FixedEpochSettings(2), artifact=artifact))
