import mne
import numpy as np
import pytest

from eegtable.preprocessing.config import (
    EventEpochSettings,
    EventSettings,
    ReferenceSettings,
    ThresholdSettings,
)
from eegtable.preprocessing.epochs import make_epochs, reference_epochs
from eegtable.preprocessing.events import resolve_events


def test_threshold_rejection_matches_mne(raw):
    from eegtable.preprocessing.rejection import reject_epochs

    settings = EventEpochSettings(
        EventSettings("stim", {"stimulus": 1}, stim_channel="STI", shortest_event=1), -0.2, 0.8
    )
    events = resolve_events(raw, settings.events)
    epochs = make_epochs(raw, events, settings)
    actual = reject_epochs(epochs, ThresholdSettings(reject={"eeg": 150e-6}))
    expected = mne.Epochs(
        raw,
        events.events,
        events.event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=None,
        proj=False,
        preload=True,
        reject={"eeg": 150e-6},
        picks=raw.ch_names,
    )
    np.testing.assert_array_equal(actual.events, expected.events)
    np.testing.assert_allclose(actual.get_data(), expected.get_data())
    assert actual.drop_log == expected.drop_log


def test_reference_preserves_auxiliary(raw):
    actual = reference_epochs(raw, ReferenceSettings("average"))
    np.testing.assert_allclose(actual.get_data(picks="eeg").sum(axis=0), 0, atol=1e-18)
    np.testing.assert_array_equal(
        actual.get_data(picks=["VEOG", "ECG", "STI"]), raw.get_data(picks=["VEOG", "ECG", "STI"])
    )


def test_missing_acquisition_reference_cannot_be_restored_after_rereferencing(raw):
    referenced = raw.copy().set_eeg_reference("average", projection=False)
    with pytest.raises(ValueError, match="before.*re-referencing"):
        reference_epochs(referenced, ReferenceSettings("average", ("Cz",)))


@pytest.mark.parametrize("artifact_reference", ["average", ("Cz",)])
def test_acquisition_reference_is_restored_before_artifact_reference(raw, artifact_reference):
    from eegtable.preprocessing.checks import validate_processing
    from eegtable.preprocessing.config import (
        ArtifactSettings,
        ChannelSettings,
        FixedEpochSettings,
        ICASettings,
        ProcessingSettings,
    )
    from eegtable.preprocessing.pipeline import StageData, execute_numeric

    settings = ProcessingSettings(
        FixedEpochSettings(2),
        channels=ChannelSettings(montage="colin27_1020"),
        artifact=ArtifactSettings("ica", ICASettings(), artifact_reference),
        reference=ReferenceSettings("average", ("Cz",)),
    )
    expected = mne.add_reference_channels(raw, ["Cz"])
    expected.set_montage("colin27_1020")
    expected.set_eeg_reference(
        artifact_reference if isinstance(artifact_reference, str) else list(artifact_reference),
        projection=False,
    )
    state = execute_numeric("artifact-reference", StageData(raw), settings)
    assert state.raw.ch_names == expected.ch_names
    np.testing.assert_allclose(state.raw.get_data(), expected.get_data(), atol=1e-18)
    np.testing.assert_allclose(
        state.raw.info["chs"][-1]["loc"][:3], expected.info["chs"][-1]["loc"][:3]
    )
    assert state.provenance["restored_reference_channels"] == ["Cz"]
    validate_processing(raw, settings)
    final = execute_numeric(
        "reference", StageData(None, epochs=state.raw, provenance=state.provenance), settings
    )
    assert final.epochs.ch_names.count("Cz") == 1
    expected.set_eeg_reference("average", projection=False)
    np.testing.assert_allclose(final.epochs.get_data(), expected.get_data(), atol=1e-18)


@pytest.mark.parametrize(
    "artifact_reference,error", [("unknown", "missing channels"), ("VEOG", "good.*EEG")]
)
def test_preflight_rejects_invalid_artifact_reference(raw, artifact_reference, error):
    from eegtable.preprocessing.checks import validate_processing
    from eegtable.preprocessing.config import (
        ArtifactSettings,
        FixedEpochSettings,
        ICASettings,
        ProcessingSettings,
    )

    settings = ProcessingSettings(
        FixedEpochSettings(2),
        artifact=ArtifactSettings("ica", ICASettings(), (artifact_reference,)),
        reference=ReferenceSettings("average", ("Cz",)),
    )
    with pytest.raises(ValueError, match=error):
        validate_processing(raw, settings)


def test_final_reference_locates_restored_electrodes(raw):
    from eegtable.preprocessing.config import (
        ChannelSettings,
        FixedEpochSettings,
        ProcessingSettings,
    )
    from eegtable.preprocessing.pipeline import StageData, execute_numeric

    settings = ProcessingSettings(
        FixedEpochSettings(2),
        channels=ChannelSettings(montage="colin27_1020"),
        reference=ReferenceSettings("average", ("Cz",)),
    )
    state = execute_numeric("reference", StageData(None, epochs=raw), settings)
    assert np.isfinite(state.epochs.info["chs"][-1]["loc"][:3]).all()
    assert state.provenance["restored_reference_channels"] == ["Cz"]


def test_preflight_rejects_restoration_from_custom_referenced_source(raw):
    from eegtable.preprocessing.checks import validate_processing
    from eegtable.preprocessing.config import FixedEpochSettings, ProcessingSettings

    settings = ProcessingSettings(
        FixedEpochSettings(2), reference=ReferenceSettings("average", ("Cz",))
    )
    with pytest.raises(ValueError, match="source already has a custom reference"):
        validate_processing(raw.copy().set_eeg_reference("average", projection=False), settings)
