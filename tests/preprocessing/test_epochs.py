import mne
import numpy as np

from eegfeat.preprocessing.config import (
    EventEpochSettings,
    EventSettings,
    ReferenceSettings,
    ThresholdSettings,
)
from eegfeat.preprocessing.epochs import make_epochs, reference_epochs
from eegfeat.preprocessing.events import resolve_events


def test_threshold_rejection_matches_mne(raw):
    from eegfeat.preprocessing.rejection import reject_epochs

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
