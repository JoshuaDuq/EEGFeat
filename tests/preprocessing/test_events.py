import numpy as np

from eegfeat.preprocessing.config import EventSettings, FixedEpochSettings


def test_original_sample_identity(raw):
    from eegfeat.preprocessing.events import resolve_events

    result = resolve_events(
        raw,
        EventSettings("stim", {"stimulus": 1}, stim_channel="STI", shortest_event=1, delay=0.004),
    )
    np.testing.assert_array_equal(
        result.events[:, 0], raw.first_samp + np.arange(1250, 6251, 1250) - 1
    )
    np.testing.assert_array_equal(result.original_samples, result.events[:, 0] + 1)


def test_fixed_sample_geometry(raw):
    from eegfeat.preprocessing.epochs import make_epochs
    from eegfeat.preprocessing.events import resolve_events

    settings = FixedEpochSettings(duration=2)
    events = resolve_events(raw, settings)
    result = make_epochs(raw, events, settings)
    assert result.get_data().shape == (15, 11, 500)
