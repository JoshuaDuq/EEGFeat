import numpy as np
import pytest

from eegfeat.preprocessing.config import FilterSettings, FixedEpochSettings, ResamplingSettings
from eegfeat.preprocessing.epochs import make_epochs
from eegfeat.preprocessing.events import resolve_events
from eegfeat.preprocessing.sampling import crop_epochs, resample_epochs


@pytest.mark.parametrize("target,padding", [(200.0, 0.5), (125.0, 0.52)])
def test_resampling_matches_direct_mne(raw, target, padding):
    settings = FixedEpochSettings(2, padding=padding)
    epochs = make_epochs(raw, resolve_events(raw, settings), settings)
    actual = resample_epochs(epochs, ResamplingSettings(target, padding), FilterSettings())
    expected = epochs.copy().resample(
        target, method="polyphase", window=("kaiser", 5.0), pad="reflect", n_jobs=1
    )
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-18)
    np.testing.assert_array_equal(actual.events, epochs.events)
    assert np.isclose(actual.times, 0).any()
    cropped = crop_epochs(actual, 0, 1.996)
    assert cropped.times[0] == 0
    assert abs(cropped.times[-1] - 1.996) * target <= 0.5 + 1e-6


def test_misaligned_padding_fails(raw):
    settings = FixedEpochSettings(2, padding=0.5)
    epochs = make_epochs(raw, resolve_events(raw, settings), settings)
    with pytest.raises(ValueError, match="origin"):
        resample_epochs(epochs, ResamplingSettings(125.0, 0.5), FilterSettings())
