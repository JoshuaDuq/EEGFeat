import numpy as np

from eegfeat.preprocessing.config import AutoRejectSettings, FixedEpochSettings
from eegfeat.preprocessing.epochs import make_epochs
from eegfeat.preprocessing.events import resolve_events
from eegfeat.preprocessing.rejection import apply_rejection, fit_rejection


def test_autoreject_minimum(raw):
    settings = FixedEpochSettings(2)
    epochs = make_epochs(raw, resolve_events(raw, settings), settings)
    model = fit_rejection(epochs, AutoRejectSettings((1,), (0.5,), 2), 0, epochs.times[-1])
    actual, log = apply_rejection(epochs, model)
    expected = model.model.transform(epochs.copy(), reject_log=log)
    np.testing.assert_allclose(actual.get_data(), expected.get_data())
