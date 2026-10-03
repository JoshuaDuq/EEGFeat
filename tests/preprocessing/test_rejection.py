import numpy as np

from eegtable.preprocessing.config import AutoRejectSettings, FixedEpochSettings
from eegtable.preprocessing.epochs import make_epochs
from eegtable.preprocessing.events import resolve_events
from eegtable.preprocessing.rejection import apply_rejection, fit_rejection


def test_autoreject_minimum(raw):
    settings = FixedEpochSettings(2)
    epochs = make_epochs(raw, resolve_events(raw, settings), settings)
    model = fit_rejection(epochs, AutoRejectSettings((1,), (0.5,), 2), 0, epochs.times[-1])
    actual, log = apply_rejection(epochs, model)
    expected = model.model.transform(epochs.copy(), reject_log=log)
    np.testing.assert_allclose(actual.get_data(), expected.get_data())
