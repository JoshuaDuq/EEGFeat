"""Common spatial patterns on the motor task.

Movement against rest is a variance contrast in the mu and beta bands, which is
exactly what CSP maximizes. Cross-fitted with run-disjoint folds, the component
favouring rest scores rest trials higher and the one favouring movement scores
movement higher, in every subject.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import eegfeat as ef
import eegfeat.model as efm
from validation.loaders import Recording

MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
WINDOW = ef.Window("window", 0.5, 3.5)

DATASET = "eegbci"


def _band_limited(recording: Recording, band: ef.Band) -> ef.Signal:
    signal = ef.BandSignal.from_epochs(recording.epochs, band, recording=recording.name)
    inside = (signal.times >= WINDOW.tmin) & (signal.times <= WINDOW.tmax)
    return ef.Signal.from_arrays(
        data=np.real(signal.analytic)[:, :, inside],
        times=signal.times[inside],
        ch_names=signal.ch_names,
        sfreq=signal.sfreq,
        row_ids=signal.row_ids,
    )


@pytest.mark.validates(
    "csp_features",
    kind="decoding",
    claim="Cross-fitted CSP components separate movement from rest in every subject",
    criterion="first component AUC below 0.5, second above; best above 0.6 in every subject",
)
@pytest.mark.parametrize("band", [MU, BETA], ids=lambda band: band.name)
def test_cross_fitted_components_separate_movement_from_rest(
    eegbci_recordings: list[Recording], band: ef.Band, record: Callable[[str], None]
) -> None:
    best = []
    for recording in eegbci_recordings:
        metadata = recording.metadata
        moving = metadata["moving"].to_numpy()
        folds = efm.within_subject_folds(
            metadata["subject"].to_numpy(dtype=object),
            metadata["run"].to_numpy(dtype=object),
            inner_splits=3,
        )
        table = ef.csp_features(
            _band_limited(recording, band), moving, folds=folds, n_components=4, regularization=0.1
        )
        assert np.isfinite(table.values).all()
        aucs = np.array([roc_auc_score(moving, column) for column in table.values.T])
        # Components alternate: the first favours the first class (rest), the second the other.
        assert aucs[0] < 0.5 < aucs[1], (recording.name, aucs.round(2))
        best.append(np.max(np.abs(aucs - 0.5)) + 0.5)
    record(
        f"{band.name}: best component AUC {min(best):.2f} to {max(best):.2f}, mean "
        f"{float(np.mean(best)):.2f} over {len(best)} subjects"
    )
    assert np.min(best) > 0.6, np.round(best, 2)
    assert np.mean(best) > 0.75, np.round(best, 2)


@pytest.mark.validates(
    "csp_features",
    kind="formula",
    claim="CSP filters and patterns are mutually inverse",
    criterion="filters times patterns transposed is the identity to 1e-8",
)
def test_filters_and_patterns_are_mutually_inverse(eegbci_recordings: list[Recording]) -> None:
    recording = eegbci_recordings[0]
    signal = _band_limited(recording, BETA)
    fitted = ef.CommonSpatialPattern.fit(
        signal, recording.metadata["moving"].to_numpy(), n_components=4, regularization=0.1
    )
    # Each pattern is the column of the inverse filter matrix that maps back to the sensors,
    # so a filter applied to its own pattern gives one and to any other pattern gives zero.
    products = fitted.filters @ fitted.patterns.T
    np.testing.assert_allclose(products, np.eye(4), atol=1e-8)
