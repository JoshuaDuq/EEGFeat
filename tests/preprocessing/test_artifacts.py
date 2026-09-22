import numpy as np

from eegfeat.preprocessing.artifacts import (
    apply_artifact,
    fit_eog_regression,
    reference_artifact_data,
    review_artifact,
)
from eegfeat.preprocessing.config import RegressionSettings


def test_regression_matches_mne_and_attenuates(raw):
    from scipy.ndimage import gaussian_filter1d

    predictor = np.zeros(raw.n_times)
    predictor[500::750] = 1e-3
    predictor = gaussian_filter1d(predictor, 10)
    raw.apply_function(lambda values: predictor, picks=["VEOG"])
    raw.apply_function(lambda values: values + 4 * predictor, picks=["Fp1"])
    referenced = reference_artifact_data(raw, "average")
    model = fit_eog_regression(referenced, RegressionSettings(("VEOG",)))
    actual = apply_artifact(
        referenced, review_artifact(model, {"fit_id": model.fit_id, "apply": True})
    )
    expected = model.model.apply(referenced.copy())
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-18)
    np.testing.assert_array_equal(
        actual.get_data(picks=["VEOG"]), referenced.get_data(picks=["VEOG"])
    )
    assert np.std(actual.get_data(picks=["Fp1"])) < np.std(referenced.get_data(picks=["Fp1"]))


def test_ssp_matches_mne_and_excludes_existing_projectors(raw):
    import mne
    from scipy.ndimage import gaussian_filter1d

    from eegfeat.preprocessing.artifacts import fit_ssp
    from eegfeat.preprocessing.config import SSPSettings

    blink = np.zeros(raw.n_times)
    blink[600::900] = 3e-4
    blink = gaussian_filter1d(blink, 12)
    raw.apply_function(lambda values: values + blink, picks=["VEOG"])
    raw.apply_function(lambda values: values + 0.3 * blink, picks=["Fp1", "Fp2"])
    raw.add_proj(mne.compute_proj_raw(raw, n_eeg=1, verbose=False), remove_existing=True)
    model = fit_ssp(raw, SSPSettings(n_eeg=1, eog_channels=("VEOG",), reject=None))
    assert len(model.model) == 1
    reviewed = review_artifact(model, {"fit_id": model.fit_id, "include": [0]})
    actual = apply_artifact(raw, reviewed)
    expected = raw.copy().add_proj(model.model).apply_proj()
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-18)
    assert np.std(actual.get_data(picks=["Fp1"])) < np.std(raw.get_data(picks=["Fp1"]))
    assert (
        apply_artifact(raw, review_artifact(model, {"fit_id": model.fit_id, "include": []}))
        .get_data()
        .tolist()
        == raw.get_data().tolist()
    )
