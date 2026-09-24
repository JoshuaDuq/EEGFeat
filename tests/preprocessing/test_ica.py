import pytest

from eegfeat.preprocessing.config import ICASettings


def test_exact_ica_review_and_no_refit(mixture):
    import numpy as np

    from eegfeat.preprocessing.artifacts import apply_artifact, review_artifact
    from eegfeat.preprocessing.ica import fit_ica

    model = fit_ica(mixture, ICASettings(n_components=4))
    reviewed = review_artifact(model, {"fit_id": model.fit_id, "exclude": []})
    result = apply_artifact(mixture, reviewed)
    np.testing.assert_array_equal(result.get_data(), mixture.get_data())
    with pytest.raises(ValueError, match="fit_id"):
        review_artifact(model, {"fit_id": "stale", "exclude": []})
    with pytest.raises(ValueError, match="outside"):
        review_artifact(model, {"fit_id": model.fit_id, "exclude": [4]})


def test_rank_follows_reference(mixture):
    from eegfeat.preprocessing.artifacts import reference_artifact_data
    from eegfeat.preprocessing.ica import fit_ica

    referenced = reference_artifact_data(mixture, "average")
    assert fit_ica(referenced, ICASettings()).evidence["rank"] == 7
    with pytest.raises(ValueError, match="usable rank"):
        fit_ica(referenced, ICASettings(n_components=8))


def test_reviewed_exclusion_removes_injected_artifact(raw):
    import mne
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    from eegfeat.preprocessing.artifacts import apply_artifact, review_artifact
    from eegfeat.preprocessing.ica import fit_ica

    # Seven neural sources plus one sparse artifact source in eight channels.
    rng = np.random.default_rng(7)
    neural = rng.normal(size=(8, 7)) @ (rng.laplace(size=(7, raw.n_times)) * 1e-6)
    artifact = np.zeros(raw.n_times)
    artifact[rng.choice(raw.n_times, 40, replace=False)] = 1.0
    artifact = gaussian_filter1d(artifact, 8) * 4e-5
    weights = np.linspace(1.0, 0.2, 8)
    data = np.vstack([neural + np.outer(weights, artifact), raw.get_data()[8:]])
    contaminated = mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
    model = fit_ica(contaminated, ICASettings())
    sources = model.model.get_sources(contaminated).get_data()
    # Choosing by correlation with the injected source is for the test only.
    scores = [abs(np.corrcoef(source, artifact)[0, 1]) for source in sources]
    chosen = int(np.argmax(scores))
    assert scores[chosen] > 0.9
    cleaned = apply_artifact(
        contaminated, review_artifact(model, {"fit_id": model.fit_id, "exclude": [chosen]})
    )
    fp1 = cleaned.get_data(picks=["Fp1"])[0]
    assert abs(np.corrcoef(fp1, artifact)[0, 1]) < 0.2
    assert np.corrcoef(fp1, neural[0])[0, 1] > 0.95


def test_detector_suggestions_are_evidence_only(mixture):
    import numpy as np

    from eegfeat.preprocessing.artifacts import reference_artifact_data
    from eegfeat.preprocessing.ica import fit_ica

    referenced = reference_artifact_data(mixture, "average")
    model = fit_ica(referenced, ICASettings(eog_channels=("VEOG",), ecg_channel="ECG"))
    evidence = model.evidence
    assert set(evidence["suggested"]) == {"VEOG", "ECG"}
    union = sorted({i for found in evidence["suggested"].values() for i in found})
    assert evidence["suggested_exclude"] == union
    assert all(isinstance(index, int) for index in union)
    # Suggestions are MNE's own verdicts on the same training copy.
    training = referenced.copy().filter(1.0, None)
    expected, _ = model.model.find_bads_eog(training, ch_name="VEOG")
    assert evidence["suggested"]["VEOG"] == [int(i) for i in expected]
    assert "iclabel" not in evidence
    np.testing.assert_array_equal(model.model.exclude, [])


@pytest.mark.parametrize("method", ["infomax", "picard"])
def test_ica_methods_follow_settings(mixture, method):
    from eegfeat.preprocessing.ica import fit_ica

    if method == "picard":
        pytest.importorskip("picard")
    model = fit_ica(mixture, ICASettings(method=method, n_components=4, max_iter=2000))
    assert model.model.method == method
    assert model.model.fit_params["extended"] is True
    if method == "picard":
        assert model.model.fit_params["ortho"] is False


def test_iclabel_requires_extended_infomax_and_average_reference():
    from eegfeat.preprocessing.config import ArtifactSettings, ICLabelSettings

    with pytest.raises(ValueError, match="infomax or picard"):
        ICASettings(iclabel=ICLabelSettings())
    with pytest.raises(ValueError, match="reference: average"):
        ArtifactSettings("ica", ICASettings(method="infomax", iclabel=ICLabelSettings()), None)
    with pytest.raises(ValueError, match="keep"):
        ICLabelSettings(keep=("cortex",))


def test_iclabel_labels_every_component(mixture):
    pytest.importorskip("mne_icalabel")
    from eegfeat.preprocessing.artifacts import reference_artifact_data
    from eegfeat.preprocessing.config import FilterSettings, ICLabelSettings
    from eegfeat.preprocessing.ica import fit_ica
    from eegfeat.preprocessing.raw import filter_raw

    # ICLabel's training band, 1-100 Hz: fit_ica adds the 1 Hz high-pass.
    lowpassed = filter_raw(mixture, FilterSettings(h_freq=100.0))
    referenced = reference_artifact_data(lowpassed, "average")
    settings = ICASettings(
        method="infomax", n_components=4, max_iter=2000, iclabel=ICLabelSettings(threshold=0.5)
    )
    evidence = fit_ica(referenced, settings).evidence["iclabel"]
    assert len(evidence["labels"]) == 4
    assert evidence["probabilities"].shape == (4, 7)
    assert set(evidence["labels"]) <= set(evidence["classes"])
    # The suggestion rule: winning label outside keep with confidence at or above threshold.
    expected = [
        index
        for index, (label, row) in enumerate(
            zip(evidence["labels"], evidence["probabilities"], strict=True)
        )
        if label not in ("brain", "other") and row.max() >= 0.5
    ]
    assert evidence["suggested"] == expected


def test_iclabel_requires_its_training_band(raw):
    import pytest

    from eegfeat.preprocessing.checks import validate_processing
    from eegfeat.preprocessing.config import (
        ArtifactSettings,
        FilterSettings,
        FixedEpochSettings,
        ICASettings,
        ICLabelSettings,
        ProcessingSettings,
    )

    ica = ICASettings(method="infomax", iclabel=ICLabelSettings())
    artifact = ArtifactSettings("ica", ica, "average")
    # ICLabel was trained on 1-100 Hz data; a 250 Hz recording without a low-pass is 125 Hz wide.
    with pytest.raises(ValueError, match="100 Hz"):
        validate_processing(raw, ProcessingSettings(FixedEpochSettings(2), artifact=artifact))
    validate_processing(
        raw,
        ProcessingSettings(
            FixedEpochSettings(2), artifact=artifact, filter=FilterSettings(h_freq=40)
        ),
    )
