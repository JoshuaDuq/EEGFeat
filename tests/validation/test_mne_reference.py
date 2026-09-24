"""eegfeat against MNE's own implementations, on the datasets MNE's documentation uses.

Where eegfeat computes a quantity MNE also computes, the two have to agree: exactly
when the definitions coincide, and in outcome when they differ by design. Peaks are
checked against ``Evoked.get_peak`` on ERP CORE and orthogonalized envelope correlation
against mne-connectivity on the motor data; then two of MNE's documented pipelines are
rerun with eegfeat in place of their own code, the Sleep-EDF staging tutorial and the
motor CSP decoding example.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import eegfeat as ef
from validation.loaders import Recording, load_erp_core

N1 = ef.Window("n1", 0.1, 0.25)
ALPHA = ef.Band("alpha", 8.0, 13.0)
RESTING = ef.Window("window", 0.5, 3.5)
# The bands of MNE's sleep-staging tutorial, as it defines them.
SLEEP_BANDS = (
    ef.Band("delta", 0.5, 4.5),
    ef.Band("theta", 4.5, 8.5),
    ef.Band("alpha", 8.5, 11.5),
    ef.Band("sigma", 11.5, 15.5),
    ef.Band("beta", 15.5, 30.0),
)


@pytest.mark.validates(
    "peak_latency",
    "peak_amplitude",
    kind="estimator",
    dataset="erp_core",
    claim="Peak latency and amplitude are what MNE's Evoked.get_peak reports",
    criterion="identical on every channel for negative, positive and absolute peaks",
)
def test_peaks_are_what_evoked_get_peak_reports(record: Callable[[str], None]) -> None:
    evoked = load_erp_core("stimulus").epochs.average()
    times = evoked.times
    # get_peak rounds its bounds to the sample grid and eegfeat does not, so the window
    # is laid on samples for the two to search the same stretch.
    first, last = np.searchsorted(times, [N1.tmin, N1.tmax])
    window = ef.Window(N1.name, times[first], times[last])
    signal = ef.Signal.from_arrays(
        data=evoked.data[np.newaxis],
        times=times,
        ch_names=tuple(evoked.ch_names),
        sfreq=evoked.info["sfreq"],
        row_ids=(("sub-001", 0, "average"),),
    )
    for mode, polarity in (("neg", "negative"), ("pos", "positive"), ("abs", "absolute")):
        options = {"windows": [window], "polarity": polarity, "include_global": False}
        latency = ef.peak_latency([signal], **options).values[0]
        amplitude = ef.peak_amplitude([signal], **options).values[0]
        reference = np.array(
            [
                evoked.copy()
                .pick([name])
                .get_peak(
                    tmin=window.tmin,
                    tmax=window.tmax,
                    mode=mode,
                    return_amplitude=True,
                    strict=False,
                )[1:]
                for name in evoked.ch_names
            ]
        )
        np.testing.assert_array_equal(latency, reference[:, 0])
        np.testing.assert_array_equal(amplitude, reference[:, 1])
    record(f"{len(evoked.ch_names)} channels, three polarities, all identical")


@pytest.mark.validates(
    "envelope_correlation",
    kind="estimator",
    dataset="eegbci",
    claim="Orthogonalized envelope correlation is the Fisher-z mean of mne-connectivity's "
    "per-trial estimate",
    criterion="relative error below 1e-9 on every pair",
)
def test_orthogonalized_envelope_correlation_matches_mne_connectivity(
    eegbci_recordings: list[Recording], record: Callable[[str], None]
) -> None:
    mne_connectivity = pytest.importorskip("mne_connectivity")
    recording = eegbci_recordings[0]
    resting = recording.epochs[recording.metadata["moving"].to_numpy() == 0]
    alpha = ef.BandSignal.from_epochs(resting, ALPHA, recording=recording.name)
    table = ef.envelope_correlation([alpha], windows=[RESTING])

    inside = (alpha.times >= RESTING.tmin) & (alpha.times <= RESTING.tmax)
    per_trial = mne_connectivity.envelope_correlation(alpha.analytic[:, :, inside])
    trials = np.asarray(per_trial.get_data(output="dense"))[..., 0]
    expected = np.tanh(np.mean(np.arctanh(np.clip(trials, -0.999999, 0.999999)), axis=0))
    index = {name: position for position, name in enumerate(alpha.ch_names)}
    reference = np.array([expected[index[m.nodes[0]], index[m.nodes[1]]] for m in table.meta])

    error = np.max(np.abs(table.values[0] - reference) / np.abs(reference))
    record(f"{reference.size} pairs over {len(trials)} trials, maximum relative error {error:.1e}")
    assert error < 1e-9


def _tutorial_features(recording: Recording) -> np.ndarray:
    # MNE's eeg_power_band, as the tutorial writes it: each PSD normalized to sum to one,
    # then the plain mean of the bins inside each band.
    spectrum = recording.epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    return np.concatenate(
        [
            psds[:, :, (freqs >= band.fmin) & (freqs < band.fmax)].mean(axis=-1)
            for band in SLEEP_BANDS
        ],
        axis=1,
    )


def _eegfeat_features(recording: Recording) -> np.ndarray:
    spectrum = recording.epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0, normalization="full")
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=recording.name, estimator_parameters={"normalization": "full"}
    )
    table = ef.mean_psd(spectra, bands=SLEEP_BANDS, include_global=False)
    # Relative power per derivation, as the tutorial's features are.
    relative = np.empty_like(table.values)
    for space in {meta.space for meta in table.meta}:
        columns = [k for k, meta in enumerate(table.meta) if meta.space == space]
        relative[:, columns] = table.values[:, columns] / table.values[:, columns].sum(
            axis=1, keepdims=True
        )
    return relative


@pytest.mark.validates(
    "mean_psd",
    kind="decoding",
    dataset="sleep",
    claim="Band power reproduces the staging accuracy of MNE's sleep-classification tutorial",
    criterion="random forest trained on one night scores the other within 0.03 of the "
    "tutorial's own features",
)
def test_band_power_reproduces_the_sleep_staging_tutorial(
    sleep_recordings: list[Recording], record: Callable[[str], None]
) -> None:
    from sklearn.ensemble import RandomForestClassifier

    train, test = sleep_recordings
    y_train, y_test = train.metadata["stage"].to_numpy(), test.metadata["stage"].to_numpy()

    def accuracy(features: Callable[[Recording], np.ndarray]) -> float:
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(features(train), y_train)
        return float(np.mean(forest.predict(features(test)) == y_test))

    tutorial, ours = accuracy(_tutorial_features), accuracy(_eegfeat_features)
    record(f"accuracy on {test.name}: tutorial {tutorial:.3f}, eegfeat {ours:.3f}")
    assert abs(ours - tutorial) < 0.03


@pytest.mark.validates(
    "CommonSpatialPattern",
    kind="decoding",
    dataset="eegbci",
    claim="In MNE's CSP decoding pipeline, eegfeat's CSP separates movement from rest about as "
    "well as mne.decoding.CSP",
    criterion="mean accuracy over 20 subjects within 0.05 of MNE's, on the same splits",
)
def test_csp_decodes_like_mne_in_its_decoding_example(
    eegbci_recordings: list[Recording], record: Callable[[str], None]
) -> None:
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    from sklearn.pipeline import Pipeline

    ours, theirs = [], []
    for recording in eegbci_recordings:
        # The example's own choices: 7-30 Hz, one second from 1 s after the cue, CSP of
        # four components into LDA, ten 80/20 shuffle splits.
        epochs = recording.epochs.copy().filter(7.0, 30.0).crop(1.0, 2.0)
        moving = recording.metadata["moving"].to_numpy()
        splits = ShuffleSplit(10, test_size=0.2, random_state=42)
        mne_pipeline = Pipeline(
            [
                ("CSP", CSP(n_components=4, reg=None, log=True)),
                ("LDA", LinearDiscriminantAnalysis()),
            ]
        )
        theirs.append(
            cross_val_score(mne_pipeline, epochs.get_data(picks="eeg"), moving, cv=splits).mean()
        )
        signal = ef.Signal.from_epochs(epochs, recording=recording.name)
        scores = []
        for train, test in splits.split(moving):
            csp = ef.CommonSpatialPattern.fit(signal, moving, rows=train, n_components=4)
            lda = LinearDiscriminantAnalysis().fit(csp.transform(signal, rows=train), moving[train])
            scores.append(lda.score(csp.transform(signal, rows=test), moving[test]))
        ours.append(float(np.mean(scores)))

    # eegfeat's features are log variance relative to the components' sum (Ramoser 2000),
    # MNE's the log of absolute power; the overall power drop that movement causes is in
    # MNE's features only, which is most of what separates the two means.
    record(
        f"mean accuracy over {len(ours)} subjects: MNE {np.mean(theirs):.3f}, "
        f"eegfeat {np.mean(ours):.3f}"
    )
    assert abs(np.mean(ours) - np.mean(theirs)) < 0.05
