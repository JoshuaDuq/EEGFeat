"""Sleep-EDF (PhysioNet): the spectral signature of sleep depth.

Deep sleep is slow waves. Relative to wake, N3 carries more of its power below
4 Hz and less in the alpha and beta ranges, so every descriptor of where the
spectrum sits, from the edge frequency to Hjorth mobility, moves down. Two
subjects, first night each, thirty-second epochs scored by an expert.

The stage contrasts are read at Pz-Oz. At Fpz-Cz the wake epochs are dominated
by eye and movement activity, which puts as much power below 4 Hz as slow-wave
sleep does, so that derivation does not separate wake from N3 in either subject.
That is a property of the recordings, and the reason the checks name a channel.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import eegfeat as ef
from validation.loaders import Recording

DELTA = ef.Band("delta", 0.5, 4.0)
ALPHA = ef.Band("alpha", 8.0, 12.0)
BETA = ef.Band("beta", 15.0, 30.0)
TOTAL = ef.Band("total", 0.5, 30.0)
WHOLE = ef.Window("epoch", 0.0, 30.0)
NREM = ("N1", "N2", "N3")
OCCIPITAL = "Pz-Oz"

DATASET = "sleep"


@pytest.fixture(scope="module")
def spectra(sleep_recordings: list[Recording]) -> list[ef.Spectra]:
    out = []
    for recording in sleep_recordings:
        sfreq = recording.epochs.info["sfreq"]
        parameters = {
            "method": "welch",
            "fmin": TOTAL.fmin,
            "fmax": TOTAL.fmax,
            "n_fft": int(2 * sfreq),
            "n_per_seg": int(2 * sfreq),
            "n_overlap": int(sfreq),
        }
        spectrum = recording.epochs.compute_psd(**parameters)
        out.append(
            ef.Spectra.from_spectrum(
                spectrum, recording=recording.name, estimator_parameters=parameters
            )
        )
    return out


def _by_stage(table: ef.FeatureTable, recording: Recording) -> pd.DataFrame:
    frame = pd.DataFrame(table.values, columns=[m.space for m in table.meta])
    frame["stage"] = recording.metadata["stage"].to_numpy()
    return frame


def _wake_vs_deep_auc(frame: pd.DataFrame, channel: str) -> float:
    """Probability that a random N3 epoch scores higher than a random wake epoch."""
    kept = frame[frame["stage"].isin(["W", "N3"])]
    return float(roc_auc_score((kept["stage"] == "N3").astype(int), kept[channel]))


def _relative_power(spectra: ef.Spectra, band: ef.Band) -> ef.FeatureTable:
    power = ef.integrated_band_power(
        spectra, bands=[band, TOTAL], normalize="log10", include_global=False
    )
    return ef.band_ratio(power, band.name, TOTAL.name)


@pytest.mark.validates(
    "integrated_band_power",
    "band_ratio",
    kind="physiology",
    claim="Relative delta rises and relative alpha and beta fall from wake to N3 at Pz-Oz",
    criterion="wake-vs-N3 AUC above 0.8 (delta), below 0.2 (alpha), below 0.1 (beta)",
)
def test_deep_sleep_moves_power_into_slow_waves(
    sleep_recordings: list[Recording], spectra: list[ef.Spectra], record: Callable[[str], None]
) -> None:
    aucs: list[str] = []
    for recording, spectrum in zip(sleep_recordings, spectra, strict=True):
        delta = _by_stage(_relative_power(spectrum, DELTA), recording)
        alpha = _by_stage(_relative_power(spectrum, ALPHA), recording)
        beta = _by_stage(_relative_power(spectrum, BETA), recording)

        aucs.append(
            f"{recording.name}: delta {_wake_vs_deep_auc(delta, OCCIPITAL):.2f}, alpha "
            f"{_wake_vs_deep_auc(alpha, OCCIPITAL):.2f}, beta "
            f"{_wake_vs_deep_auc(beta, OCCIPITAL):.2f}"
        )
        assert _wake_vs_deep_auc(delta, OCCIPITAL) > 0.8, recording.name
        assert _wake_vs_deep_auc(alpha, OCCIPITAL) < 0.2, recording.name
        assert _wake_vs_deep_auc(beta, OCCIPITAL) < 0.1, recording.name
    record("AUC " + "; ".join(aucs))


@pytest.mark.validates(
    "integrated_band_power",
    "band_ratio",
    kind="physiology",
    claim="The slow-wave fraction increases monotonically N1, N2, N3",
    criterion="median ordering holds on both derivations, both subjects",
)
def test_slow_wave_fraction_grows_with_nrem_depth(
    sleep_recordings: list[Recording], spectra: list[ef.Spectra]
) -> None:
    for recording, spectrum in zip(sleep_recordings, spectra, strict=True):
        delta = _by_stage(_relative_power(spectrum, DELTA), recording)
        medians = delta.groupby("stage").median().loc[list(NREM)]
        for channel in medians.columns:
            assert medians[channel].is_monotonic_increasing, (recording.name, channel, medians)


@pytest.mark.validates(
    "spectral_edge",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_bandwidth",
    kind="physiology",
    claim="Edge, centroid, entropy and bandwidth are lower in N3 than in wake at Pz-Oz",
    criterion="medians ordered; wake-vs-N3 AUC below 0.25",
)
def test_spectral_descriptors_fall_in_deep_sleep(
    sleep_recordings: list[Recording], spectra: list[ef.Spectra]
) -> None:
    for recording, spectrum in zip(sleep_recordings, spectra, strict=True):
        descriptors = {
            "edge": ef.spectral_edge(spectrum, band=TOTAL, include_global=False),
            "centroid": ef.spectral_centroid(spectrum, band=TOTAL, include_global=False),
            "entropy": ef.spectral_entropy(spectrum, band=TOTAL, include_global=False),
            "bandwidth": ef.spectral_bandwidth(spectrum, band=TOTAL, include_global=False),
        }
        for name, table in descriptors.items():
            frame = _by_stage(table, recording)
            medians = frame.groupby("stage").median()[OCCIPITAL]
            assert medians["N3"] < medians["W"], (recording.name, name, medians)
            assert _wake_vs_deep_auc(frame, OCCIPITAL) < 0.25, (recording.name, name)


@pytest.mark.validates(
    "hjorth_mobility",
    kind="physiology",
    claim="Mobility falls wake, N2, N3 at Pz-Oz",
    criterion="median ordering holds; wake-vs-N3 AUC below 0.1",
)
def test_hjorth_mobility_tracks_the_dominant_frequency(
    sleep_recordings: list[Recording],
) -> None:
    for recording in sleep_recordings:
        signal = ef.Signal.from_epochs(recording.epochs, recording=recording.name)
        mobility = _by_stage(
            ef.hjorth_mobility([signal], windows=[WHOLE], include_global=False), recording
        )
        ordered = mobility.groupby("stage").median().loc[["W", "N2", "N3"], OCCIPITAL]
        assert ordered.is_monotonic_decreasing, (recording.name, ordered)
        assert _wake_vs_deep_auc(mobility, OCCIPITAL) < 0.1, recording.name


@pytest.mark.validates(
    "hjorth_complexity",
    kind="physiology",
    claim="Complexity is higher in N3 than in wake at Pz-Oz",
    criterion="wake-vs-N3 AUC above 0.8, both subjects",
)
def test_hjorth_complexity_rises_in_deep_sleep(sleep_recordings: list[Recording]) -> None:
    """Slow waves are far from a pure sine, so complexity, the ratio of the derivative's
    mobility to the signal's, is higher in N3 than in the mixed-frequency wake EEG."""
    for recording in sleep_recordings:
        signal = ef.Signal.from_epochs(recording.epochs, recording=recording.name)
        complexity = _by_stage(
            ef.hjorth_complexity([signal], windows=[WHOLE], include_global=False), recording
        )
        assert _wake_vs_deep_auc(complexity, OCCIPITAL) > 0.8, recording.name


@pytest.mark.validates(
    "hjorth_mobility",
    kind="formula",
    claim="Mobility is the documented formula in hertz",
    criterion="relative error below 1e-9",
)
def test_hjorth_mobility_is_the_documented_formula(sleep_recordings: list[Recording]) -> None:
    """Hertz, not radians per sample: the derivative is per second and divided by 2*pi."""
    recording = sleep_recordings[0]
    signal = ef.Signal.from_epochs(recording.epochs, recording=recording.name)
    table = ef.hjorth_mobility([signal], windows=[WHOLE], include_global=False)

    data = recording.epochs.get_data(picks="eeg")
    derivative = np.diff(data, axis=-1) * signal.sfreq
    reference = np.sqrt(np.var(derivative, axis=-1) / np.var(data, axis=-1)) / (2.0 * np.pi)
    np.testing.assert_allclose(table.values, reference, rtol=1e-9)


@pytest.mark.validates(
    "Signal.from_epochs",
    kind="behaviour",
    claim="The loaded epochs match the published recording",
    criterion="100 Hz, Fpz-Cz and Pz-Oz, at least 50 wake and 50 N3 epochs",
)
def test_epochs_are_what_the_tutorial_describes(sleep_recordings: list[Recording]) -> None:
    for recording in sleep_recordings:
        assert recording.epochs.info["sfreq"] == 100.0
        assert recording.epochs.ch_names == ["Fpz-Cz", "Pz-Oz"]
        counts = recording.metadata["stage"].value_counts()
        assert counts["W"] >= 50 and counts["N3"] >= 50, counts.to_dict()
