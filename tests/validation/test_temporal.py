"""Time-domain measures: formulas on Sleep-EDF, amplitude with sleep depth, and
derived asymmetry on the motor task."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

import eegfeat as ef
from validation.loaders import Recording

WHOLE_EPOCH = ef.Window("epoch", 0.0, 30.0)
OCCIPITAL = "Pz-Oz"
MU = ef.Band("mu", 8.0, 13.0)


@pytest.fixture(scope="module")
def sleep_signal(sleep_recordings: list[Recording]) -> tuple[ef.Signal, np.ndarray, np.ndarray]:
    recording = sleep_recordings[0]
    signal = ef.Signal.from_epochs(recording.epochs, recording=recording.name)
    data = recording.epochs.get_data(picks="eeg")
    return signal, data, recording.metadata["stage"].to_numpy()


def test_time_domain_formulas(sleep_signal: tuple[ef.Signal, np.ndarray, np.ndarray]) -> None:
    signal, data, _ = sleep_signal

    def values(measure: object, **kwargs: object) -> np.ndarray:
        table = measure([signal], windows=[WHOLE_EPOCH], include_global=False, **kwargs)  # type: ignore[operator]
        return np.asarray(table.values)

    np.testing.assert_allclose(values(ef.root_mean_square), np.sqrt(np.mean(data**2, axis=-1)))
    np.testing.assert_allclose(values(ef.peak_to_peak), np.ptp(data, axis=-1))
    np.testing.assert_allclose(values(ef.mean_amplitude), data.mean(axis=-1))
    np.testing.assert_allclose(values(ef.amplitude_quantile), np.median(data, axis=-1))
    np.testing.assert_allclose(
        values(ef.amplitude_quantile, q=0.9), np.quantile(data, 0.9, axis=-1)
    )
    np.testing.assert_allclose(values(ef.skewness), stats.skew(data, axis=-1))
    # Fisher's definition: zero for a Gaussian.
    np.testing.assert_allclose(values(ef.kurtosis), stats.kurtosis(data, axis=-1))
    # Per second, so recordings at different rates are comparable.
    np.testing.assert_allclose(
        values(ef.line_length), np.mean(np.abs(np.diff(data, axis=-1)), axis=-1) * signal.sfreq
    )


def test_slow_waves_are_large_and_smooth(
    sleep_signal: tuple[ef.Signal, np.ndarray, np.ndarray],
) -> None:
    signal, _, stage = sleep_signal
    kept = np.isin(stage, ["W", "N3"])
    deep = (stage[kept] == "N3").astype(int)

    def auc(measure: object) -> float:
        table = measure([signal], windows=[WHOLE_EPOCH], include_global=False)  # type: ignore[operator]
        column = [m.space for m in table.meta].index(OCCIPITAL)
        return float(roc_auc_score(deep, table.values[kept, column]))

    assert auc(ef.root_mean_square) > 0.7
    assert auc(ef.peak_to_peak) > 0.7
    assert auc(ef.line_length) < 0.3


def test_asymmetry_is_right_minus_left_on_log_power(
    eegbci_recordings: list[Recording],
) -> None:
    recording = eegbci_recordings[0]
    parameters = {"fmin": 1.0, "fmax": 40.0, "tmin": 0.5, "tmax": 3.5, "n_fft": 320}
    spectrum = recording.epochs.compute_psd("welch", **parameters)
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=recording.name, estimator_parameters={"method": "welch", **parameters}
    )
    power = ef.integrated_band_power(spectra, bands=[MU], normalize="log10", include_global=False)
    asymmetry = ef.asymmetry(power, pairs=[("C3", "C4"), ("O1", "O2")])
    columns = {m.space: i for i, m in enumerate(power.meta)}
    for i, meta in enumerate(asymmetry.meta):
        left, right = meta.space.split("-")
        expected = power.values[:, columns[right]] - power.values[:, columns[left]]
        np.testing.assert_allclose(asymmetry.values[:, i], expected, rtol=1e-12)
