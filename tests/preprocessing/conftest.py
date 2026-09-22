import mne
import numpy as np
import pytest


@pytest.fixture
def raw():
    rng = np.random.default_rng(42)
    sfreq = 250.0
    times = np.arange(7500) / sfreq
    channels = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2", "VEOG", "ECG", "STI"]
    data = rng.normal(scale=2e-6, size=(11, len(times)))
    data[:8] += 5e-6 * np.sin(2 * np.pi * 10 * times)
    data[:8] += 3e-6 * np.sin(2 * np.pi * 20 * times)
    data[-1] = 0
    data[-1, [1250, 2500, 3750, 5000, 6250]] = 1
    result = mne.io.RawArray(
        data,
        mne.create_info(channels, sfreq, ["eeg"] * 8 + ["eog", "ecg", "stim"]),
        first_samp=1000,
    )
    result.set_montage("colin27_1020")
    return result


@pytest.fixture
def mixture(raw):
    # Full-rank non-Gaussian sources: identical sinusoids cannot be separated by ICA.
    rng = np.random.default_rng(13)
    sources = rng.laplace(size=(8, raw.n_times)) * 1e-6
    data = np.vstack([rng.normal(size=(8, 8)) @ sources, raw.get_data()[8:]])
    return mne.io.RawArray(data, raw.info.copy(), first_samp=raw.first_samp)
