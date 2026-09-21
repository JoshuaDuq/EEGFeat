"""MNE's SSVEP example: 12 Hz and 15 Hz visual flicker, one subject, twenty trials.

Steady-state visual evoked potentials put a narrow spectral line at the flicker
frequency and its harmonics over occipital cortex. The endogenous alpha rhythm
sits just below the 12 Hz line in this subject, which is exactly the situation a
spectral feature extractor has to get right.
"""

from __future__ import annotations

import numpy as np
import pytest

import eegfeat as ef
from validation.loaders import Recording

OCCIPITAL = {"occipital": ["O1", "Oz", "O2"]}
SEGMENT_SEC = 8.0


@pytest.fixture(scope="module")
def spectra(ssvep_recording: Recording) -> ef.Spectra:
    recording = ssvep_recording
    n_fft = int(recording.epochs.info["sfreq"] * SEGMENT_SEC)
    parameters = {
        "method": "welch",
        "fmin": 1.0,
        "fmax": 45.0,
        "tmin": 0.0,
        "tmax": 20.0,
        "n_fft": n_fft,
        "n_per_seg": n_fft,
        "n_overlap": n_fft // 2,
    }
    spectrum = recording.epochs.compute_psd(**parameters)
    return ef.Spectra.from_spectrum(
        spectrum, recording=recording.name, estimator_parameters=parameters
    )


@pytest.fixture(scope="module")
def flicker_hz(ssvep_recording: Recording) -> np.ndarray:
    return ssvep_recording.metadata["flicker_hz"].to_numpy(dtype=float)


def _occipital_log_ratio(
    spectra: ef.Spectra, numerator: ef.Band, denominator: ef.Band, *, mean: bool = False
) -> np.ndarray:
    measure = ef.mean_psd if mean else ef.integrated_band_power
    power = measure(
        spectra,
        bands=[numerator, denominator],
        groups=OCCIPITAL,
        include_global=False,
        normalize="log10",
    )
    return ef.band_ratio(power, numerator.name, denominator.name).values[:, 0]


def test_power_at_the_flicker_frequency_separates_every_trial(
    spectra: ef.Spectra, flicker_hz: np.ndarray
) -> None:
    ratio = _occipital_log_ratio(spectra, ef.Band("f12", 11.5, 12.5), ef.Band("f15", 14.5, 15.5))
    assert ratio[flicker_hz == 12.0].min() > ratio[flicker_hz == 15.0].max()


def test_the_second_harmonic_follows_the_flicker(
    spectra: ef.Spectra, flicker_hz: np.ndarray
) -> None:
    ratio = _occipital_log_ratio(spectra, ef.Band("h24", 23.5, 24.5), ef.Band("h30", 29.5, 30.5))
    assert ratio[flicker_hz == 12.0].min() > ratio[flicker_hz == 15.0].max()


@pytest.mark.parametrize("frequency", [12.0, 15.0])
def test_the_flicker_line_stands_above_its_neighbourhood(
    spectra: ef.Spectra, flicker_hz: np.ndarray, frequency: float
) -> None:
    """Signal-to-noise as MNE's tutorial defines it: the line against the bins around it."""
    line = ef.Band("line", frequency - 0.25, frequency + 0.25)
    neighbourhood = ef.Band("around", frequency - 1.5, frequency + 1.5)
    snr = _occipital_log_ratio(spectra, line, neighbourhood, mean=True)
    driven = flicker_hz == frequency
    assert snr[driven].min() > snr[~driven].max()


def test_peak_frequency_recovers_the_flicker(spectra: ef.Spectra, flicker_hz: np.ndarray) -> None:
    """A narrow entrained line needs the search told not to smooth it away.

    The default 1 Hz smoothing suits broad endogenous peaks; here it blurs the
    flicker line into the alpha shoulder. With smoothing off and alpha excluded
    from the band, the interpolated peak lands on the stimulation frequency.
    """
    peak = ef.peak_frequency(
        spectra,
        band=ef.Band("ssvep", 11.0, 16.0),
        smoothing_hz=0.0,
        groups=OCCIPITAL,
        include_global=False,
    )
    found = peak.values[:, 0]
    assert np.isfinite(found).all()
    for frequency in (12.0, 15.0):
        assert abs(np.median(found[flicker_hz == frequency]) - frequency) < 0.3
    assert np.mean(np.abs(found - flicker_hz) < 0.3) >= 0.7
