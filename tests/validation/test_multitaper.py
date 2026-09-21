"""The multitaper PSD path on the motor task.

Multitaper and Welch are two estimators of the same density. With
``normalization="full"`` the multitaper band power integrates the same way,
agrees with Welch at the median, and MNE's default length normalization is
refused because it is not a density.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

import eegfeat as ef
from validation.loaders import Recording

MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
ESTIMATE = {"fmin": 1.0, "fmax": 40.0, "tmin": 0.5, "tmax": 3.5}


@pytest.fixture(scope="module")
def first(eegbci_recordings: list[Recording]) -> Recording:
    return eegbci_recordings[0]


@pytest.fixture(scope="module")
def multitaper(first: Recording) -> tuple[ef.Spectra, np.ndarray, np.ndarray]:
    parameters = {"method": "multitaper", "bandwidth": 2.0, "normalization": "full", **ESTIMATE}
    spectrum = first.epochs.compute_psd(**parameters)
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=first.name, estimator_parameters=parameters
    )
    return spectra, spectrum.get_data(), spectrum.freqs


@pytest.fixture(scope="module")
def welch(first: Recording) -> ef.Spectra:
    parameters = {"method": "welch", "n_fft": 480, "n_per_seg": 480, "n_overlap": 240, **ESTIMATE}
    spectrum = first.epochs.compute_psd(**parameters)
    return ef.Spectra.from_spectrum(spectrum, recording=first.name, estimator_parameters=parameters)


def test_multitaper_band_power_integrates_to_the_exact_band_edges(
    multitaper: tuple[ef.Spectra, np.ndarray, np.ndarray],
) -> None:
    """A multitaper grid does not land on 8 or 13 Hz; the integral still runs edge to edge.

    The reference interpolates the PSD at both edges and integrates from one to the
    other, which is what the band-integration weights encode.
    """
    spectra, psd, freqs = multitaper
    assert MU.fmin not in freqs and MU.fmax not in freqs
    inside = (freqs >= MU.fmin) & (freqs <= MU.fmax)
    grid = np.concatenate([[MU.fmin], freqs[inside], [MU.fmax]])
    reference = np.array(
        [[trapezoid(np.interp(grid, freqs, channel), grid) for channel in epoch] for epoch in psd]
    )
    power = ef.integrated_band_power(spectra, bands=[MU], include_global=False)
    np.testing.assert_allclose(power.values, reference, rtol=1e-9)


@pytest.mark.parametrize("band", [MU, BETA], ids=lambda band: band.name)
def test_multitaper_agrees_with_welch(
    multitaper: tuple[ef.Spectra, np.ndarray, np.ndarray], welch: ef.Spectra, band: ef.Band
) -> None:
    spectra, _, _ = multitaper
    mt = ef.integrated_band_power(spectra, bands=[band], include_global=False).values
    wl = ef.integrated_band_power(welch, bands=[band], include_global=False).values
    assert 0.9 < np.median(mt / wl) < 1.1, np.median(mt / wl)
    assert np.corrcoef(np.log(mt.ravel()), np.log(wl.ravel()))[0, 1] > 0.9


def test_length_normalized_multitaper_is_refused(first: Recording) -> None:
    spectrum = first.epochs.compute_psd(method="multitaper", **ESTIMATE)
    with pytest.raises(ValueError, match='normalization="full"'):
        ef.Spectra.from_spectrum(
            spectrum, recording=first.name, estimator_parameters={"method": "multitaper"}
        )


def test_passband_helpers_read_the_recording_filters(ssvep_recording: Recording) -> None:
    """The SSVEP recording was high-passed at 0.1 Hz; a band below that is refused."""
    info = ssvep_recording.epochs.info
    highpass, lowpass = info["highpass"], info["lowpass"]
    assert highpass == pytest.approx(0.1)
    assert ef.passband_fraction(MU, highpass, lowpass) == 1.0
    assert ef.passband_fraction(
        ef.Band("straddle", 0.05, 0.15), highpass, lowpass
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ef.check_passband(ef.Band("dc", 0.0, 0.05), highpass, lowpass, source="validation")
    with pytest.warns(UserWarning):
        ef.check_passband(ef.Band("edge", 0.05, 0.2), highpass, lowpass, source="validation")
