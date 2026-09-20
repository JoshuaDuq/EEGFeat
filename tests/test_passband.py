"""A requested band has to lie inside what preprocessing left behind.

Asking for gamma on data lowpassed at 30 Hz does not fail: it returns filter
roll-off shaped like a measurement. MNE tracks the edges, so the mismatch is
detectable, and these tests pin down that it is detected everywhere a band meets
a recording.
"""

import warnings

import mne
import numpy as np
import pytest

import eegfeat as ef
from eegfeat.bands import Band, check_passband, passband_fraction

SFREQ = 250.0
ALPHA = Band("alpha", 8.0, 13.0)
GAMMA = Band("gamma", 30.0, 45.0)
STRADDLING = Band("high-beta", 25.0, 35.0)


def _epochs(l_freq: float | None, h_freq: float | None, duration: float = 4.0):
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(
        rng.normal(size=(3, int(60 * SFREQ))) * 1e-5,
        mne.create_info(["C3", "Cz", "C4"], SFREQ, "eeg"),
        verbose="ERROR",
    )
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose="ERROR")
    return mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose="ERROR")


def _spectra(epochs):
    spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose="ERROR")
    return ef.Spectra.from_spectrum(
        spectrum, recording="test", estimator_parameters={"method": "welch"}
    )


@pytest.mark.parametrize(
    ("band", "highpass", "lowpass", "expected"),
    [
        (ALPHA, 1.0, 40.0, 1.0),
        (GAMMA, 0.0, 30.0, 0.0),
        (GAMMA, 0.0, 40.0, 2.0 / 3.0),
        (Band("delta", 0.5, 4.0), 1.0, 40.0, (4.0 - 1.0) / 3.5),
        (ALPHA, None, None, 1.0),  # unknown edges constrain nothing
        (ALPHA, 0.0, np.inf, 1.0),
    ],
)
def test_passband_fraction(band, highpass, lowpass, expected) -> None:
    assert passband_fraction(band, highpass, lowpass) == pytest.approx(expected)


def test_check_passband_refuses_a_disjoint_band() -> None:
    with pytest.raises(ValueError, match="lies outside the passband"):
        check_passband(GAMMA, 0.0, 30.0, source="a test")


def test_check_passband_warns_once_for_a_straddling_band_and_reports_the_fraction() -> None:
    with pytest.warns(UserWarning, match="50% of it carries signal"):
        fraction = check_passband(STRADDLING, 0.0, 30.0, source="a test")
    assert fraction == pytest.approx(0.5)


def test_check_passband_is_silent_when_the_band_fits() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_passband(ALPHA, 1.0, 40.0, source="a test") == 1.0


# --- the three places a band meets a recording ----------------------------------------


def test_a_band_signal_outside_the_passband_is_refused() -> None:
    epochs = _epochs(1.0, 30.0)
    with pytest.raises(ValueError, match="lies outside the passband"):
        ef.BandSignal.from_epochs(epochs, GAMMA, recording="test")


def test_a_band_signal_inside_the_passband_is_silent() -> None:
    epochs = _epochs(1.0, 30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ef.BandSignal.from_epochs(epochs, ALPHA, recording="test").band is ALPHA


def test_spectral_band_power_outside_the_passband_is_refused() -> None:
    spectra = _spectra(_epochs(1.0, 30.0))
    assert spectra.passband == (1.0, 30.0)
    with pytest.raises(ValueError, match="lies outside the passband"):
        ef.integrated_band_power(spectra, bands=[GAMMA], include_global=False)


def test_spectral_band_power_straddling_the_passband_warns_but_computes() -> None:
    spectra = _spectra(_epochs(1.0, 30.0))
    with pytest.warns(UserWarning, match="extends past the passband"):
        table = ef.integrated_band_power(spectra, bands=[STRADDLING], include_global=False)
    assert np.isfinite(table.values).all()


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("mne_connectivity") is None,
    reason="mne-connectivity is not installed",
)
def test_spectral_connectivity_outside_the_passband_is_refused() -> None:
    signal = ef.Signal.from_epochs(_epochs(1.0, 30.0), recording="test")
    assert signal.passband == (1.0, 30.0)
    with pytest.raises(ValueError, match="lies outside the passband"):
        ef.spectral_connectivity(
            signal, method="wpli", bands=[GAMMA], windows=[ef.Window("all", 0.0, 3.9)]
        )


# --- the check never fires when the edges are unknown ---------------------------------


def test_arrays_carry_no_passband_so_nothing_is_refused() -> None:
    # A caller who built the container themselves has made their own claim about
    # what the data contains; there is no info to contradict them.
    signal = ef.Signal.from_arrays(
        data=np.ones((2, 3, 400)),
        times=np.arange(400) / SFREQ,
        ch_names=("C3", "Cz", "C4"),
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(2)),
    )
    assert signal.passband is None


def test_unfiltered_epochs_constrain_nothing_below_nyquist() -> None:
    epochs = _epochs(None, None)
    assert epochs.info["highpass"] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ef.BandSignal.from_epochs(epochs, GAMMA, recording="test")
