"""Hjorth mobility and complexity, in physical units.

The usual ``std(diff(x)) / std(x)`` is per sample, so the same recording at
another sampling rate reports a different mobility. These tests pin the
measures to hertz and to the reference algorithm.
"""

import numpy as np
import pytest
from scipy.signal import resample

import eegfeat as ef
from eegfeat.signal import Signal
from eegfeat.spectra import Window


def _series(real: np.ndarray, sfreq: float) -> Signal:
    n = real.shape[-1]
    return Signal.from_arrays(
        data=np.asarray(real, dtype=float).reshape(1, 1, n),
        times=np.arange(n) / sfreq,
        ch_names=("C3",),
        sfreq=sfreq,
        row_ids=(("test", 0, "event"),),
    )


def _value(function, real: np.ndarray, sfreq: float) -> float:
    signal = _series(real, sfreq)
    window = Window("all", 0.0, (real.shape[-1] - 1) / sfreq)
    return float(function([signal], windows=[window], include_global=False).values[0, 0])


def _sine(f0: float, sfreq: float, seconds: float = 8.0) -> np.ndarray:
    return np.sin(2 * np.pi * f0 * np.arange(int(seconds * sfreq)) / sfreq)


@pytest.mark.parametrize("f0", [5.0, 10.0, 25.0])
@pytest.mark.parametrize("sfreq", [128.0, 250.0, 1000.0])
def test_mobility_of_a_sine_is_its_frequency_in_hertz(f0: float, sfreq: float) -> None:
    measured = _value(ef.hjorth_mobility, _sine(f0, sfreq), sfreq)
    # A finite difference has gain sin(pi f / fs), not pi f / fs, so the estimate
    # falls slightly short near Nyquist. That exact shortfall is the target here:
    # anything else means the sampling interval did not cancel.
    expected = f0 * np.sin(np.pi * f0 / sfreq) / (np.pi * f0 / sfreq)
    assert measured == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("sfreq", [128.0, 250.0, 1000.0])
def test_complexity_of_a_sine_is_one(sfreq: float) -> None:
    assert _value(ef.hjorth_complexity, _sine(10.0, sfreq), sfreq) == pytest.approx(1.0, abs=5e-3)


def test_mobility_survives_resampling_of_the_same_content() -> None:
    # The failure this measure exists to avoid: a per-sample mobility changes by the
    # ratio of the sampling rates, an 8x swing here, for identical signal content.
    base = 1000.0
    t = np.arange(int(8 * base)) / base
    wave = sum(np.sin(2 * np.pi * f * t + p) for f, p in [(6, 0.1), (10, 0.7), (14, 2.1)])
    values = {}
    for sfreq in (1000.0, 500.0, 250.0, 125.0):
        values[sfreq] = _value(
            ef.hjorth_mobility, resample(wave, int(len(wave) * sfreq / base)), sfreq
        )
    spread = max(values.values()) / min(values.values())
    assert spread < 1.05, values
    assert values[1000.0] == pytest.approx(10.0, rel=0.2)


def test_complexity_is_dimensionless_and_rate_free() -> None:
    base = 1000.0
    t = np.arange(int(8 * base)) / base
    wave = sum(np.sin(2 * np.pi * f * t + p) for f, p in [(6, 0.1), (10, 0.7), (14, 2.1)])
    values = [
        _value(ef.hjorth_complexity, resample(wave, int(len(wave) * sfreq / base)), sfreq)
        for sfreq in (1000.0, 500.0, 250.0)
    ]
    assert max(values) / min(values) < 1.02, values


def test_amplitude_scaling_leaves_both_unchanged() -> None:
    wave = _sine(10.0, 250.0) * 1e-5
    for function in (ef.hjorth_mobility, ef.hjorth_complexity):
        assert _value(function, wave, 250.0) == pytest.approx(
            _value(function, wave * 1e6, 250.0), rel=1e-9
        )


def test_complexity_is_never_below_one_and_is_one_only_for_a_single_tone() -> None:
    # Cauchy-Schwarz on the spectral moments gives M2**2 <= M0 * M4, so
    # sqrt(M4 * M0) / M2 >= 1 with equality only when the spectrum is one line.
    # That bound, not any notion of "broader spectrum", is what complexity obeys:
    # two separated tones score above white noise.
    sfreq = 500.0
    t = np.arange(int(8 * sfreq)) / sfreq
    rng = np.random.default_rng(0)
    signals = {
        "pure": np.sin(2 * np.pi * 10 * t),
        "two tones": np.sin(2 * np.pi * 10 * t) + 0.7 * np.sin(2 * np.pi * 30 * t),
        "white noise": rng.normal(size=t.size),
        "brown noise": np.cumsum(rng.normal(size=t.size)),
    }
    values = {k: _value(ef.hjorth_complexity, x, sfreq) for k, x in signals.items()}
    assert all(v >= 1.0 - 1e-6 for v in values.values()), values
    assert values["pure"] == pytest.approx(1.0, abs=5e-3)
    assert all(v > 1.01 for k, v in values.items() if k != "pure"), values


def test_complexity_matches_the_spectral_moments_of_a_two_tone_signal() -> None:
    # An exact target rather than an ordering: for tones at f1 and f2 with powers
    # p1 and p2, complexity is sqrt(M4 * M0) / M2 over those two lines.
    sfreq = 2000.0  # far from Nyquist, so the finite difference is near-exact
    t = np.arange(int(16 * sfreq)) / sfreq
    f1, f2, a2 = 10.0, 30.0, 0.7
    wave = np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    powers = np.array([1.0, a2**2])
    freqs = np.array([f1, f2])
    m0 = powers.sum()
    m2 = (powers * freqs**2).sum()
    m4 = (powers * freqs**4).sum()
    assert _value(ef.hjorth_complexity, wave, sfreq) == pytest.approx(
        float(np.sqrt(m4 * m0) / m2), rel=1e-3
    )


def test_a_flat_signal_has_no_mobility_or_complexity() -> None:
    flat = np.ones(400)
    assert np.isnan(_value(ef.hjorth_mobility, flat, 250.0))
    assert np.isnan(_value(ef.hjorth_complexity, flat, 250.0))


def test_the_units_say_what_the_numbers_are() -> None:
    signal = _series(_sine(10.0, 250.0), 250.0)
    window = [Window("all", 0.0, 7.9)]
    assert ef.hjorth_mobility([signal], windows=window, include_global=False).meta[0].unit == "Hz"
    assert (
        ef.hjorth_complexity([signal], windows=window, include_global=False).meta[0].unit == "a.u."
    )


def test_they_work_on_a_band_envelope_too() -> None:
    # The TimeSeries protocol, not just raw signals: a band envelope is slower than
    # the band it came from, and mobility should say so.
    sfreq = 250.0
    t = np.arange(int(8 * sfreq)) / sfreq
    carrier = (1.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t)) * np.cos(2 * np.pi * 10.0 * t)
    band = ef.BandSignal.from_arrays(
        analytic=np.asarray(carrier, dtype=complex).reshape(1, 1, -1),
        times=t,
        ch_names=("C3",),
        band=ef.Band("alpha", 8.0, 13.0),
        sfreq=sfreq,
        row_ids=(("test", 0, "event"),),
    )
    window = [Window("all", 0.0, (t.size - 1) / sfreq)]
    table = ef.hjorth_mobility([band], windows=window, include_global=False)
    assert table.meta[0].band is not None and table.meta[0].band.name == "alpha"
    assert np.isfinite(table.values).all()
