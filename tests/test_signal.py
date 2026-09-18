import mne
import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.signal import BandSignal

BETA = Band("beta", 13.0, 30.0)


def _signal(analytic: np.ndarray, sfreq: float = 100.0) -> BandSignal:
    times = np.arange(analytic.shape[-1]) / sfreq
    return BandSignal.from_arrays(
        analytic=analytic, times=times, ch_names=("C3", "C4"), band=BETA, sfreq=sfreq
    )


def test_envelope_phase_and_power_are_views_of_the_analytic_signal() -> None:
    analytic = np.array([[[1 + 1j, 0 + 2j], [3 + 0j, -1 - 1j]]])
    signal = _signal(analytic)
    np.testing.assert_allclose(signal.envelope, np.abs(analytic))
    np.testing.assert_allclose(signal.phase, np.angle(analytic))
    np.testing.assert_allclose(signal.power, np.abs(analytic) ** 2)


def test_power_is_the_squared_envelope_exactly() -> None:
    signal = _signal(np.array([[[3 + 4j, 5 + 12j], [1 + 0j, 0 + 1j]]]))
    np.testing.assert_allclose(signal.power, signal.envelope**2)
    np.testing.assert_allclose(signal.envelope[0, 0], [5.0, 13.0])


def test_coverage_defaults_to_where_the_input_was_finite() -> None:
    analytic = np.ones((1, 2, 3), dtype=complex)
    analytic[0, 1, 2] = np.nan
    signal = _signal(analytic)
    assert signal.coverage[0, 1, 2] == 0.0
    assert signal.coverage.sum() == 5.0


def test_shape_mismatches_raise() -> None:
    good = dict(
        analytic=np.ones((2, 2, 4), dtype=complex),
        times=np.arange(4) / 100.0,
        ch_names=("C3", "C4"),
        band=BETA,
        sfreq=100.0,
    )
    with pytest.raises(ValueError, match="ch_names"):
        BandSignal.from_arrays(**{**good, "ch_names": ("C3",)})
    with pytest.raises(ValueError, match="times"):
        BandSignal.from_arrays(**{**good, "times": np.arange(3) / 100.0})
    with pytest.raises(ValueError, match="ascending"):
        BandSignal.from_arrays(**{**good, "times": np.array([3.0, 2.0, 1.0, 0.0])})
    with pytest.raises(ValueError, match="3-D"):
        BandSignal.from_arrays(**{**good, "analytic": np.ones((2, 4), dtype=complex)})


def test_a_real_valued_input_raises_rather_than_silently_losing_phase() -> None:
    with pytest.raises(TypeError, match="complex"):
        BandSignal.from_arrays(
            analytic=np.ones((1, 2, 4)),
            times=np.arange(4) / 100.0,
            ch_names=("C3", "C4"),
            band=BETA,
            sfreq=100.0,
        )


def test_non_positive_sfreq_raises() -> None:
    with pytest.raises(ValueError, match="sfreq"):
        BandSignal.from_arrays(
            analytic=np.ones((1, 2, 4), dtype=complex),
            times=np.arange(4),
            ch_names=("C3", "C4"),
            band=BETA,
            sfreq=0.0,
        )


ALPHA = Band("alpha", 8.0, 13.0)


def _epochs(freq_hz: float, n_epochs: int = 3, sfreq: float = 200.0, dur: float = 4.0):
    n = int(sfreq * dur)
    t = np.arange(n) / sfreq
    wave = np.cos(2 * np.pi * freq_hz * t)
    data = np.tile(wave, (n_epochs, 2, 1))
    info = mne.create_info(["C3", "C4"], sfreq, "eeg")
    return mne.EpochsArray(data, info, tmin=-1.0, verbose="ERROR")


def test_a_sine_inside_the_band_yields_a_flat_envelope_at_its_amplitude() -> None:
    signal = BandSignal.from_epochs(_epochs(10.0), ALPHA)
    interior = signal.envelope[:, :, 100:-100]
    np.testing.assert_allclose(interior, 1.0, rtol=0.05)


def test_a_sine_outside_the_band_is_attenuated() -> None:
    inside = BandSignal.from_epochs(_epochs(10.0), ALPHA).envelope[:, :, 100:-100].mean()
    outside = BandSignal.from_epochs(_epochs(40.0), ALPHA).envelope[:, :, 100:-100].mean()
    assert outside < inside / 20.0


def test_padding_protects_the_edges() -> None:
    padded = BandSignal.from_epochs(_epochs(10.0), ALPHA)
    unpadded = BandSignal.from_epochs(_epochs(10.0), ALPHA, pad_sec=0.0, pad_cycles=0.0)
    edge = slice(0, 20)
    assert abs(padded.envelope[0, 0, edge].mean() - 1.0) < abs(
        unpadded.envelope[0, 0, edge].mean() - 1.0
    )


def test_shape_times_and_band_are_carried_through() -> None:
    epochs = _epochs(10.0)
    signal = BandSignal.from_epochs(epochs, ALPHA)
    assert signal.analytic.shape == (3, 2, len(epochs.times))
    np.testing.assert_allclose(signal.times, epochs.times)
    assert signal.ch_names == ("C3", "C4")
    assert signal.band is ALPHA
    assert signal.sfreq == 200.0


def test_a_band_above_nyquist_raises() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        BandSignal.from_epochs(_epochs(10.0), Band("vhf", 90.0, 150.0))


def test_coverage_reflects_the_filtered_output_not_the_raw_input() -> None:
    # One bad input sample propagates through the FIR convolution and the Hilbert
    # transform and destroys the whole epoch. Coverage must say so, or a caller
    # filtering on coverage keeps a column that is entirely NaN.
    epochs = _epochs(10.0)
    data = np.asarray(epochs.get_data())
    data[0, 0, data.shape[2] // 2] = np.nan
    dirty = mne.EpochsArray(data, epochs.info, tmin=epochs.tmin, verbose="ERROR")
    signal = BandSignal.from_epochs(dirty, ALPHA)
    assert not np.isfinite(signal.envelope[0, 0]).any()
    assert signal.coverage[0, 0].max() == 0.0
    assert signal.coverage[1, 0].min() == 1.0
