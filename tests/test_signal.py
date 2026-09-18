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
