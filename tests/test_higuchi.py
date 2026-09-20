"""Higuchi's fractal dimension.

Checked against known values for signals whose dimension is not in doubt, and
against antropy, which implements the same 1988 algorithm.
"""

import importlib.util

import numpy as np
import pytest

import eegfeat as ef
from eegfeat.signal import Signal
from eegfeat.spectra import Window

SFREQ = 250.0
_HAS_ANTROPY = importlib.util.find_spec("antropy") is not None


def _run(real: np.ndarray, sfreq: float = SFREQ, **kwargs) -> float:
    n = real.shape[-1]
    signal = Signal.from_arrays(
        data=np.asarray(real, dtype=float).reshape(1, 1, n),
        times=np.arange(n) / sfreq,
        ch_names=("C3",),
        sfreq=sfreq,
        row_ids=(("test", 0, "event"),),
    )
    table = ef.higuchi_fractal_dimension(
        [signal], windows=[Window("all", 0.0, (n - 1) / sfreq)], include_global=False, **kwargs
    )
    return float(table.values[0, 0])


def _signals(n: int = 4000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "sine": np.sin(2 * np.pi * 10.0 * np.arange(n) / SFREQ),
        "brown": np.cumsum(rng.normal(size=n)),
        "white": rng.normal(size=n),
    }


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [("sine", 1.0, 1.2), ("brown", 1.4, 1.6), ("white", 1.9, 2.1)],
)
def test_known_dimensions_are_recovered(name: str, low: float, high: float) -> None:
    assert low <= _run(_signals()[name]) <= high


def test_the_ordering_is_smooth_then_brown_then_white() -> None:
    values = {k: _run(v) for k, v in _signals().items()}
    assert values["sine"] < values["brown"] < values["white"]


@pytest.mark.skipif(not _HAS_ANTROPY, reason="antropy is not installed")
@pytest.mark.parametrize("k_max", [5, 10, 20])
@pytest.mark.parametrize("name", ["sine", "brown", "white"])
def test_it_agrees_with_antropy(name: str, k_max: int) -> None:
    import antropy

    signal = _signals()[name]
    assert _run(signal, k_max=k_max) == pytest.approx(
        float(antropy.higuchi_fd(signal, kmax=k_max)), rel=1e-9
    )


def test_it_is_invariant_to_amplitude_and_offset() -> None:
    signal = _signals()["brown"]
    reference = _run(signal)
    assert _run(signal * 1e6) == pytest.approx(reference, rel=1e-9)
    assert _run(signal + 500.0) == pytest.approx(reference, rel=1e-9)


def test_a_window_with_a_gap_is_withheld_rather_than_shortened() -> None:
    # A non-finite sample shortens one sub-curve and not the others, which tilts
    # the fit. The measure is about how length scales, so a partial curve is not
    # comparable and the cell is NaN instead.
    signal = _signals()["white"].copy()
    signal[500] = np.nan
    assert np.isnan(_run(signal))


def test_a_window_shorter_than_k_max_is_withheld() -> None:
    assert np.isnan(_run(np.random.default_rng(0).normal(size=8), k_max=10))


@pytest.mark.parametrize("k_max", [0, 1, -3, 2.5, True])
def test_an_unusable_k_max_raises(k_max) -> None:
    with pytest.raises(ValueError, match="k_max"):
        _run(_signals()["white"], k_max=k_max)


def test_it_is_recorded_with_its_k_max() -> None:
    n = 2000
    signal = Signal.from_arrays(
        data=np.random.default_rng(0).normal(size=(1, 1, n)),
        times=np.arange(n) / SFREQ,
        ch_names=("C3",),
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"),),
    )
    window = [Window("all", 0.0, (n - 1) / SFREQ)]
    five = ef.higuchi_fractal_dimension([signal], windows=window, include_global=False, k_max=5)
    ten = ef.higuchi_fractal_dimension([signal], windows=window, include_global=False, k_max=10)
    assert five.meta[0].computation.parameters["parameters"] == {"k_max": 5}
    assert five.names[0] != ten.names[0]
