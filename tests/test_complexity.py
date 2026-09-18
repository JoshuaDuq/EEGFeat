import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.complexity import _coarse_grain, _sample_entropy, multiscale_entropy, sample_entropy
from eegfeat.signal import BandSignal, Signal
from eegfeat.spectra import Window

SFREQ = 100.0
WINDOW = Window("all", 0.0, 2.0)


def _signal(values: np.ndarray) -> Signal:
    data = values.reshape(1, 1, values.size)
    return Signal.from_arrays(
        data=data, times=np.arange(values.size) / SFREQ, ch_names=("C3",), sfreq=SFREQ
    )


def test_a_constant_signal_has_zero_entropy() -> None:
    # Every template matches every other at both lengths, so the ratio is one.
    table = sample_entropy([_signal(np.full(201, 2.0))], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(0.0, abs=1e-12)


def test_noise_is_less_self_similar_than_a_sine() -> None:
    rng = np.random.RandomState(0)
    noise = sample_entropy([_signal(rng.randn(201))], windows=[WINDOW], include_global=False)
    sine = sample_entropy(
        [_signal(np.sin(np.linspace(0, 20 * np.pi, 201)))], windows=[WINDOW], include_global=False
    )
    assert noise.values.item() > sine.values.item()


def test_a_window_too_short_to_embed_yields_nan() -> None:
    short = Window("short", 0.0, 0.02)  # 3 samples at 100 Hz, order 2 needs 4
    table = sample_entropy(
        [_signal(np.arange(201, dtype=float))], windows=[short], include_global=False
    )
    assert np.isnan(table.values).all()


def test_coarse_graining_averages_non_overlapping_blocks() -> None:
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(_coarse_grain(x, 2), [2.0, 6.0])  # trailing sample dropped
    np.testing.assert_allclose(_coarse_grain(x, 1), x)


def test_coarse_graining_drops_non_finite_samples_before_blocking() -> None:
    # Removing the NaN leaves three finite samples, so one block of two fits and
    # the trailing sample is discarded. Note this closes the gap rather than
    # preserving sample positions.
    x = np.array([1.0, np.nan, 3.0, 5.0])
    np.testing.assert_allclose(_coarse_grain(x, 2), [2.0])


def test_multiscale_scale_one_equals_sample_entropy() -> None:
    rng = np.random.RandomState(1)
    signal = _signal(rng.randn(201))
    mse = multiscale_entropy([signal], windows=[WINDOW], scales=(1,), include_global=False)
    plain = sample_entropy([signal], windows=[WINDOW], include_global=False)
    np.testing.assert_allclose(mse.values, plain.values)


def test_multiscale_emits_one_column_per_scale() -> None:
    rng = np.random.RandomState(2)
    table = multiscale_entropy(
        [_signal(rng.randn(201))], windows=[WINDOW], scales=(1, 2, 3), include_global=False
    )
    assert [m.measure for m in table.meta] == ["mse01", "mse02", "mse03"]


def test_a_scale_that_leaves_too_few_samples_yields_nan() -> None:
    rng = np.random.RandomState(3)
    table = multiscale_entropy(
        [_signal(rng.randn(201))], windows=[WINDOW], scales=(100,), include_global=False
    )
    assert np.isnan(table.values).all()


def test_entropy_works_on_a_band_envelope() -> None:
    n = 201
    rng = np.random.RandomState(4)
    envelope = (1.0 + 0.1 * rng.randn(n)).astype(complex).reshape(1, 1, n)
    band_signal = BandSignal.from_arrays(
        analytic=envelope,
        times=np.arange(n) / SFREQ,
        ch_names=("C3",),
        band=Band("beta", 13.0, 30.0),
        sfreq=SFREQ,
    )
    table = sample_entropy([band_signal], windows=[WINDOW], include_global=False)
    assert np.isfinite(table.values).all()
    assert table.meta[0].band is not None


def test_no_matching_template_is_undefined_not_zero() -> None:
    # A strictly increasing ramp: with a small tolerance nothing matches anything.
    assert np.isnan(_sample_entropy(np.arange(50, dtype=float), 2, 1e-9))


def test_chunking_does_not_change_the_result() -> None:
    import eegfeat.complexity as mod

    rng = np.random.RandomState(5)
    x = rng.randn(400)
    full = _sample_entropy(x, 2, 0.2)
    original = mod._PAIR_BUDGET
    try:
        mod._PAIR_BUDGET = 64  # force many small chunks
        chunked = _sample_entropy(x, 2, 0.2)
    finally:
        mod._PAIR_BUDGET = original
    assert chunked == pytest.approx(full, rel=1e-12)


@pytest.mark.parametrize(("order", "r"), [(0, 0.2), (-1, 0.2), (2, 0.0), (2, -0.1)])
def test_invalid_parameters_raise(order: int, r: float) -> None:
    with pytest.raises(ValueError):
        sample_entropy([_signal(np.zeros(50))], windows=[WINDOW], order=order, r=r)


def test_invalid_scales_raise() -> None:
    with pytest.raises(ValueError, match="scales"):
        multiscale_entropy([_signal(np.zeros(50))], windows=[WINDOW], scales=(0,))


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("antropy") is None,
    reason="antropy is not installed in this environment",
)
def test_matches_antropy_where_it_is_available() -> None:
    from antropy import sample_entropy as antropy_sampen

    rng = np.random.RandomState(6)
    for values in (rng.randn(300), np.sin(np.linspace(0, 30 * np.pi, 300)), np.full(200, 1.0)):
        tolerance = 0.2 * float(np.std(values))
        expected = antropy_sampen(values, order=2, tolerance=tolerance, metric="chebyshev")
        assert _sample_entropy(values, 2, 0.2) == pytest.approx(expected, rel=1e-12, abs=1e-12)
