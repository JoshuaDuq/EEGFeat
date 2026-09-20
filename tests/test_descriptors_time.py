"""Time-domain descriptors: exact targets, not ranges.

Several of these are rate-dependent in their conventional per-sample form. The
tests pin the physical units, and pin zero crossings to hand-counted signals.
"""

import numpy as np
import pytest
from scipy import stats
from scipy.signal import butter, filtfilt, resample

import eegfeat as ef
from eegfeat.signal import Signal
from eegfeat.spectra import Window

SFREQ = 250.0


def _run(function, real: np.ndarray, sfreq: float = SFREQ, **kwargs) -> float:
    n = real.shape[-1]
    signal = Signal.from_arrays(
        data=np.asarray(real, dtype=float).reshape(1, 1, n),
        times=np.arange(n) / sfreq,
        ch_names=("C3",),
        sfreq=sfreq,
        row_ids=(("test", 0, "event"),),
    )
    table = function(
        [signal], windows=[Window("all", 0.0, (n - 1) / sfreq)], include_global=False, **kwargs
    )
    return float(table.values[0, 0])


@pytest.fixture
def wave() -> np.ndarray:
    return np.random.default_rng(0).normal(size=4000) * 3.0 + 1.5


def test_root_mean_square_matches_its_definition(wave) -> None:
    assert _run(ef.root_mean_square, wave) == pytest.approx(float(np.sqrt(np.mean(wave**2))))


def test_rms_exceeds_the_standard_deviation_when_there_is_an_offset(wave) -> None:
    # The distinction from variance: this one does not remove the mean.
    assert _run(ef.root_mean_square, wave) > float(np.std(wave))
    centred = wave - wave.mean()
    assert _run(ef.root_mean_square, centred) == pytest.approx(float(np.std(centred)))


def test_skewness_and_kurtosis_match_scipy(wave) -> None:
    assert _run(ef.skewness, wave) == pytest.approx(float(stats.skew(wave)))
    assert _run(ef.kurtosis, wave) == pytest.approx(float(stats.kurtosis(wave)))


def test_kurtosis_is_excess_so_a_gaussian_is_zero() -> None:
    gaussian = np.random.default_rng(1).normal(size=200_000)
    assert _run(ef.kurtosis, gaussian) == pytest.approx(0.0, abs=0.05)


def test_skewness_and_kurtosis_are_scale_free(wave) -> None:
    for function in (ef.skewness, ef.kurtosis):
        assert _run(function, wave) == pytest.approx(_run(function, wave * 1e6), rel=1e-9)


def test_line_length_is_per_second_not_per_sample(wave) -> None:
    assert _run(ef.line_length, wave) == pytest.approx(
        float(np.mean(np.abs(np.diff(wave))) * SFREQ)
    )


def test_line_length_survives_resampling_of_the_same_content() -> None:
    base = 1000.0
    t = np.arange(int(8 * base)) / base
    signal = sum(np.sin(2 * np.pi * f * t + p) for f, p in [(6, 0.1), (10, 0.7), (14, 2.1)])
    values = [
        _run(ef.line_length, resample(signal, int(len(signal) * sfreq / base)), sfreq)
        for sfreq in (1000.0, 500.0, 250.0)
    ]
    assert max(values) / min(values) < 1.05, values


@pytest.mark.parametrize(
    ("samples", "crossings"),
    [
        ([1.0, -1.0, 1.0, -1.0], 3),
        ([-1.0, -0.5, 0.0, -0.5, -1.0], 0),  # touches the axis, never crosses
        ([-1.0, -0.5, 0.0, 0.5, 1.0], 1),
        ([1.0, 0.0, 1.0, 0.0, 1.0], 0),  # rests at zero and returns the same side
        ([1.0, 1.0, 1.0], 0),
        ([0.0, 0.0, 0.0], 0),
    ],
)
def test_zero_crossings_are_counted_exactly(samples, crossings) -> None:
    sfreq = 10.0
    data = np.array(samples, dtype=float)
    rate = _run(ef.zero_crossing_rate, data, sfreq)
    assert rate * data.size / sfreq == pytest.approx(float(crossings))


def test_zero_crossing_rate_is_a_rate_so_window_length_does_not_change_it() -> None:
    sfreq = 250.0
    t = np.arange(int(8 * sfreq)) / sfreq
    sine = np.sin(2 * np.pi * 10.0 * t)
    assert _run(ef.zero_crossing_rate, sine, sfreq) == pytest.approx(20.0, rel=0.01)
    assert _run(ef.zero_crossing_rate, sine[: t.size // 2], sfreq) == pytest.approx(20.0, rel=0.02)


def test_zero_crossing_rate_approximates_twice_mobility_for_a_band_limited_signal() -> None:
    # Rice's formula, and the docstring's claim: it holds for band-limited content
    # well below Nyquist, which is what this asserts, not for raw white noise.
    sfreq = 1000.0
    coefficients = butter(4, [8.0 / (sfreq / 2), 13.0 / (sfreq / 2)], btype="band")
    signal = filtfilt(*coefficients, np.random.default_rng(0).normal(size=int(60 * sfreq)))
    rate = _run(ef.zero_crossing_rate, signal, sfreq)
    assert rate == pytest.approx(2.0 * _run(ef.hjorth_mobility, signal, sfreq), rel=0.05)


def test_amplitude_quantile_defaults_to_the_median(wave) -> None:
    assert _run(ef.amplitude_quantile, wave) == pytest.approx(float(np.median(wave)))
    assert _run(ef.amplitude_quantile, wave, q=0.9) == pytest.approx(float(np.quantile(wave, 0.9)))


def test_the_median_resists_a_transient_that_moves_the_mean(wave) -> None:
    spiked = wave.copy()
    spiked[100] = 1e6
    assert _run(ef.amplitude_quantile, spiked) == pytest.approx(_run(ef.amplitude_quantile, wave))
    assert _run(ef.mean_amplitude, spiked) != pytest.approx(_run(ef.mean_amplitude, wave))


@pytest.mark.parametrize("q", [-0.1, 1.1, np.nan])
def test_an_out_of_range_quantile_raises(q) -> None:
    with pytest.raises(ValueError, match="q must be"):
        _run(ef.amplitude_quantile, np.ones(100), q=q)


def test_the_units_say_what_the_numbers_are() -> None:
    n = 400
    signal = Signal.from_arrays(
        data=np.random.default_rng(0).normal(size=(1, 1, n)),
        times=np.arange(n) / SFREQ,
        ch_names=("C3",),
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"),),
    )
    window = [Window("all", 0.0, (n - 1) / SFREQ)]
    expected = {
        ef.root_mean_square: "V",
        ef.skewness: "a.u.",
        ef.kurtosis: "a.u.",
        ef.line_length: "V/s",
        ef.zero_crossing_rate: "1/s",
        ef.amplitude_quantile: "V",
    }
    for function, unit in expected.items():
        assert function([signal], windows=window, include_global=False).meta[0].unit == unit
