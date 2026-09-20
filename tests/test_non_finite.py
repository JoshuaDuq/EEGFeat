"""A non-finite sample is missing data, whichever non-finite value it happens to be.

``Signal.coverage`` and every kernel's ``_finite`` guard decide usability with
``np.isfinite``, so NaN and infinity are already the same thing to this package.
The nan-aware reductions disagree: ``np.nanmean`` skips NaN and averages in
infinity, and scipy's ``nan_policy="omit"`` omits only NaN. These tests pin the
two to the same answer, which is the one the coverage contract already promises.
"""

import numpy as np
import pytest
from scipy.signal import hilbert

import eegfeat as ef
from eegfeat.bands import Band
from eegfeat.signal import BandSignal, Signal
from eegfeat.spectra import Window

SFREQ = 250.0
N = 600
BAND = Band("alpha", 8.0, 13.0)
WINDOW = Window("all", 0.0, (N - 1) / SFREQ)
POST = Window("post", 1.0, (N - 1) / SFREQ)
PRE = Window("pre", 0.0, 0.9)
# The sample that goes bad. Interior, so it is not an edge case of any window.
SPOILED = (0, 0, N // 2)
IN_BASELINE = (0, 0, 100)


def _real() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(N) / SFREQ
    wave = np.sin(2 * np.pi * 10.0 * t) + 0.3 * np.sin(2 * np.pi * 22.0 * t)
    return wave + 0.5 * rng.normal(size=(3, 2, N))


def _signal(data: np.ndarray) -> Signal:
    return Signal.from_arrays(
        data=data,
        times=np.arange(N) / SFREQ,
        ch_names=("C3", "C4"),
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(data.shape[0])),
    )


def _band(data: np.ndarray) -> BandSignal:
    return BandSignal.from_arrays(
        analytic=data,
        times=np.arange(N) / SFREQ,
        ch_names=("C3", "C4"),
        band=BAND,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(data.shape[0])),
    )


def _spoiled(base: np.ndarray, value: float, where: tuple[int, int, int] = SPOILED) -> np.ndarray:
    out = base.copy()
    out[where] = value
    return out


def _values(function, wrap, base, bad, kwargs) -> tuple[np.ndarray, np.ndarray]:
    reference = function([wrap(_spoiled(base, np.nan))], **kwargs).values
    measured = function([wrap(_spoiled(base, bad))], **kwargs).values
    return np.asarray(reference, dtype=float), np.asarray(measured, dtype=float)


TIME_MEASURES = [
    (ef.mean_amplitude, {}),
    (ef.variance, {}),
    (ef.peak_to_peak, {}),
    (ef.root_mean_square, {}),
    (ef.line_length, {}),
    (ef.hjorth_mobility, {}),
    (ef.hjorth_complexity, {}),
    (ef.skewness, {}),
    (ef.kurtosis, {}),
    (ef.zero_crossing_rate, {}),
    (ef.amplitude_quantile, {}),
    (ef.amplitude_quantile, {"q": 0.95}),
    (ef.area_under_curve, {}),
    (ef.higuchi_fractal_dimension, {}),
    (ef.peak_amplitude, {"polarity": "positive"}),
    (ef.peak_latency, {"polarity": "negative"}),
]


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
@pytest.mark.parametrize(
    ("function", "kwargs"),
    TIME_MEASURES,
    ids=[f"{f.__name__}{sorted(k.items())}" for f, k in TIME_MEASURES],
)
def test_an_infinite_sample_reads_the_same_as_a_missing_one(function, kwargs, bad) -> None:
    reference, measured = _values(
        function, _signal, _real(), bad, {"windows": [WINDOW], "include_global": True, **kwargs}
    )
    assert np.allclose(reference, measured, equal_nan=True, rtol=1e-9), {
        "nan": reference,
        "inf": measured,
    }


BAND_MEASURES = [
    (ef.erds_mean, {"baseline": PRE}),
    (ef.erds_slope, {"baseline": PRE}),
    (ef.erd_magnitude, {"baseline": PRE}),
    (ef.ers_magnitude, {"baseline": PRE}),
    (ef.erd_duration, {"baseline": PRE}),
    (ef.erds_peak_latency, {"baseline": PRE}),
    (ef.burst_count, {}),
    (ef.burst_rate, {}),
    (ef.burst_duration, {}),
    (ef.burst_amplitude, {}),
    (ef.fraction_above_threshold, {}),
    (ef.itpc, {}),
    (ef.ppc, {}),
]


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
@pytest.mark.parametrize(
    ("function", "kwargs"), BAND_MEASURES, ids=[f.__name__ for f, _ in BAND_MEASURES]
)
def test_a_band_measure_treats_infinity_as_missing_too(function, kwargs, bad) -> None:
    analytic = np.asarray(hilbert(_real(), axis=2))
    reference, measured = _values(
        function, _band, analytic, bad, {"windows": [POST], "include_global": True, **kwargs}
    )
    assert np.allclose(reference, measured, equal_nan=True, rtol=1e-9), {
        "nan": reference,
        "inf": measured,
    }


def test_an_infinite_channel_value_does_not_poison_its_roi() -> None:
    # The spatial aggregate is the last place a non-finite value can spread: one
    # channel's infinity would otherwise become every ROI it belongs to.
    from eegfeat.groups import aggregate

    values = np.array([[[1.0], [3.0], [np.inf]]])
    coverage = np.ones_like(values)
    units = aggregate(values, coverage, ("C3", "C4", "Cz"), {"roi": ("C3", "Cz")}, True)
    by_name = {unit.space: float(unit.values[0, 0]) for unit in units}
    assert by_name["roi"] == pytest.approx(1.0)
    assert by_name["global"] == pytest.approx(2.0)


def test_the_fraction_above_threshold_is_a_share_of_the_samples_that_exist() -> None:
    # A missing sample is not a sample that stayed below threshold. Counting it in
    # the denominator dilutes the fraction by however much data went missing.
    envelope = np.zeros((1, 1, 100))
    envelope[0, 0, :40] = 10.0  # 40 of 100 above any threshold between 0 and 10
    envelope[0, 0, 40:] = 1.0
    window = [Window("all", 0.0, 99 / SFREQ)]
    level = np.full((1, 1), 5.0)

    def fraction(data: np.ndarray) -> float:
        signal = BandSignal.from_arrays(
            analytic=np.asarray(data, dtype=complex),
            times=np.arange(100) / SFREQ,
            ch_names=("C3",),
            band=BAND,
            sfreq=SFREQ,
            row_ids=(("test", 0, "event"),),
        )
        table = ef.fraction_above_threshold(
            [signal], windows=window, threshold=level, min_duration_ms=0.0, include_global=False
        )
        return float(table.values[0, 0])

    assert fraction(envelope) == pytest.approx(0.40)
    spoiled = envelope.copy()
    spoiled[0, 0, 90:] = np.nan  # ten below-threshold samples lost, so 40 of 90
    assert fraction(spoiled) == pytest.approx(40 / 90)


def test_the_burst_threshold_is_calibrated_on_the_samples_that_exist() -> None:
    # The percentile threshold is an order statistic, so an infinity counted as a
    # sample pushes it up a rank and changes which samples are called bursts. A
    # ramp makes that shift exact instead of a fraction of a noisy distribution.
    ramp = np.arange(1.0, 101.0).reshape(1, 1, 100)
    window = [Window("all", 0.0, 99 / SFREQ)]

    def fraction(bad: float) -> float:
        data = ramp.copy()
        data[0, 0, 10] = bad
        signal = BandSignal.from_arrays(
            analytic=np.asarray(data, dtype=complex),
            times=np.arange(100) / SFREQ,
            ch_names=("C3",),
            band=BAND,
            sfreq=SFREQ,
            row_ids=(("test", 0, "event"),),
        )
        table = ef.fraction_above_threshold(
            [signal], windows=window, threshold=0.75, min_duration_ms=0.0, include_global=False
        )
        return float(table.values[0, 0])

    assert fraction(np.inf) == pytest.approx(fraction(np.nan))


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
@pytest.mark.parametrize(
    ("function", "kwargs"), BAND_MEASURES[:6], ids=[f.__name__ for f, _ in BAND_MEASURES[:6]]
)
def test_a_bad_baseline_sample_is_skipped_rather_than_discarding_the_channel(
    function, kwargs, bad
) -> None:
    # The baseline reference is read from the power directly, not through the
    # analysis trace, so it is the one place an infinity can still reach a mean.
    # There it costs more than a wrong number: a non-finite reference is called
    # degenerate, and every ERDS measure for that channel is withheld.
    analytic = np.asarray(hilbert(_real(), axis=2))
    call = {"windows": [POST], "include_global": True, **kwargs}
    reference = function([_band(_spoiled(analytic, np.nan, IN_BASELINE))], **call).values
    measured = function([_band(_spoiled(analytic, bad, IN_BASELINE))], **call).values
    assert np.allclose(reference, measured, equal_nan=True, rtol=1e-9), {
        "nan": reference,
        "inf": measured,
    }
    assert np.isfinite(reference).all(), "the nan reference itself should survive one bad sample"


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_csp_features_skip_a_bad_sample_rather_than_voiding_the_epoch(bad: float) -> None:
    # Fitting already drops a whole epoch that is not entirely finite, on purpose.
    # Projecting does not, so its variance has to skip the bad sample the same way
    # a NaN one is skipped, instead of turning the epoch's components into NaN.
    epochs, channels = 12, 4
    rng = np.random.default_rng(0)
    t = np.arange(400) / SFREQ
    data = np.sin(2 * np.pi * 10.0 * t) + 0.6 * rng.normal(size=(epochs, channels, t.size))
    labels = np.array([0, 1] * (epochs // 2))
    folds = [
        (np.setdiff1d(np.arange(epochs), np.arange(k, epochs, 3)), np.arange(k, epochs, 3))
        for k in range(3)
    ]

    def features(value: float) -> np.ndarray:
        spoiled = data.copy()
        spoiled[0, 0, 200] = value
        signal = Signal.from_arrays(
            data=spoiled,
            times=t,
            ch_names=tuple(f"C{i}" for i in range(channels)),
            sfreq=SFREQ,
            row_ids=tuple(("test", i, "event") for i in range(epochs)),
        )
        window = Window("all", 0.0, (t.size - 1) / SFREQ)
        return np.asarray(
            ef.csp_features(signal, labels, folds=folds, window=window, n_components=2).values,
            dtype=float,
        )

    reference = features(np.nan)
    assert np.isfinite(reference).all()
    assert np.allclose(features(bad), reference, equal_nan=True, rtol=1e-9)
