import numpy as np
import pytest

from eegfeat.spectra import (
    Spectra,
    Window,
    gradient_weights,
    support_restricted_mask,
    trapezoid_weights,
)
from eegfeat.table import ComputationSpec


def test_trapezoid_weights_sum_to_the_frequency_span() -> None:
    for freqs in (np.linspace(1.0, 45.0, 89), np.logspace(0.0, 2.0, 40)):
        assert trapezoid_weights(freqs).sum() == pytest.approx(freqs[-1] - freqs[0])


def test_trapezoid_weighting_recovers_an_analytic_integral_on_a_log_grid() -> None:
    # A plain mean cannot do this: the log grid over-samples low frequencies.
    freqs = np.logspace(np.log10(1.0), np.log10(100.0), 40)
    power = 2.0 * freqs  # integral over [1, 100] is 100^2 - 1^2 = 9999
    weights = trapezoid_weights(freqs)
    assert (power * weights).sum() == pytest.approx(9999.0, rel=1e-3)


def test_single_frequency_gets_unit_weight() -> None:
    assert trapezoid_weights(np.array([10.0])).tolist() == [1.0]
    assert gradient_weights(np.array([10.0])).tolist() == [1.0]


def test_the_two_weightings_differ_only_at_the_endpoints() -> None:
    freqs = np.linspace(1.0, 10.0, 10)
    trap, grad = trapezoid_weights(freqs), gradient_weights(freqs)
    np.testing.assert_allclose(trap[1:-1], grad[1:-1])
    assert trap[0] == pytest.approx(grad[0] / 2.0)
    assert trap[-1] == pytest.approx(grad[-1] / 2.0)


def test_gradient_weights_are_uniform_on_a_uniform_grid() -> None:
    # This is what makes a flat spectrum have exactly maximal entropy.
    assert len(set(gradient_weights(np.linspace(1.0, 10.0, 10)).round(12))) == 1


def test_window_requires_ordered_bounds() -> None:
    with pytest.raises(ValueError):
        Window("stim", 5.0, 5.0)
    with pytest.raises(ValueError):
        Window("", 0.0, 1.0)
    assert Window("all", -np.inf, np.inf).name == "all"


def test_from_epochs_spectrum_gains_a_singleton_window_axis() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4"], 200.0, "eeg")
    epochs = mne.EpochsArray(
        np.random.RandomState(0).randn(5, 2, 400) * 1e-6, info, tmin=-1.0, verbose="ERROR"
    )
    spectrum = epochs.compute_psd(
        "multitaper", fmin=2.0, fmax=40.0, normalization="full", verbose="ERROR"
    )
    spectra = Spectra.from_spectrum(
        spectrum,
        recording="test",
        estimator_parameters={"bandwidth": None, "normalization": "full"},
    )
    assert spectra.data.ndim == 4
    assert spectra.data.shape[0] == 5
    assert spectra.data.shape[1] == 2
    assert spectra.data.shape[2] == 1
    assert spectra.ch_names == ("C3", "C4")
    assert spectra.windows[0].name == "all"
    assert spectra.source == "multitaper"
    assert spectra.coverage.shape == spectra.data.shape
    assert (spectra.coverage == 1.0).all()


def test_from_continuous_spectrum_gains_epoch_and_window_axes() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4"], 200.0, "eeg")
    raw = mne.io.RawArray(np.random.RandomState(0).randn(2, 4000) * 1e-6, info, verbose="ERROR")
    spectrum = raw.compute_psd("welch", fmin=2.0, fmax=40.0, verbose="ERROR")
    spectra = Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={"n_fft": 400})
    assert spectra.data.shape[0] == 1
    assert spectra.data.shape[2] == 1


@pytest.mark.parametrize("parameters", [{}, {"normalization": "length"}])
def test_multitaper_spectrum_requires_explicit_density_normalization(parameters) -> None:
    import mne

    epochs = mne.EpochsArray(
        np.random.default_rng(0).normal(size=(2, 1, 400)),
        mne.create_info(["Cz"], 100.0, "eeg"),
        verbose=False,
    )
    spectrum = epochs.compute_psd("multitaper", verbose=False)
    with pytest.raises(ValueError, match="normalization.*full"):
        Spectra.from_spectrum(spectrum, recording="test", estimator_parameters=parameters)


def test_complex_fourier_coefficients_cannot_be_interpreted_as_power() -> None:
    from types import SimpleNamespace

    spectrum = SimpleNamespace(
        get_data=lambda: np.ones((1, 3), dtype=complex) * (1 + 2j),
        freqs=np.array([1.0, 2.0, 3.0]),
        ch_names=["Cz"],
        method="welch",
    )
    with pytest.raises(ValueError, match="complex.*power"):
        Spectra.from_spectrum(spectrum, recording="test", estimator_parameters={})


def test_non_finite_input_lowers_coverage_rather_than_raising() -> None:
    data = np.ones((2, 2, 1, 4))
    data[0, 0, 0, 1] = np.nan
    spectra = Spectra(
        data=data,
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
        ch_names=("C3", "C4"),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="test",
        representation="psd",
        support=np.ones(data.shape),
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
        computation=ComputationSpec.create("test"),
    )
    assert spectra.coverage[0, 0, 0, 1] == 0.0
    assert spectra.coverage.sum() == 15.0


def test_shape_and_axis_mismatches_raise() -> None:
    good = dict(
        data=np.ones((2, 2, 1, 4)),
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
        ch_names=("C3", "C4"),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones((2, 2, 1, 4)),
        source="test",
        representation="psd",
        support=np.ones((2, 2, 1, 4)),
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
        computation=ComputationSpec.create("test"),
    )
    with pytest.raises(ValueError, match="ch_names"):
        Spectra(**{**good, "ch_names": ("C3",)})
    with pytest.raises(ValueError, match="freqs"):
        Spectra(**{**good, "freqs": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="ascending"):
        Spectra(**{**good, "freqs": np.array([4.0, 3.0, 2.0, 1.0])})
    with pytest.raises(ValueError, match="coverage"):
        Spectra(**{**good, "coverage": np.ones((2, 2, 1, 3))})
    with pytest.raises(ValueError, match="support"):
        Spectra(**{**good, "support": np.ones((2, 2, 1, 3))})


@pytest.mark.parametrize("field", ["coverage", "support"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_spectral_fractions_must_be_finite_and_bounded(field: str, invalid: float) -> None:
    good = dict(
        data=np.ones((1, 1, 1, 2)),
        freqs=np.array([1.0, 2.0]),
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones((1, 1, 1, 2)),
        source="test",
        representation="psd",
        support=np.ones((1, 1, 1, 2)),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )
    values = good[field].copy()
    values[0, 0, 0, 0] = invalid

    with pytest.raises(ValueError, match=rf"{field} must contain finite values in \[0, 1\]"):
        Spectra(**{**good, field: values})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("freqs", np.array([1.0, np.nan]), "finite"),
        ("freqs", np.array([-1.0, 2.0]), "non-negative"),
        ("data", np.array([[[[-1.0, 1.0]], [[1.0, 1.0]]]]), "negative power"),
        ("ch_names", ("C3", "C3"), "unique"),
    ],
)
def test_spectral_domain_invariants_raise(field: str, value: object, message: str) -> None:
    good = dict(
        data=np.ones((1, 2, 1, 2)),
        freqs=np.array([1.0, 2.0]),
        ch_names=("C3", "C4"),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.ones((1, 2, 1, 2)),
        source="test",
        representation="psd",
        support=np.ones((1, 2, 1, 2)),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )

    with pytest.raises(ValueError, match=message):
        Spectra(**{**good, field: value})


def test_spectra_require_nonempty_axes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Spectra(
            data=np.empty((1, 1, 1, 0)),
            freqs=np.empty(0),
            ch_names=("C3",),
            windows=(Window("all", -np.inf, np.inf),),
            coverage=np.empty((1, 1, 1, 0)),
            source="test",
            representation="psd",
            support=np.empty((1, 1, 1, 0)),
            row_ids=(("test", 0, "event"),),
            computation=ComputationSpec.create("test"),
        )


def _toy_tfr(n_epochs: int = 4):
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3", "C4"], 200.0, "eeg")
    rng = np.random.RandomState(0)
    epochs = mne.EpochsArray(rng.randn(n_epochs, 2, 800) * 1e-6, info, tmin=-2.0, verbose="ERROR")
    return epochs.compute_tfr(
        "morlet",
        freqs=np.array([8.0, 10.0, 12.0]),
        n_cycles=3.0,
        return_itc=False,
        verbose="ERROR",
    )


def test_from_tfr_produces_one_spectrum_per_window() -> None:
    windows = (Window("base", -2.0, -1.0), Window("stim", 0.0, 1.0))
    spectra = Spectra.from_tfr(_toy_tfr(), windows, recording="test", n_cycles=3.0)
    assert spectra.data.shape == (4, 2, 2, 3)
    assert tuple(w.name for w in spectra.windows) == ("base", "stim")
    assert spectra.source == "morlet"


def test_from_tfr_window_mean_equals_a_manual_mean_over_the_time_mask() -> None:
    tfr = _toy_tfr()
    window = Window("stim", 0.0, 1.0)
    spectra = Spectra.from_tfr(tfr, (window,), recording="test", n_cycles=3.0)
    times = np.asarray(tfr.times)
    mask = support_restricted_mask(times, np.asarray(tfr.freqs), window, 3.0)
    data = np.asarray(tfr.get_data())
    expected = np.stack(
        [data[:, :, index, row].mean(axis=2) for index, row in enumerate(mask)], axis=2
    )
    np.testing.assert_allclose(spectra.data[:, :, 0, :], expected)


def test_from_tfr_refuses_an_already_baselined_tfr() -> None:
    tfr = _toy_tfr().apply_baseline((-2.0, -1.0), mode="logratio", verbose="ERROR")
    with pytest.raises(ValueError, match="already baseline"):
        Spectra.from_tfr(tfr, (Window("stim", 0.0, 1.0),), recording="test", n_cycles=3.0)


def test_from_tfr_rejects_a_window_outside_the_time_axis() -> None:
    with pytest.raises(ValueError, match="no samples"):
        Spectra.from_tfr(_toy_tfr(), (Window("late", 30.0, 40.0),), recording="test", n_cycles=3.0)


def test_from_tfr_requires_at_least_one_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        Spectra.from_tfr(_toy_tfr(), (), recording="test", n_cycles=3.0)


def test_from_tfr_rejects_complex_output() -> None:
    tfr = _toy_tfr()
    complex_tfr = type(
        "T",
        (),
        {
            "baseline": None,
            "times": np.asarray(tfr.times),
            "freqs": np.asarray(tfr.freqs),
            "ch_names": list(tfr.ch_names),
            "method": "morlet",
            "get_data": lambda self: np.asarray(tfr.get_data(), dtype=complex),
        },
    )()
    with pytest.raises(ValueError, match="complex"):
        Spectra.from_tfr(
            complex_tfr,
            (Window("stim", 0.0, 1.0),),
            recording="test",
            n_cycles=3.0,
        )


def test_support_restriction_narrows_low_frequencies_more_than_high_ones() -> None:
    times = np.linspace(-2.0, 2.0, 401)
    freqs = np.array([4.0, 40.0])
    mask = support_restricted_mask(times, freqs, Window("stim", 0.0, 1.0), n_cycles=6.0)
    # MNE extends to five Gaussian standard deviations, where
    # sigma_t = n_cycles / (2*pi*f).
    assert mask.shape == (2, 401)
    assert mask[0].sum() < mask[1].sum()
    expected_half_support = 5.0 * 6.0 / (2.0 * np.pi * 40.0)
    assert times[mask[1]].min() == pytest.approx(expected_half_support, abs=0.01)
    assert times[mask[1]].max() == pytest.approx(1.0 - expected_half_support, abs=0.01)


def test_a_frequency_whose_support_never_fits_drops_out_entirely() -> None:
    times = np.linspace(-2.0, 2.0, 401)
    freqs = np.array([1.0, 40.0])
    # at 1 Hz half support is 3 s, far wider than the 1 s window
    mask = support_restricted_mask(times, freqs, Window("stim", 0.0, 1.0), n_cycles=6.0)
    assert not mask[0].any()
    assert mask[1].any()


def test_frequencies_drop_out_of_a_window_individually() -> None:
    # MNE's five-sigma half supports are approximately 0.298 / 0.239 / 0.199 s.
    # A window of half-width 0.22 s therefore holds only 12 Hz.
    tfr = _toy_tfr()
    spectra = Spectra.from_tfr(tfr, (Window("narrow", 0.0, 0.44),), recording="test", n_cycles=3.0)
    assert np.isnan(spectra.data[:, :, 0, 0]).all()
    assert np.isnan(spectra.data[:, :, 0, 1]).all()
    assert np.isfinite(spectra.data[:, :, 0, 2]).all()
    assert (spectra.coverage[:, :, 0, :2] == 0.0).all()
    assert (spectra.coverage[:, :, 0, 2] > 0.0).all()


def test_a_window_narrower_than_every_wavelet_raises() -> None:
    # Every half-support exceeds this window's half-width, so nothing survives.
    # That is a specification error, not a data condition: the message says so.
    tfr = _toy_tfr()
    with pytest.raises(ValueError, match="retains no coefficients"):
        Spectra.from_tfr(tfr, (Window("tiny", 0.0, 0.1),), recording="test", n_cycles=3.0)


def test_from_tfr_requires_the_morlet_cycle_count() -> None:
    with pytest.raises(TypeError, match="n_cycles"):
        Spectra.from_tfr(_toy_tfr(), (Window("stim", 0.0, 1.0),), recording="test")


def test_support_fraction_is_distinct_from_finite_coverage() -> None:
    tfr = _toy_tfr()
    spectra = Spectra.from_tfr(tfr, (Window("stim", 0.0, 1.0),), recording="test", n_cycles=3.0)

    assert (spectra.coverage == 1.0).all()
    assert (spectra.support < 1.0).all()
    assert (spectra.support > 0.0).all()


def test_event_immediately_outside_window_cannot_affect_retained_coefficients() -> None:
    mne = pytest.importorskip("mne")
    sfreq = 200.0
    info = mne.create_info(["C3"], sfreq, "eeg")
    data = np.zeros((2, 1, 401))
    times = -1.0 + np.arange(data.shape[-1]) / sfreq
    data[1, 0, np.flatnonzero(times < 0.0)[-1]] = 1.0
    epochs = mne.EpochsArray(data, info, tmin=-1.0, verbose="ERROR")
    tfr = epochs.compute_tfr(
        "morlet",
        freqs=np.array([10.0]),
        n_cycles=3.0,
        return_itc=False,
        verbose="ERROR",
    )

    spectra = Spectra.from_tfr(tfr, (Window("target", 0.0, 0.6),), recording="test", n_cycles=3.0)

    np.testing.assert_allclose(spectra.data[1], spectra.data[0], atol=1e-14)


def test_from_spectrum_accepts_method_in_estimator_parameters() -> None:
    mne = pytest.importorskip("mne")
    info = mne.create_info(["C3"], 200.0, "eeg")
    epochs = mne.EpochsArray(
        np.random.RandomState(0).randn(2, 1, 400) * 1e-6, info, tmin=-1.0, verbose="ERROR"
    )
    spectrum = epochs.compute_psd("welch", fmin=2.0, fmax=40.0, verbose="ERROR")
    spectra = Spectra.from_spectrum(
        spectrum,
        recording="test",
        estimator_parameters={"method": "welch", "fmin": 2.0, "fmax": 40.0},
    )
    assert spectra.source == "welch"
    assert spectra.computation.method == "welch"
