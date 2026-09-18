import numpy as np
import pytest

from eegfeat.spectra import Spectra, Window, gradient_weights, trapezoid_weights


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
    spectrum = epochs.compute_psd("multitaper", fmin=2.0, fmax=40.0, verbose="ERROR")
    spectra = Spectra.from_spectrum(spectrum)
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
    spectra = Spectra.from_spectrum(spectrum)
    assert spectra.data.shape[0] == 1
    assert spectra.data.shape[2] == 1


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
    )
    with pytest.raises(ValueError, match="ch_names"):
        Spectra(**{**good, "ch_names": ("C3",)})
    with pytest.raises(ValueError, match="freqs"):
        Spectra(**{**good, "freqs": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="ascending"):
        Spectra(**{**good, "freqs": np.array([4.0, 3.0, 2.0, 1.0])})
    with pytest.raises(ValueError, match="coverage"):
        Spectra(**{**good, "coverage": np.ones((2, 2, 1, 3))})
