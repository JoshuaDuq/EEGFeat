import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.power import band_power
from eegfeat.spectra import Spectra, Window

ALPHA = Band("alpha", 8.0, 13.0)


def _spectra(power: np.ndarray, freqs: np.ndarray) -> Spectra:
    data = power.reshape(1, 1, 1, freqs.size)
    return Spectra(
        data=data,
        freqs=freqs,
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="test",
    )


def test_flat_power_returns_that_value() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = band_power(
        _spectra(np.full(freqs.size, 3.0), freqs), bands=(ALPHA,), include_global=False
    )
    np.testing.assert_allclose(table.values, 3.0)


def test_band_value_is_the_width_weighted_mean_not_the_bin_mean() -> None:
    # For P(f) = 2f the trapezoid rule is exact, so the width-weighted mean over
    # [f0, f1] is (f1^2 - f0^2) / (f1 - f0) = f0 + f1 on any grid at all. A plain
    # bin mean on a log grid is not, because it over-weights low frequencies.
    # Compare against that closed form rather than against np.trapezoid, which
    # does not exist below numpy 2.0 while this package supports numpy 1.26.
    # The grid sits strictly inside the band: np.logspace does not reproduce its
    # own endpoints exactly, so a grid ending on a band bound lands on the wrong
    # side of it by ~1e-15 and silently gains or loses a bin.
    freqs = np.logspace(np.log10(8.5), np.log10(12.5), 9)
    power = 2.0 * freqs
    table = band_power(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(freqs[0] + freqs[-1], rel=1e-12)
    assert table.values.item() != pytest.approx(power.mean(), rel=1e-4)


def test_a_bin_exactly_on_the_upper_bound_is_excluded() -> None:
    # Half-open bands tile without overlap, so 13.0 belongs to beta, not alpha.
    freqs = np.array([8.0, 10.0, 13.0])
    power = np.array([1.0, 1.0, 100.0])
    table = band_power(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(1.0)


def test_nan_frequencies_are_excluded_rather_than_poisoning_the_band() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    power = np.full(freqs.size, 4.0)
    power[2] = np.nan
    table = band_power(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(4.0)
    assert table.coverage.item() < 1.0


def test_an_all_nan_band_yields_nan_and_zero_coverage() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = band_power(
        _spectra(np.full(freqs.size, np.nan), freqs), bands=(ALPHA,), include_global=False
    )
    assert np.isnan(table.values).all()
    assert table.coverage.item() == 0.0


def test_unit_and_normalization_are_recorded() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = band_power(
        _spectra(np.full(freqs.size, 3.0), freqs), bands=(ALPHA,), include_global=False
    )
    assert table.meta[0].normalization == "raw"
    assert table.meta[0].unit == "V^2/Hz"
    assert table.meta[0].measure == "power"
