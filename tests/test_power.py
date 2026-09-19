from dataclasses import replace

import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.power import integrated_band_power, mean_psd, mean_tfr_power
from eegfeat.spectra import Spectra, Window
from eegfeat.table import ComputationSpec

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
        representation="psd",
        support=np.ones(data.shape),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )


def test_flat_power_returns_that_value() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    table = mean_psd(
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
    freqs = np.r_[8.0, np.logspace(np.log10(8.5), np.log10(12.5), 9), 13.0]
    power = 2.0 * freqs
    table = mean_psd(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(ALPHA.fmin + ALPHA.fmax, rel=1e-12)
    assert table.values.item() != pytest.approx(power.mean(), rel=1e-4)


def test_integration_uses_the_exact_requested_boundaries() -> None:
    freqs = np.array([7.0, 9.0, 12.0, 14.0])
    power = 2.0 * freqs
    table = integrated_band_power(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    expected = ALPHA.fmax**2 - ALPHA.fmin**2
    assert table.values.item() == pytest.approx(expected, rel=1e-12)


def test_integrated_power_is_stable_across_linear_and_log_grids() -> None:
    linear = np.linspace(7.0, 14.0, 31)
    logarithmic = np.geomspace(7.0, 14.0, 47)
    results = [
        integrated_band_power(
            _spectra(3.0 * axis + 2.0, axis), bands=(ALPHA,), include_global=False
        ).values.item()
        for axis in (linear, logarithmic)
    ]
    expected = 1.5 * (ALPHA.fmax**2 - ALPHA.fmin**2) + 2.0 * (ALPHA.fmax - ALPHA.fmin)
    np.testing.assert_allclose(results, expected, rtol=1e-12)


def test_nan_frequencies_are_excluded_rather_than_poisoning_the_band() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    power = np.full(freqs.size, 4.0)
    power[2] = np.nan
    table = mean_psd(_spectra(power, freqs), bands=(ALPHA,), include_global=False)
    assert table.values.item() == pytest.approx(4.0)
    assert table.coverage.item() < 1.0


def test_an_all_nan_band_yields_nan_and_zero_coverage() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    table = mean_psd(
        _spectra(np.full(freqs.size, np.nan), freqs), bands=(ALPHA,), include_global=False
    )
    assert np.isnan(table.values).all()
    assert table.coverage.item() == 0.0


def test_unit_and_normalization_are_recorded() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    table = mean_psd(
        _spectra(np.full(freqs.size, 3.0), freqs), bands=(ALPHA,), include_global=False
    )
    assert table.meta[0].normalization == "raw"
    assert table.meta[0].unit == "V^2/Hz"
    assert table.meta[0].measure == "mean_psd"


def test_psd_and_tfr_reductions_reject_the_wrong_representation() -> None:
    freqs = np.arange(8.0, 13.5, 0.5)
    psd = _spectra(np.ones(freqs.size), freqs)
    tfr = replace(psd, representation="time_frequency_power")
    with pytest.raises(ValueError, match="requires spectral representation"):
        mean_psd(tfr, bands=(ALPHA,))
    with pytest.raises(ValueError, match="requires spectral representation"):
        integrated_band_power(tfr, bands=(ALPHA,))
    assert np.isfinite(mean_tfr_power(tfr, bands=(ALPHA,)).values).all()
