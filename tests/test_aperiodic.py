import numpy as np
import pytest

from eegfeat.aperiodic import aperiodic, aperiodic_ratio
from eegfeat.spectra import Spectra, Window
from eegfeat.table import ComputationSpec

FREQS = np.logspace(np.log10(2.0), np.log10(40.0), 60)


def _spectra(power: np.ndarray) -> Spectra:
    data = power.reshape(1, 1, 1, FREQS.size)
    return Spectra(
        data=data,
        freqs=FREQS,
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="test",
        representation="psd",
        support=np.ones(data.shape),
        row_ids=(("test", 0, "event"),),
        computation=ComputationSpec.create("test"),
    )


def test_recovers_a_known_exponent_from_a_synthetic_power_law() -> None:
    table = aperiodic(_spectra(10.0 * FREQS**-1.7), include_global=False)
    slope = table.select(measure="slope").values.item()
    assert slope == pytest.approx(-1.7, abs=0.02)


def test_recovers_the_offset() -> None:
    table = aperiodic(_spectra(10.0 * FREQS**-1.7), include_global=False)
    offset = table.select(measure="offset").values.item()
    assert offset == pytest.approx(np.log10(10.0), abs=0.05)


def test_an_alpha_peak_does_not_tilt_the_fit() -> None:
    clean = 10.0 * FREQS**-1.7
    peaked = clean + 3.0 * clean.max() * np.exp(-0.5 * ((FREQS - 10.0) / 1.0) ** 2)
    biased = np.polyfit(np.log10(FREQS), np.log10(peaked), 1)[0]
    robust = aperiodic(_spectra(peaked), include_global=False).select(measure="slope").values.item()
    assert abs(robust - (-1.7)) < abs(biased - (-1.7))


def test_both_measures_are_returned_as_columns_of_one_table() -> None:
    table = aperiodic(_spectra(10.0 * FREQS**-1.7), include_global=False)
    assert sorted(m.measure for m in table.meta) == ["offset", "slope"]
    assert all(m.band is None for m in table.meta)


def test_too_few_usable_points_yields_nan() -> None:
    power = np.full(FREQS.size, np.nan)
    power[:4] = 1.0
    table = aperiodic(_spectra(power), include_global=False)
    assert np.isnan(table.values).all()


def test_non_positive_power_is_excluded_rather_than_producing_neg_inf() -> None:
    power = 10.0 * FREQS**-1.7
    power[10] = 0.0
    table = aperiodic(_spectra(power), include_global=False)
    assert np.isfinite(table.values).all()


def test_fit_range_outside_the_axis_raises() -> None:
    with pytest.raises(ValueError, match="no frequencies"):
        aperiodic(_spectra(10.0 * FREQS**-1.7), fit_range=(100.0, 200.0), include_global=False)


@pytest.mark.parametrize("peak_rejection_z", [0.0, -1.0, np.nan, np.inf, True])
@pytest.mark.parametrize("measure", [aperiodic, aperiodic_ratio])
def test_peak_rejection_threshold_must_be_finite_and_positive(
    peak_rejection_z: float, measure: object
) -> None:
    with pytest.raises(ValueError, match="peak_rejection_z"):
        measure(_spectra(10.0 * FREQS**-1.7), peak_rejection_z=peak_rejection_z)


@pytest.mark.parametrize("max_iterations", [0, -1, 1.5, True])
@pytest.mark.parametrize("measure", [aperiodic, aperiodic_ratio])
def test_max_iterations_must_be_a_positive_integer(max_iterations: object, measure: object) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        measure(_spectra(10.0 * FREQS**-1.7), max_iterations=max_iterations)


# --- aperiodic_ratio -----------------------------------------------------------------


def test_a_pure_power_law_flattens_to_one() -> None:
    ratio = aperiodic_ratio(_spectra(10.0 * FREQS**-1.7))
    np.testing.assert_allclose(ratio.data, 1.0, rtol=1e-6)


def test_an_oscillation_survives_the_whitening_and_stands_above_one() -> None:
    background = 10.0 * FREQS**-1.7
    peaked = background * (1.0 + 1.5 * np.exp(-0.5 * ((FREQS - 10.0) / 1.0) ** 2))
    ratio = aperiodic_ratio(_spectra(peaked))
    at_peak = ratio.data[0, 0, 0, int(np.argmin(np.abs(FREQS - 10.0)))]
    assert at_peak > 2.0
    assert ratio.data[0, 0, 0, 0] == pytest.approx(1.0, abs=0.2)


def test_whitening_is_recorded_in_the_provenance() -> None:
    assert aperiodic_ratio(_spectra(10.0 * FREQS**-1.7)).source == "test+aperiodic_ratio"


def test_the_grid_and_coverage_are_carried_through_unchanged() -> None:
    spectra = _spectra(10.0 * FREQS**-1.7)
    ratio = aperiodic_ratio(spectra)
    np.testing.assert_array_equal(ratio.freqs, spectra.freqs)
    np.testing.assert_array_equal(ratio.coverage, spectra.coverage)
    assert ratio.windows == spectra.windows


def test_a_fit_range_holding_too_few_bins_raises() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        aperiodic_ratio(_spectra(10.0 * FREQS**-1.7), fit_range=(2.0, 2.2))


def test_a_cell_that_cannot_be_fitted_is_withheld() -> None:
    power = np.full(FREQS.size, np.nan)
    power[:4] = 1.0
    spectra = _spectra(power)
    ratio = aperiodic_ratio(spectra)
    assert np.isnan(ratio.data).all()
    assert ratio.flags["aperiodic_fit_failed"].all()
