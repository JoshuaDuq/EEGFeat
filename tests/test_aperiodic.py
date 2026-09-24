import numpy as np
import pytest
from scipy import stats

from eegfeat.aperiodic import aperiodic, aperiodic_ratio
from eegfeat.spectra import Spectra, Window
from eegfeat.table import ComputationSpec

# geomspace, not logspace: logspace lands the top bin on 40 Hz or just under it depending
# on the platform, which decides whether the half-open default fit_range keeps it.
FREQS = np.geomspace(2.0, 40.0, 60)
# A Welch grid starts at DC, where a power law has no value.
WELCH_FREQS = np.arange(0.0, 45.0, 0.5)


def _spectra(power: np.ndarray, freqs: np.ndarray = FREQS) -> Spectra:
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


def test_every_measure_is_returned_as_a_column_of_one_table() -> None:
    table = aperiodic(_spectra(10.0 * FREQS**-1.7), include_global=False)
    assert sorted(m.measure for m in table.meta) == ["offset", "r_squared", "slope"]
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


def _power_law_with_dc(dc: float = 1e-3) -> np.ndarray:
    positive = WELCH_FREQS > 0
    power = np.full(WELCH_FREQS.size, dc)
    power[positive] = 10.0 * WELCH_FREQS[positive] ** -1.7
    return power


def test_the_zero_hertz_bin_is_left_out_of_the_fit() -> None:
    # log10(0) is -inf: a DC bin inside the fit range breaks the least squares.
    table = aperiodic(
        _spectra(_power_law_with_dc(), WELCH_FREQS), fit_range=(0.0, 40.0), include_global=False
    )
    assert table.select(measure="slope").values.item() == pytest.approx(-1.7, abs=1e-9)


def test_whitening_leaves_the_zero_hertz_bin_out_of_its_fit() -> None:
    # Read as log f = 0, the DC bin would sit at 1 Hz, far below the line, and tilt it.
    spectra = _spectra(_power_law_with_dc(), WELCH_FREQS)
    whitened = aperiodic_ratio(spectra, fit_range=(0.0, 40.0))
    np.testing.assert_allclose(whitened.data[..., WELCH_FREQS > 0], 1.0, rtol=1e-9)


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


def test_a_clean_power_law_reports_a_near_perfect_fit() -> None:
    table = aperiodic(_spectra(10.0 * FREQS**-1.7), include_global=False)
    assert table.select(measure="r_squared").values.item() == pytest.approx(1.0, abs=1e-6)


def test_a_knee_is_visible_in_the_fit_quality() -> None:
    # A single line through a spectrum with a bend is a poor model, and the slope
    # alone cannot say so. Most real EEG has a knee inside the default (2, 40) Hz
    # range, so reporting an exponent without a fit statistic hides the failure.
    knee = 10.0 / (5.0**2 + FREQS**2)
    assert (
        aperiodic(_spectra(knee), include_global=False).select(measure="r_squared").values.item()
        < 0.99
    )


def test_the_fit_quality_is_dimensionless_and_named() -> None:
    meta = (
        aperiodic(_spectra(10.0 * FREQS**-1.7), include_global=False)
        .select(measure="r_squared")
        .meta[0]
    )
    assert meta.unit == "a.u."
    assert meta.band is None


def test_one_refit_round_rejects_the_peak_and_scores_the_line_it_fitted() -> None:
    # max_iterations counts refit rounds: fit, reject what sits above z * MAD, refit,
    # and score r_squared on the points of that last fit. A rejection that is not
    # followed by a refit would leave the slope blind to it and score the line on
    # points it never saw.
    clean = 10.0 * FREQS**-1.7
    peaked = clean + 3.0 * clean.max() * np.exp(-0.5 * ((FREQS - 10.0) / 1.0) ** 2)
    # fit_range is half-open like every band, so the 40 Hz bin is not fitted.
    in_range = FREQS < 40.0
    log_f, log_p = np.log10(FREQS[in_range]), np.log10(peaked[in_range])
    slope, offset = np.polyfit(log_f, log_p, 1)
    residuals = log_p - (offset + slope * log_f)
    mad = stats.median_abs_deviation(residuals, scale="normal")
    keep = residuals <= 2.5 * mad
    slope, offset = np.polyfit(log_f[keep], log_p[keep], 1)
    fitted = offset + slope * log_f[keep]
    observed = log_p[keep]
    r_squared = 1.0 - np.sum((observed - fitted) ** 2) / np.sum((observed - observed.mean()) ** 2)

    table = aperiodic(_spectra(peaked), include_global=False, max_iterations=1)
    assert table.select(measure="slope").values.item() == pytest.approx(slope)
    assert table.select(measure="offset").values.item() == pytest.approx(offset)
    assert table.select(measure="r_squared").values.item() == pytest.approx(r_squared)
