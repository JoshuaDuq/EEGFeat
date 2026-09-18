import numpy as np
import pytest

from eegfeat.aperiodic import aperiodic
from eegfeat.spectra import Spectra, Window

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
