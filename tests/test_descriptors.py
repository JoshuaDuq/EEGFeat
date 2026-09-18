import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.descriptors import (
    peak_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_edge,
    spectral_entropy,
)
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


def _gaussian(freqs: np.ndarray, centre: float, width: float = 0.8) -> np.ndarray:
    return np.exp(-0.5 * ((freqs - centre) / width) ** 2)


def test_recovers_a_known_peak_to_better_than_the_bin_spacing() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(_gaussian(freqs, 10.2), freqs), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx(10.2, abs=0.1)


def test_peak_on_a_band_edge_is_flagged() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    rising = _spectra(np.linspace(1.0, 5.0, freqs.size), freqs)
    table = peak_frequency(rising, band=ALPHA, include_global=False)
    assert table.flags["edge_hit"].all()


def test_an_interior_peak_is_not_flagged() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(_gaussian(freqs, 10.2), freqs), band=ALPHA, include_global=False
    )
    assert not table.flags["edge_hit"].any()


def test_resolution_is_reported_on_the_column() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(_gaussian(freqs, 10.2), freqs), band=ALPHA, include_global=False
    )
    assert table.meta[0].freq_resolution_hz == pytest.approx(0.5)
    assert table.meta[0].unit == "Hz"


def test_a_grid_too_coarse_for_an_interior_peak_raises() -> None:
    # The pipeline's Morlet grid puts only 4 bins in alpha; two is below the
    # definition domain of the estimator and must not silently return a number.
    freqs = np.array([8.0, 11.0, 14.0])
    with pytest.raises(ValueError, match="3 bins"):
        peak_frequency(_spectra(np.array([1.0, 2.0, 1.0]), freqs), band=ALPHA, include_global=False)


def test_all_nan_input_yields_nan_without_raising() -> None:
    freqs = np.arange(8.0, 13.0, 0.5)
    table = peak_frequency(
        _spectra(np.full(freqs.size, np.nan), freqs), band=ALPHA, include_global=False
    )
    assert np.isnan(table.values).all()


UNIFORM = np.arange(8.0, 13.0, 0.25)


def test_centroid_of_a_flat_band_is_its_midpoint() -> None:
    table = spectral_centroid(
        _spectra(np.ones(UNIFORM.size), UNIFORM), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx((UNIFORM[0] + UNIFORM[-1]) / 2.0)


def test_centroid_follows_where_the_power_is() -> None:
    table = spectral_centroid(
        _spectra(_gaussian(UNIFORM, 9.0, 0.3), UNIFORM), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx(9.0, abs=0.1)


def test_bandwidth_of_a_flat_band_matches_a_uniform_distribution() -> None:
    # The discrete population std of n equally spaced bins, h*sqrt((n^2-1)/12).
    # NOT the continuous span/sqrt(12): for n = 20 those differ by 5%, because the
    # estimator sums over bins rather than integrating over the band.
    table = spectral_bandwidth(
        _spectra(np.ones(UNIFORM.size), UNIFORM), band=ALPHA, include_global=False
    )
    expected = np.sqrt(np.mean((UNIFORM - UNIFORM.mean()) ** 2))
    assert table.values.item() == pytest.approx(expected, rel=1e-12)


def test_bandwidth_is_smaller_for_a_narrower_peak() -> None:
    narrow = spectral_bandwidth(
        _spectra(_gaussian(UNIFORM, 10.5, 0.2), UNIFORM), band=ALPHA, include_global=False
    )
    wide = spectral_bandwidth(
        _spectra(_gaussian(UNIFORM, 10.5, 1.0), UNIFORM), band=ALPHA, include_global=False
    )
    assert narrow.values.item() < wide.values.item()


def test_entropy_of_a_flat_band_is_exactly_one() -> None:
    table = spectral_entropy(
        _spectra(np.ones(UNIFORM.size), UNIFORM), band=ALPHA, include_global=False
    )
    assert table.values.item() == pytest.approx(1.0)


def test_entropy_of_a_single_occupied_bin_is_zero() -> None:
    power = np.zeros(UNIFORM.size)
    power[5] = 1.0
    table = spectral_entropy(_spectra(power, UNIFORM), band=ALPHA, include_global=False)
    assert table.values.item() == pytest.approx(0.0, abs=1e-12)


def test_edge_at_half_of_a_flat_band_is_near_its_midpoint() -> None:
    table = spectral_edge(
        _spectra(np.ones(UNIFORM.size), UNIFORM),
        band=ALPHA,
        percentile=0.5,
        include_global=False,
    )
    midpoint = (UNIFORM[0] + UNIFORM[-1]) / 2.0
    assert abs(table.values.item() - midpoint) <= 0.25


def test_edge_returns_a_grid_frequency_because_it_does_not_interpolate() -> None:
    table = spectral_edge(
        _spectra(np.ones(UNIFORM.size), UNIFORM),
        band=ALPHA,
        percentile=0.5,
        include_global=False,
    )
    assert table.values.item() in set(UNIFORM.tolist())


@pytest.mark.parametrize("percentile", [0.0, -0.1, 1.5, np.nan])
def test_an_out_of_range_percentile_raises(percentile: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        spectral_edge(
            _spectra(np.ones(UNIFORM.size), UNIFORM),
            band=ALPHA,
            percentile=percentile,
            include_global=False,
        )


def test_an_empty_band_yields_nan_for_every_descriptor() -> None:
    dead = _spectra(np.zeros(UNIFORM.size), UNIFORM)
    for fn in (spectral_centroid, spectral_bandwidth, spectral_entropy):
        assert np.isnan(fn(dead, band=ALPHA, include_global=False).values).all()
