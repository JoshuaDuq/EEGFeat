import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.descriptors import peak_frequency
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
