import numpy as np
import pytest

from eegfeat.bands import BANDS_STANDARD, Band


def test_mask_is_half_open_so_standard_bands_tile_without_overlap() -> None:
    freqs = np.array([7.9, 8.0, 12.9, 13.0])
    alpha = Band("alpha", 8.0, 13.0)
    assert alpha.mask(freqs).tolist() == [False, True, True, False]


def test_every_standard_frequency_belongs_to_exactly_one_band() -> None:
    freqs = np.arange(1.0, 45.0, 0.1)
    hits = np.sum([b.mask(freqs) for b in BANDS_STANDARD], axis=0)
    assert set(np.unique(hits).tolist()) == {1}


@pytest.mark.parametrize(
    ("name", "fmin", "fmax"),
    [("", 8.0, 13.0), ("alpha", 13.0, 8.0), ("alpha", 8.0, 8.0), ("alpha", -1.0, 13.0)],
)
def test_invalid_bounds_raise(name: str, fmin: float, fmax: float) -> None:
    with pytest.raises(ValueError):
        Band(name, fmin, fmax)


def test_band_is_frozen() -> None:
    with pytest.raises(AttributeError):
        Band("alpha", 8.0, 13.0).fmin = 9.0  # type: ignore[misc]
