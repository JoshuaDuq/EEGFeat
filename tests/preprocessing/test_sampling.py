import pytest

from eegtable.preprocessing.config import DecimationSettings, FilterSettings


def test_unsafe_decimation_fails():
    from eegtable.preprocessing.sampling import validate_sampling

    with pytest.raises(ValueError, match="transition"):
        validate_sampling(250, DecimationSettings(5), FilterSettings(h_freq=20), 0.0, 500)
