import numpy as np
import pytest

from eegfeat.groups import aggregate

CH = ("C3", "Cz", "C4")


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    return values, np.ones((2, 3, 2))


def test_without_groups_every_channel_becomes_its_own_unit() -> None:
    values, coverage = _inputs()
    units = aggregate(values, coverage, CH, groups=None, include_global=False)
    assert [u.space for u in units] == ["C3", "Cz", "C4"]
    assert all(u.space_kind == "channel" for u in units)
    np.testing.assert_allclose(units[1].values, values[:, 1, :])


def test_groups_average_their_member_channels() -> None:
    values, coverage = _inputs()
    units = aggregate(values, coverage, CH, groups={"central": ["C3", "C4"]}, include_global=False)
    assert [u.space for u in units] == ["central"]
    assert units[0].space_kind == "roi"
    np.testing.assert_allclose(units[0].values, values[:, [0, 2], :].mean(axis=1))


def test_global_is_the_mean_across_all_channels() -> None:
    values, coverage = _inputs()
    units = aggregate(values, coverage, CH, groups=None, include_global=True)
    assert units[-1].space == "global"
    assert units[-1].space_kind == "global"
    np.testing.assert_allclose(units[-1].values, values.mean(axis=1))


def test_nan_channels_are_ignored_rather_than_poisoning_the_group() -> None:
    values, coverage = _inputs()
    values[:, 0, :] = np.nan
    coverage[:, 0, :] = 0.0
    units = aggregate(values, coverage, CH, groups={"central": ["C3", "C4"]}, include_global=False)
    np.testing.assert_allclose(units[0].values, values[:, 2, :])
    np.testing.assert_allclose(units[0].coverage, np.full((2, 2), 0.5))


def test_a_group_whose_channels_are_all_nan_yields_nan() -> None:
    values, coverage = _inputs()
    values[:] = np.nan
    coverage[:] = 0.0
    units = aggregate(values, coverage, CH, groups={"all": list(CH)}, include_global=False)
    assert np.isnan(units[0].values).all()


def test_units_expose_the_channel_indices_behind_them() -> None:
    values, coverage = _inputs()
    units = aggregate(values, coverage, CH, groups={"central": ["C3", "C4"]}, include_global=True)
    assert units[0].picks == (0, 2)
    assert units[-1].picks == (0, 1, 2)


def test_unknown_channel_in_a_group_raises() -> None:
    values, coverage = _inputs()
    with pytest.raises(KeyError, match="Fz"):
        aggregate(values, coverage, CH, groups={"front": ["Fz"]}, include_global=False)


def test_empty_group_raises() -> None:
    values, coverage = _inputs()
    with pytest.raises(ValueError, match="no channels"):
        aggregate(values, coverage, CH, groups={"front": []}, include_global=False)
