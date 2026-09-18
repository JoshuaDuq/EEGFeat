import numpy as np

from eegfeat.qc import band_coverage


def test_full_coverage_stays_one() -> None:
    coverage = np.ones((2, 3, 2, 4))
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(band_coverage(coverage, weights), np.ones((2, 3, 2)))


def test_coverage_is_weighted_by_bin_width_not_bin_count() -> None:
    # One wide bin fully covered, three narrow bins empty: count says 0.25, width says 0.8.
    coverage = np.array([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 4)
    weights = np.array([8.0, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(band_coverage(coverage, weights), [[[0.8]]])


def test_a_dropped_frequency_lowers_coverage_proportionally() -> None:
    coverage = np.array([1.0, 1.0, 1.0, 0.0]).reshape(1, 1, 1, 4)
    weights = np.ones(4)
    np.testing.assert_allclose(band_coverage(coverage, weights), [[[0.75]]])


def test_all_frequencies_dropped_gives_zero_not_nan() -> None:
    coverage = np.zeros((1, 1, 1, 4))
    result = band_coverage(coverage, np.ones(4))
    assert result.item() == 0.0


def test_zero_total_weight_gives_zero_rather_than_dividing_by_zero() -> None:
    result = band_coverage(np.ones((1, 1, 1, 2)), np.zeros(2))
    assert result.item() == 0.0
