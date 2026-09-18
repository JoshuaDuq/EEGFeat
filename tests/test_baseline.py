import numpy as np
import pytest

from eegfeat.baseline import EPS, normalize


def _values() -> np.ndarray:
    return np.array([[[1.0, 10.0, 100.0]]])  # (1 epoch, 1 channel, 3 windows)


def test_raw_passes_values_through() -> None:
    np.testing.assert_allclose(normalize(_values(), baseline=None, mode="raw"), _values())


def test_log10_needs_no_baseline() -> None:
    out = normalize(_values(), baseline=None, mode="log10")
    np.testing.assert_allclose(out, [[[0.0, 1.0, 2.0]]])


def test_log_ratio_against_itself_is_exactly_zero() -> None:
    values = _values()
    baseline = values[:, :, 0]
    out = normalize(values, baseline=baseline, mode="log_ratio")
    assert out[0, 0, 0] == 0.0


def test_db_is_exactly_ten_times_log_ratio() -> None:
    values, baseline = _values(), _values()[:, :, 0]
    ratio = normalize(values, baseline=baseline, mode="log_ratio")
    db = normalize(values, baseline=baseline, mode="db")
    np.testing.assert_allclose(db, 10.0 * ratio)


def test_zero_baseline_is_floored_symmetrically_rather_than_dividing_by_zero() -> None:
    values = np.array([[[1.0]]])
    out = normalize(values, baseline=np.array([[0.0]]), mode="log_ratio")
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, np.log10(1.0 / EPS))


def test_non_finite_baseline_propagates_nan() -> None:
    out = normalize(_values(), baseline=np.array([[np.nan]]), mode="log_ratio")
    assert np.isnan(out).all()


def test_log_ratio_without_a_baseline_raises() -> None:
    with pytest.raises(ValueError, match="requires a baseline"):
        normalize(_values(), baseline=None, mode="log_ratio")


def test_log10_with_a_baseline_raises() -> None:
    with pytest.raises(ValueError, match="takes no baseline"):
        normalize(_values(), baseline=np.array([[1.0]]), mode="log10")


def test_percent_is_relative_change_in_percent() -> None:
    values = np.array([[[0.5, 1.0, 2.0]]])
    out = normalize(values, baseline=np.array([[1.0]]), mode="percent")
    np.testing.assert_allclose(out, [[[-50.0, 0.0, 100.0]]])


def test_percent_against_itself_is_exactly_zero() -> None:
    values = _values()
    out = normalize(values, baseline=values[:, :, 0], mode="percent")
    assert out[0, 0, 0] == 0.0


def test_percent_without_a_baseline_raises() -> None:
    with pytest.raises(ValueError, match="requires a baseline"):
        normalize(_values(), baseline=None, mode="percent")


def test_percent_propagates_a_non_finite_baseline() -> None:
    out = normalize(_values(), baseline=np.array([[np.nan]]), mode="percent")
    assert np.isnan(out).all()
