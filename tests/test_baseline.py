import numpy as np
import pytest

from eegfeat.baseline import normalize


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


# Band power of an EEG channel in V^2, against a baseline of 4e-12: ratios 0.5, 1.25, 2.5.
EEG_POWER = np.array([[[2e-12, 5e-12, 1e-11]]])
EEG_BASELINE = np.array([[4e-12]])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("log_ratio", np.log10([0.5, 1.25, 2.5])),
        ("db", 10.0 * np.log10([0.5, 1.25, 2.5])),
        ("percent", [-50.0, 25.0, 150.0]),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_a_ratio_to_baseline_does_not_depend_on_the_power_unit(
    mode: str, expected: list[float], scale: float
) -> None:
    # Scaled by 1e-12 the power sits where an eLORETA estimate in A.m does, below the
    # old absolute floor of 1e-20, which turned every ratio into 1 and every feature
    # into one constant.
    out = normalize(EEG_POWER * scale, baseline=EEG_BASELINE * scale, mode=mode)
    np.testing.assert_allclose(out, [[expected]], rtol=1e-12)


def test_log10_of_rescaled_power_shifts_by_the_log_of_the_scale() -> None:
    out = normalize(EEG_POWER * 1e-12, baseline=None, mode="log10")
    np.testing.assert_allclose(out, [[[-23.69897000433602, -23.30102999566398, -23.0]]])


@pytest.mark.parametrize("scale", [1.0, 1e-24])
def test_zero_baseline_is_floored_rather_than_dividing_by_zero(scale: float) -> None:
    # Positive power against a baseline of nothing is an increase at any unit. An
    # absolute floor above the power made it read as a decrease.
    out = normalize(np.array([[[scale]]]), baseline=np.array([[0.0]]), mode="log_ratio")
    assert np.isfinite(out).all()
    assert out.item() > 0.0


def test_a_floored_zero_baseline_gives_the_same_ratio_at_any_unit() -> None:
    native, rescaled = (
        normalize(np.array([[[scale]]]), baseline=np.array([[0.0]]), mode="log_ratio")
        for scale in (1.0, 1e-24)
    )
    np.testing.assert_allclose(rescaled, native, rtol=1e-12)


def test_a_zero_window_against_a_zero_baseline_is_no_change() -> None:
    # Both sides share one floor. Floored only in the numerator, silence against
    # silence would read as a decrease of every decade the floor spans.
    values = np.array([[[0.0, 1e-24]]])
    out = normalize(values, baseline=np.array([[0.0]]), mode="log_ratio")
    assert out[0, 0, 0] == 0.0


def test_a_silent_window_against_a_real_baseline_is_a_full_decrease() -> None:
    # The baseline sets the floor too; without it this cell would have no scale and be
    # withheld, when it is the largest decrease there is.
    values, baseline = np.array([[[0.0]]]), np.array([[4e-24]])
    assert normalize(values, baseline=baseline, mode="percent").item() == -100.0
    assert normalize(values, baseline=baseline, mode="log_ratio").item() < 0.0


@pytest.mark.parametrize("mode", ["log10", "log_ratio", "db", "percent"])
def test_an_epoch_with_no_power_at_all_is_missing(mode: str) -> None:
    # A flat channel has no scale to floor against, so any finite answer would be a
    # constant chosen by the floor rather than measured.
    baseline = None if mode == "log10" else np.zeros((1, 1))
    out = normalize(np.zeros((1, 1, 2)), baseline=baseline, mode=mode)
    assert np.isnan(out).all()


def test_a_floored_value_does_not_depend_on_other_epochs() -> None:
    # A floor pooled over the table would move one trial's feature with another
    # trial's power, which leaks across the folds of any trial-level model.
    alone = normalize(np.array([[[0.0, 1.0]]]), baseline=np.array([[1.0]]), mode="log_ratio")
    beside = normalize(
        np.array([[[0.0, 1.0]], [[1e6, 1e6]]]),
        baseline=np.array([[1.0], [1e6]]),
        mode="log_ratio",
    )
    np.testing.assert_array_equal(beside[0], alone[0])


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
