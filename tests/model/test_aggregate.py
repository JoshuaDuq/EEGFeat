from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegfeat.model.aggregate import (
    AggregationConfig,
    bootstrap_mean_ci,
    fold_results,
    paired_signflip_p_value,
    subject_level_errors,
    subject_level_r,
)


def _predictions(per_subject: dict[str, tuple[list[float], list[float]]]) -> pd.DataFrame:
    rows = [
        {"subject_id": subject, "y_true": t, "y_pred": p}
        for subject, (truths, preds) in per_subject.items()
        for t, p in zip(truths, preds, strict=True)
    ]
    return pd.DataFrame(rows)


def test_each_subject_counts_once_regardless_of_trial_count() -> None:
    # A subject with forty trials and one with four contribute equally. Pooling instead
    # would let the largest subject decide the cohort result.
    small = ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    large = ([float(i) for i in range(40)], [float(-i) for i in range(40)])
    result = subject_level_r(_predictions({"s1": small, "s2": large}))
    assert result.r == pytest.approx(0.0, abs=1e-9)


def test_per_subject_correlations_are_reported_individually() -> None:
    result = subject_level_r(
        _predictions(
            {
                "s1": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                "s2": ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
            }
        )
    )
    assert dict(result.per_subject) == pytest.approx({"s1": 1.0, "s2": -1.0})


def test_a_subject_whose_correlation_is_undefined_does_not_become_zero() -> None:
    # A flat predictor within a subject is unscored, not scored as no correlation.
    result = subject_level_r(
        _predictions(
            {
                "s1": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                "s2": ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
            }
        )
    )
    assert result.r == pytest.approx(1.0)


def test_aggregation_config_rejects_an_unknown_weighting() -> None:
    with pytest.raises(ValueError, match="subject_weighting"):
        AggregationConfig(subject_weighting="by_vibes")  # type: ignore[arg-type]


def test_subject_level_r_uses_equal_subject_weighting_by_default() -> None:
    cfg = AggregationConfig()
    assert cfg.subject_weighting == "equal"


def test_subject_level_r_rejects_invalid_subjects() -> None:
    # A subject with fewer than 2 finite predictions cannot produce a correlation.
    df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"],
            "y_true": [1.0, 2.0, 1.0, np.nan],
            "y_pred": [1.0, 2.0, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="fewer than 2 finite predictions"):
        subject_level_r(df)


def test_subject_level_errors_reject_invalid_subjects() -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"],
            "y_true": [1.0, 2.0, 1.0, 2.0],
            "y_pred": [1.1, 2.1, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="no finite predictions"):
        subject_level_errors(df)


def test_bootstrap_mean_ci_brackets_mean() -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = bootstrap_mean_ci(vals, iterations=1000, seed=42)
    assert lo <= 3.0 <= hi


def test_paired_signflip_p_value_zero_differences() -> None:
    diffs = np.zeros(10)
    p = paired_signflip_p_value(diffs, iterations=100, seed=42)
    assert p == pytest.approx(1.0)


def test_fold_results_preserves_fold_order() -> None:
    records = [
        {"fold": 2, "y_true": [2.0], "y_pred": [2.1], "groups": ["s2"], "test_idx": [1]},
        {"fold": 1, "y_true": [1.0], "y_pred": [1.1], "groups": ["s1"], "test_idx": [0]},
    ]
    yt, yp, grps, test_idx, fold_ids = fold_results(records)
    np.testing.assert_array_equal(yt, [1.0, 2.0])
    np.testing.assert_array_equal(yp, [1.1, 2.1])
    assert grps == ["s1", "s2"]
    assert test_idx == [0, 1]
    assert fold_ids == [1, 2]
