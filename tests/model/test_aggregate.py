from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eegfeat.model.aggregate import (
    AggregationConfig,
    bootstrap_mean_ci,
    fold_results,
    paired_signflip_p_value,
    subject_level_errors,
    subject_level_r,
)
from eegfeat.model.crossfit import FoldPrediction


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


def test_a_subject_whose_correlation_is_undefined_raises() -> None:
    # A flat predictor within a subject is undefined, raising ValueError rather
    # than being silently dropped.
    with pytest.raises(ValueError, match="non-finite correlation"):
        subject_level_r(
            _predictions(
                {
                    "s1": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                    "s2": ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
                }
            )
        )


def test_aggregation_config_rejects_an_unknown_weighting() -> None:
    with pytest.raises(ValueError, match="subject_weighting"):
        AggregationConfig(subject_weighting="by_vibes")  # type: ignore[arg-type]


def test_subject_level_r_uses_equal_subject_weighting_by_default() -> None:
    cfg = AggregationConfig()
    assert cfg.subject_weighting == "equal"


def test_subject_level_r_rejects_invalid_subjects() -> None:
    # A subject with fewer than 3 finite predictions cannot produce a correlation.
    df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2"],
            "y_true": [1.0, 2.0, 1.0, np.nan],
            "y_pred": [1.0, 2.0, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="fewer than 3 finite predictions"):
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


def test_fold_results_accepts_fold_prediction_dataclass() -> None:
    pred1 = FoldPrediction(
        fold=1,
        subject="s1",
        rows=np.array([0], dtype=np.intp),
        y_true=np.array([1.0]),
        y_pred=np.array([1.1]),
        best_params={},
    )
    pred2 = FoldPrediction(
        fold=2,
        subject="s2",
        rows=np.array([1], dtype=np.intp),
        y_true=np.array([2.0]),
        y_pred=np.array([2.1]),
        best_params={},
    )
    yt, yp, grps, test_idx, fold_ids = fold_results([pred2, pred1])
    np.testing.assert_array_equal(yt, [1.0, 2.0])
    np.testing.assert_array_equal(yp, [1.1, 2.1])
    assert grps == ["s1", "s2"]
    assert test_idx == [0, 1]
    assert fold_ids == [1, 2]


def test_fold_results_rejects_mismatched_lengths() -> None:
    records = [
        {
            "fold": 1,
            "y_true": [1.0, 2.0],
            "y_pred": [1.1],
            "groups": ["s1", "s1"],
            "test_idx": [0, 1],
        },
    ]
    with pytest.raises(ValueError, match="mismatched lengths"):
        fold_results(records)


def test_an_undefined_subject_can_count_as_no_correlation() -> None:
    # Permutation nulls score a subject whose predictions do not vary as r = 0 instead of
    # discarding the whole draw, so the other subjects' correlations still count.
    df = _predictions(
        {
            "s1": ([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]),
            "s2": ([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]),
            "s3": ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
        }
    )
    # s1 and s2 each correlate at exactly 0.5; s3 enters Fisher z as 0.
    expected = np.tanh((2.0 * np.arctanh(0.5) + 0.0) / 3.0)
    assert subject_level_r(df, undefined="zero").r == pytest.approx(expected)


def test_counting_undefined_subjects_as_zero_still_refuses_too_few_trials() -> None:
    # Too few trials is a property of the design, shared by every permutation, so it
    # stays an error rather than a zero.
    df = _predictions({"s1": ([1.0, 2.0], [1.0, 2.0]), "s2": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])})
    with pytest.raises(ValueError, match="fewer than 3"):
        subject_level_r(df, undefined="zero")


def test_fold_results_accepts_array_groups_in_dict_records() -> None:
    records = [
        {
            "fold": 1,
            "y_true": np.array([1.0, 2.0]),
            "y_pred": np.array([1.1, 2.1]),
            "groups": np.array(["s1", "s1"], dtype=object),
            "test_idx": np.array([0, 1]),
        },
    ]
    _, _, grps, _, _ = fold_results(records)
    assert grps == ["s1", "s1"]


def test_fold_results_labels_loso_predictions_from_the_groups_array() -> None:
    # LOSO folds carry no subject label, so the rows are labelled from the design's groups.
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    predictions = [
        FoldPrediction(
            fold=2,
            subject=None,
            rows=np.array([0, 1], dtype=np.intp),
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.5, 2.5]),
            best_params={},
        ),
        FoldPrediction(
            fold=1,
            subject=None,
            rows=np.array([2, 3], dtype=np.intp),
            y_true=np.array([3.0, 4.0]),
            y_pred=np.array([3.5, 4.5]),
            best_params={},
        ),
    ]
    _, _, grps, test_idx, _ = fold_results(predictions, groups=groups)
    assert grps == ["s2", "s2", "s1", "s1"]
    assert test_idx == [2, 3, 0, 1]


def test_fold_results_refuses_to_mix_labelled_and_unlabelled_records() -> None:
    # A groups list shorter than y_true would misalign every subject after the first gap.
    predictions = [
        FoldPrediction(
            fold=1,
            subject="s1",
            rows=np.array([0], dtype=np.intp),
            y_true=np.array([1.0]),
            y_pred=np.array([1.1]),
            best_params={},
        ),
        FoldPrediction(
            fold=2,
            subject=None,
            rows=np.array([1], dtype=np.intp),
            y_true=np.array([2.0]),
            y_pred=np.array([2.1]),
            best_params={},
        ),
    ]
    with pytest.raises(ValueError, match="subject labels"):
        fold_results(predictions)


def _noise_frame(n_subjects: int, n_trials: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "subject_id": f"s{s}",
                    "y_true": rng.normal(size=n_trials),
                    "y_pred": rng.normal(size=n_trials),
                }
            )
            for s in range(n_subjects)
        ]
    )


def test_trial_count_weighting_refuses_subjects_with_an_undefined_fisher_z_variance() -> None:
    # var(z) = 1/(n-3) is infinite at n=3. Flooring that weight at 1.0 gave such subjects the
    # same weight as a 4-trial subject and pushed the false positive rate from 5% to 17%.
    with pytest.raises(ValueError, match="3 or fewer"):
        subject_level_r(
            _noise_frame(20, 3, seed=0),
            config=AggregationConfig(subject_weighting="trial_count"),
        )


def test_equal_weighting_uses_student_t_for_the_estimated_between_subject_error() -> None:
    # The between-subject SD is estimated from the subjects, so a normal quantile makes the
    # interval too narrow: at 5 subjects a nominal 95% interval covered about 88%.
    frame = _noise_frame(5, 20, seed=1)
    result = subject_level_r(frame)

    z_vals = np.arctanh(np.clip([r for _, r in result.per_subject], -0.999999, 0.999999))
    se = float(np.std(z_vals, ddof=1) / np.sqrt(len(z_vals)))
    expected_half_width = float(stats.t.ppf(0.975, len(z_vals) - 1)) * se

    observed_half_width = (np.arctanh(result.ci_high) - np.arctanh(result.ci_low)) / 2
    assert observed_half_width == pytest.approx(expected_half_width, rel=1e-6)
    assert float(stats.t.ppf(0.975, len(z_vals) - 1)) > 1.96


def _uneven_frame() -> pd.DataFrame:
    rows = []
    for s, n in enumerate([5, 50, 5, 50, 5]):
        rng = np.random.default_rng(s)
        rows.append(
            pd.DataFrame(
                {"subject_id": f"s{s}", "y_true": rng.normal(size=n), "y_pred": rng.normal(size=n)}
            )
        )
    return pd.concat(rows)


def test_subject_level_errors_honour_the_configured_weighting() -> None:
    # One config must not produce a trial-weighted correlation and an equal-weighted error.
    frame = _uneven_frame()
    equal = subject_level_errors(frame, config=AggregationConfig(subject_weighting="equal"))
    weighted = subject_level_errors(
        frame, config=AggregationConfig(subject_weighting="trial_count")
    )
    assert equal["mean_mae"] != weighted["mean_mae"]


def test_subject_level_errors_honour_the_configured_seed() -> None:
    # A hardcoded seed made the bootstrap look perfectly stable across seeds.
    frame = _uneven_frame()
    first = subject_level_errors(
        frame, config=AggregationConfig(ci_method="bootstrap", bootstrap_iterations=200, seed=1)
    )
    second = subject_level_errors(
        frame, config=AggregationConfig(ci_method="bootstrap", bootstrap_iterations=200, seed=999)
    )
    assert first["ci_low_mae"] != second["ci_low_mae"]


def test_bootstrap_intervals_are_not_silently_replaced_by_fixed_effects() -> None:
    # With two subjects the bootstrap branch was skipped and the fixed-effects interval was
    # returned instead, byte-identical and with nothing on the result to say which ran.
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "subject_id": f"s{s}",
                    "y_true": np.random.default_rng(s).normal(size=20),
                    "y_pred": np.random.default_rng(s + 9).normal(size=20),
                }
            )
            for s in range(2)
        ]
    )
    with pytest.raises(ValueError, match="at least 3 subjects"):
        subject_level_r(frame, config=AggregationConfig(ci_method="bootstrap"))


@pytest.mark.parametrize("missing", ["subject_id", "y_true", "y_pred"])
@pytest.mark.parametrize("function", [subject_level_r, subject_level_errors])
def test_subject_level_aggregates_name_the_columns_they_need(function, missing: str) -> None:
    frame = _predictions({"s1": ([1.0, 2.0, 3.0], [1.0, 2.5, 2.8])}).drop(columns=[missing])

    with pytest.raises(ValueError, match=f"missing.*{missing}"):
        function(frame)
