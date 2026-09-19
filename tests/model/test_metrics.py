from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.metrics import (
    classification_metrics,
    regression_metrics,
    within_condition_metrics,
    within_subject_centered_metrics,
)


def test_a_single_class_fold_does_not_report_a_balanced_accuracy() -> None:
    # One class in the held-out subject makes balanced accuracy undefined. Reporting NaN
    # keeps the fold visible as unscored; reporting 0.5 would invent a chance result.
    result = classification_metrics(
        np.array([1, 1, 1], dtype=np.intp),
        np.array([1, 1, 0], dtype=np.intp),
        groups=np.array(["s1", "s1", "s1"], dtype=object),
    )
    assert np.isnan(result.balanced_accuracy)


def test_a_single_class_confusion_matrix_still_has_both_axes() -> None:
    # A 1x1 matrix would silently reindex downstream, turning "no negatives were seen"
    # into "no negatives exist".
    result = classification_metrics(
        np.array([1, 1], dtype=np.intp), np.array([1, 1], dtype=np.intp)
    )
    assert result.confusion.shape == (2, 2)


def test_regression_metrics_report_subject_level_r() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    groups = np.array(["s1"] * 3 + ["s2"] * 3, dtype=object)
    summary, _ = regression_metrics(y_true, y_true.copy(), groups)
    assert summary["subject_level_r"] == pytest.approx(1.0, abs=1e-5)


def test_classification_primary_precision_recall_f1_are_subject_level() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.intp)
    y_pred = np.array([0, 1, 0, 0], dtype=np.intp)
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    result = classification_metrics(y_true, y_pred, groups=groups)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_group_classification_permutations_do_not_fallback_to_pooled_auc() -> None:
    # If subject-level AUC cannot be computed, mean_subject_auc remains NaN rather than
    # quietly falling back to pooled trials.
    y_true = np.array([0, 0, 1, 1], dtype=np.intp)
    y_pred = np.array([0, 0, 1, 1], dtype=np.intp)
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    result = classification_metrics(y_true, y_pred, groups=groups)
    assert np.isnan(result.mean_subject_auc)


def test_within_subject_centered_metrics_computes_r2() -> None:
    target = np.array([1.0, 2.0, 10.0, 12.0])
    full = np.array([1.0, 2.0, 10.0, 12.0])
    nuis = np.array([1.5, 1.5, 11.0, 11.0])
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=object)
    res = within_subject_centered_metrics(target, full, nuis, groups)
    assert res["within_subject_centered_full_r2"] == pytest.approx(1.0)


def test_within_condition_metrics_computes_r2() -> None:
    target = np.array([1.0, 2.0, 1.0, 2.0])
    full = np.array([1.0, 2.0, 1.0, 2.0])
    nuis = np.array([1.5, 1.5, 1.5, 1.5])
    groups = np.array(["s1", "s1", "s1", "s1"], dtype=object)
    conditions = np.array(["c1", "c1", "c1", "c1"])
    res = within_condition_metrics(target, full, nuis, groups, conditions)
    assert res["within_condition_centered_full_r2"] == pytest.approx(1.0)


def test_within_subject_centered_metrics_equal_subject_weighting() -> None:
    t_a = np.linspace(0.0, 10.0, 100)
    f_a = t_a.copy()
    n_a = np.full(100, 5.0)
    t_b = np.linspace(0.0, 10.0, 10)
    f_b = np.full(10, 5.0)
    n_b = np.full(10, 5.0)

    target = np.concatenate([t_a, t_b])
    full = np.concatenate([f_a, f_b])
    nuis = np.concatenate([n_a, n_b])
    groups = np.array(["sA"] * 100 + ["sB"] * 10, dtype=object)

    res = within_subject_centered_metrics(target, full, nuis, groups)
    assert res["within_subject_centered_full_r2"] == pytest.approx(0.5)


def test_classification_metrics_handles_2d_probabilities() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    y_pred = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    y_prob_2d = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        dtype=float,
    )
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
    result = classification_metrics(y_true, y_pred, y_prob=y_prob_2d, groups=groups)
    assert result.auc == pytest.approx(1.0)
    assert result.average_precision == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([1, 1, 2, 2], [1, 2, 2, 2]), ([0, 0, 1, 1], [0, 2, 1, 1])],
    ids=["event-codes", "prediction-out-of-range"],
)
def test_labels_other_than_zero_and_one_are_refused(y_true: list[int], y_pred: list[int]) -> None:
    # The confusion matrix, specificity and the positive class of precision and recall all
    # assume 0/1 coding; event codes 1/2 would be scored against the wrong classes.
    with pytest.raises(ValueError, match="0/1"):
        classification_metrics(np.array(y_true), np.array(y_pred))
