from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_classification
from eegfeat.model.splits import InnerSplit, loso_folds, within_subject_folds

GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 6).astype(object)
Y = np.tile([0, 1], GROUPS.size // 2).astype(np.intp)
X = np.random.default_rng(0).normal(size=(GROUPS.size, 3)) + Y[:, None]
PIPE = Pipeline([("classifier", LogisticRegression(max_iter=500))])
GRID = {"classifier__C": [0.1, 1.0]}
STRATIFIED = InnerSplit(grouping="subject", stratified=True, n_splits=2)


def test_probability_columns_are_pinned_to_the_recorded_classes() -> None:
    for prediction in cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
    ):
        assert prediction.y_prob is not None
        assert prediction.y_prob.shape[1] == len(prediction.classes)
        np.testing.assert_allclose(prediction.y_prob.sum(axis=1), 1.0)


def test_a_single_class_training_fold_raises() -> None:
    y_degenerate = np.where(GROUPS == "s1", 1, 0).astype(np.intp)
    with pytest.raises(ValueError, match="one class"):
        cross_fit_classification(
            [f for f in loso_folds(GROUPS) if set(GROUPS[f.test]) != {"s1"}][:1],
            X,
            y_degenerate,
            GROUPS,
            PIPE,
            GRID,
            inner=STRATIFIED,
            seed=0,
        )


def test_an_estimator_without_predict_proba_reports_none_not_a_decision_function() -> None:
    pipe = Pipeline([("classifier", RidgeClassifier())])
    predictions = cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, pipe, {}, inner=STRATIFIED, seed=0
    )
    assert all(p.y_prob is None for p in predictions)


def test_labels_stay_integers_through_the_fold_loop() -> None:
    for prediction in cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
    ):
        assert prediction.y_pred.dtype == np.intp


def test_within_subject_classification_serves_run_splits() -> None:
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 2).astype(object)
    y = np.tile([0, 1], groups.size // 2).astype(np.intp)
    x = np.ones((groups.size, 2))
    folds = within_subject_folds(groups, runs, inner_splits=2, seed=0)
    inner_run = InnerSplit(grouping="run", stratified=True, n_splits=2)
    predictions = cross_fit_classification(
        folds, x, y, groups, PIPE, GRID, inner=inner_run, seed=0, runs=runs
    )
    assert len(predictions) > 0
    assert all(p.subject is not None for p in predictions)


def test_within_subject_classification_applies_harmonization() -> None:
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 2).astype(object)
    y = np.tile([0, 1], groups.size // 2).astype(np.intp)
    x = np.column_stack([np.ones(groups.size), np.zeros(groups.size)])
    x[groups == "s2", -1] = 1.0
    folds = within_subject_folds(groups, runs, inner_splits=2, seed=0)
    inner_run = InnerSplit(grouping="run", stratified=True, n_splits=2)
    predictions = cross_fit_classification(
        folds,
        x,
        y,
        groups,
        PIPE,
        GRID,
        inner=inner_run,
        seed=0,
        runs=runs,
        harmonization="intersection",
    )
    assert len(predictions) > 0


def test_within_subject_classification_raises_when_training_fold_has_one_class() -> None:
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3", "r4"], 2), 2).astype(object)
    y = np.zeros(groups.size, dtype=np.intp)
    y[groups == "s2"] = 1
    folds = within_subject_folds(groups, runs, inner_splits=2, seed=0)
    inner_run = InnerSplit(grouping="run", stratified=True, n_splits=2)
    with pytest.raises(ValueError, match="one class"):
        cross_fit_classification(
            folds[:1],
            np.ones((groups.size, 2)),
            y,
            groups,
            PIPE,
            GRID,
            inner=inner_run,
            seed=0,
            runs=runs,
        )

