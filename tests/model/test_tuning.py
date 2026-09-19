from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.estimators import ridge_grid, ridge_pipeline
from eegfeat.model.splits import InnerSplit
from eegfeat.model.transformers import PreprocessingConfig
from eegfeat.model.tuning import fit_untuned, tune

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean", "median"]}
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)
BY_RUN = InnerSplit(grouping="run", n_splits=2)
TWO_SUBJECTS = np.array(["s1"] * 4 + ["s2"] * 4, dtype=object)
TWO_RUNS = np.array(["r1"] * 4 + ["r2"] * 4, dtype=object)
X8 = np.arange(8.0).reshape(-1, 1)
Y8 = np.arange(8.0)


def test_tuning_refuses_a_training_fold_with_one_group() -> None:
    # With one group the inner split cannot be group-disjoint, so the chosen
    # hyperparameters would be selected on data from the subject being predicted.
    with pytest.raises(ValueError, match="at least 2"):
        tune(
            PIPE,
            GRID,
            X8,
            Y8,
            np.array(["s1"] * 8, dtype=object),
            split=BY_SUBJECT,
            seed=0,
            fold=1,
        )


def test_a_within_subject_fold_tunes_on_runs_not_subjects() -> None:
    # The training block is one subject, so runs are the only split that exists. This is
    # what upstream does at orchestration.py:868, and the reason tune takes an InnerSplit
    # rather than "the groups".
    assert tune(PIPE, GRID, X8, Y8, TWO_RUNS, split=BY_RUN, seed=0, fold=1).best_params


def test_the_fold_is_named_when_tuning_fails() -> None:
    with pytest.raises(ValueError, match="Fold 7"):
        tune(
            PIPE,
            GRID,
            X8,
            Y8,
            np.array(["s1"] * 8, dtype=object),
            split=BY_SUBJECT,
            seed=0,
            fold=7,
        )


def test_tuning_does_not_fall_back_to_an_untuned_fit() -> None:
    # A failed inner search must surface. Falling back to a default fit would report a
    # score for a model nobody selected, indistinguishable from a tuned one downstream.
    with pytest.raises(ValueError):
        tune(
            PIPE,
            {"regressor__nonexistent": [1]},
            X8,
            Y8,
            TWO_SUBJECTS,
            split=BY_SUBJECT,
            seed=0,
            fold=1,
        )


def test_within_subject_inner_cv_failure_raises_instead_of_default_fit() -> None:
    with pytest.raises(ValueError, match="inner CV failed"):
        tune(
            PIPE,
            {"regressor__nonexistent": [1]},
            X8,
            Y8,
            TWO_RUNS,
            split=BY_RUN,
            seed=0,
            fold=1,
        )


class _NaNScoringRegressor(DummyRegressor):
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: object = None) -> float:
        return float("nan")


def test_non_finite_inner_scores_are_refused() -> None:
    # A grid point scoring NaN on every inner split is not the best model; selecting it
    # by argmax over NaN picks whichever the sort happened to put first.
    pipe = Pipeline([("regressor", _NaNScoringRegressor(strategy="mean"))])
    with pytest.raises(ValueError, match="finite"):
        tune(pipe, GRID, X8, Y8, TWO_SUBJECTS, split=BY_SUBJECT, seed=0, fold=1)


def test_fit_untuned_sets_available_random_state_parameter() -> None:
    pipe = Pipeline([("rf", RandomForestRegressor(n_estimators=1))])
    fitted = fit_untuned(pipe, X8, Y8, seed=17)
    assert fitted.named_steps["rf"].random_state == 17


def test_within_subject_regression_tunes_with_real_pipeline() -> None:
    cfg = PreprocessingConfig()
    pipe = ridge_pipeline(cfg, seed=42)
    grid = ridge_grid()
    fit = tune(pipe, grid, X8, Y8, TWO_RUNS, split=BY_RUN, seed=42, fold=1)
    assert "regressor__alpha" in fit.best_params
