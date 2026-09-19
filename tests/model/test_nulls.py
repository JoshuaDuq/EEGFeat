from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from eegfeat.model.aggregate import AggregationConfig
from eegfeat.model.nulls import (
    NullConfig,
    changed_fraction,
    circular_shift_group,
    permutation_test,
    permute,
)
from eegfeat.model.splits import InnerSplit, loso_folds, within_subject_folds

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean"]}
GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 6).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2"], 3), 4).astype(object)
X = np.arange(GROUPS.size, dtype=float).reshape(-1, 1)
Y = np.arange(GROUPS.size, dtype=float)
FOLDS = loso_folds(GROUPS)
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)


def test_the_shift_set_is_the_whole_cycle_including_the_identity() -> None:
    assert circular_shift_group(11) == tuple(range(11))


def test_an_empty_run_yields_no_shifts() -> None:
    assert circular_shift_group(0) == ()


def test_a_permutation_that_changed_little_still_enters_the_null() -> None:
    groups = np.repeat(["s1", "s2"], 5).astype(object)
    y = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    x = np.ones((10, 2))
    folds = loso_folds(groups)
    result = permutation_test(
        folds,
        x,
        y,
        groups,
        None,
        PIPE,
        {},
        observed=0.0,
        config=NullConfig(n_permutations=20),
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=0,
    )
    assert result.null.size == 20
    assert any(result.changed_fractions < 0.3)


def test_constant_predictions_enter_null_as_zero() -> None:
    groups = np.repeat(["s1", "s2"], 5).astype(object)
    y = np.arange(10, dtype=float)
    x = np.ones((10, 2))
    folds = loso_folds(groups)
    const_pipe = Pipeline([("regressor", DummyRegressor(strategy="constant", constant=3.0))])
    result = permutation_test(
        folds,
        x,
        y,
        groups,
        None,
        const_pipe,
        {},
        observed=0.0,
        config=NullConfig(n_permutations=10),
        inner=InnerSplit(grouping="subject", n_splits=2),
        seed=0,
    )
    assert result.null.size == 10
    assert result.n_incomplete == 0
    np.testing.assert_array_equal(result.null, np.zeros(10))


def test_run_wise_permutation_is_bijection_on_unequal_run_lengths() -> None:
    runs = np.array(["A", "A", "A", "A", "B", "B", "C"], dtype=object)
    groups = np.full(len(runs), "s1", dtype=object)
    y = np.arange(len(runs), dtype=float)

    for s in range(10):
        y_perm = permute(
            y,
            groups,
            runs=runs,
            config=NullConfig(scheme="run_wise"),
            rng=np.random.default_rng(s),
        )
        np.testing.assert_array_equal(np.sort(y_perm), np.sort(y))


def test_the_changed_fraction_is_reported_not_acted_on() -> None:
    result = permutation_test(
        FOLDS,
        X,
        Y,
        GROUPS,
        RUNS,
        PIPE,
        GRID,
        observed=0.0,
        config=NullConfig(n_permutations=20),
        inner=BY_SUBJECT,
        seed=0,
    )
    assert result.changed_fractions.size == result.null.size


def test_a_scheme_under_which_nothing_ever_changes_is_refused() -> None:
    # Every trial is its own run, so shuffling within runs can never move a label.
    own_runs = np.array([f"r{i}" for i in range(GROUPS.size)], dtype=object)
    with pytest.raises(ValueError, match="never changes"):
        permutation_test(
            FOLDS,
            X,
            Y,
            GROUPS,
            own_runs,
            PIPE,
            GRID,
            observed=0.0,
            config=NullConfig(scheme="run_wise", n_permutations=10),
            inner=BY_SUBJECT,
            seed=0,
        )


def test_the_p_value_counts_the_observed_statistic_in_both_terms() -> None:
    result = permutation_test(
        FOLDS,
        X,
        Y,
        GROUPS,
        RUNS,
        PIPE,
        GRID,
        observed=np.inf,
        config=NullConfig(n_permutations=19),
        inner=BY_SUBJECT,
        seed=0,
    )
    assert result.p_value == pytest.approx(1.0 / 20.0)


def test_changed_fraction_refuses_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        changed_fraction(np.arange(4.0), np.arange(5.0))


def test_an_unknown_scheme_is_refused_rather_than_downgraded() -> None:
    with pytest.raises(ValueError, match="scheme"):
        NullConfig(scheme="shuffle_everything")  # type: ignore[arg-type]


def test_null_config_has_no_changed_fraction_filter() -> None:
    assert "min_changed_fraction" not in NullConfig.__dataclass_fields__


def test_permute_requires_runs_for_runwise_scheme() -> None:
    with pytest.raises(ValueError, match="run labels"):
        permute(
            Y,
            GROUPS,
            runs=None,
            config=NullConfig(scheme="run_wise"),
            rng=np.random.default_rng(0),
        )


def test_permute_rejects_run_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        permute(
            Y,
            GROUPS,
            runs=np.array(["r1", "r2"], dtype=object),
            config=NullConfig(scheme="run_wise"),
            rng=np.random.default_rng(0),
        )


def test_permute_rejects_all_missing_runs() -> None:
    with pytest.raises(ValueError, match="run labels"):
        permute(
            Y,
            GROUPS,
            runs=np.full(GROUPS.size, np.nan, dtype=object),
            config=NullConfig(scheme="run_wise"),
            rng=np.random.default_rng(0),
        )


def test_circular_shift_requires_min_retained_trials() -> None:
    with pytest.raises(ValueError, match="retained trials"):
        permute(
            Y,
            GROUPS,
            runs=RUNS,
            trial_indices=np.arange(Y.size),
            config=NullConfig(scheme="circular_shift_within_run", min_retained_trials=8),
            rng=np.random.default_rng(0),
        )


def test_circular_shift_requires_trial_indices() -> None:
    with pytest.raises(ValueError, match="within-run trial indices"):
        permute(
            Y,
            GROUPS,
            runs=RUNS,
            trial_indices=None,
            config=NullConfig(scheme="circular_shift_within_run", min_retained_trials=2),
            rng=np.random.default_rng(0),
        )


def test_circular_shift_requires_finite_trial_indices() -> None:
    bad_trials = np.arange(Y.size, dtype=float)
    bad_trials[0] = np.nan
    with pytest.raises(ValueError, match="finite within-run trial indices"):
        permute(
            Y,
            GROUPS,
            runs=RUNS,
            trial_indices=bad_trials,
            config=NullConfig(scheme="circular_shift_within_run", min_retained_trials=2),
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("n_retained", [8, 9, 10, 11, 12])
def test_the_shift_set_is_closed_under_composition(n_retained: int) -> None:
    shifts = set(circular_shift_group(n_retained))
    assert 0 in shifts
    for first in shifts:
        for second in shifts:
            assert (first + second) % n_retained in shifts


@pytest.mark.parametrize("n_retained", [0, 1, 7])
def test_short_runs_carry_no_shift(n_retained: int) -> None:
    from eegfeat.model.nulls import is_permutation_valid_run

    assert not is_permutation_valid_run(np.arange(1, n_retained + 1, dtype=int))


@pytest.mark.parametrize("n_retained", [8, 11])
def test_runs_with_enough_retained_trials_are_permutation_valid(n_retained: int) -> None:
    from eegfeat.model.nulls import is_permutation_valid_run

    assert is_permutation_valid_run(np.arange(1, n_retained + 1, dtype=int))


def test_sampled_shifts_cover_the_whole_cycle() -> None:
    groups = np.array(["sub-0001"] * 11, dtype=object)
    runs = np.ones(11, dtype=object)
    trial_indices = np.arange(1, 12, dtype=np.intp)
    rng = np.random.default_rng(0)
    y = np.arange(11, dtype=float)

    observed = set()
    cfg = NullConfig(scheme="circular_shift_within_run", min_retained_trials=8)
    for _ in range(400):
        permuted = permute(y, groups, runs=runs, trial_indices=trial_indices, config=cfg, rng=rng)
        observed.add(int(permuted[0]))

    assert observed == set(range(11))


def test_permutation_completion_threshold_enforces_failure_fraction() -> None:
    class FailingPipeline(Pipeline):
        def fit(self, *args: object, **kwargs: object) -> FailingPipeline:
            raise RuntimeError("Intentional fold failure")

    bad_pipe = FailingPipeline([("regressor", DummyRegressor())])
    with pytest.raises(ValueError, match="Insufficient valid permutations"):
        permutation_test(
            FOLDS,
            X,
            Y,
            GROUPS,
            RUNS,
            bad_pipe,
            {},
            observed=0.0,
            config=NullConfig(n_permutations=5, min_complete_fraction=0.8),
            inner=BY_SUBJECT,
            seed=0,
        )


class _DivergesOnSomeTargets(DummyRegressor):
    # Fails whenever a permutation put the first two training targets out of order.
    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: object = None
    ) -> _DivergesOnSomeTargets:
        if y[0] > y[1]:
            raise RuntimeError("solver diverged")
        return super().fit(X, y, sample_weight)


class _ConstantForFirstSubject(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    # Column 0 holds the subject index; a model fitted on subject 0 alone predicts a constant.
    def fit(self, X: np.ndarray, y: np.ndarray) -> _ConstantForFirstSubject:
        self.constant_ = bool(np.all(X[:, 0] == 0.0))
        self.model_ = LinearRegression().fit(X[:, 1:], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.constant_:
            return np.ones(len(X))
        return np.asarray(self.model_.predict(X[:, 1:]))


def test_the_loso_null_scores_subject_level_r_like_the_observed_statistic() -> None:
    # Each subject's target is offset by a constant the features carry. A within-subject
    # shuffle keeps the offsets, so r pooled over trials stays near 1 in every draw; the
    # documented observed statistic is subject-level r, whose null centres on zero.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"s{i}" for i in range(6)], 12).astype(object)
    offsets = np.repeat(np.arange(6) * 3.0, 12)
    x = np.column_stack([rng.normal(size=72), offsets])
    y = 0.5 * x[:, 0] + rng.normal(size=72) + offsets
    result = permutation_test(
        loso_folds(groups),
        x,
        y,
        groups,
        None,
        Pipeline([("regressor", LinearRegression())]),
        {},
        observed=0.0,
        config=NullConfig(n_permutations=30),
        inner=BY_SUBJECT,
        seed=0,
    )
    assert abs(float(np.mean(result.null))) < 0.3


def test_the_null_aggregates_subjects_with_the_callers_weighting() -> None:
    # Equal and trial-count weighting give different cohort values when subjects differ in
    # size; the null must use the weighting the observed value was computed with.
    rng = np.random.default_rng(1)
    groups = np.array(["s1"] * 30 + ["s2"] * 6 + ["s3"] * 6, dtype=object)
    x = rng.normal(size=(42, 2))
    y = x[:, 0] + rng.normal(size=42)
    nulls = [
        permutation_test(
            loso_folds(groups),
            x,
            y,
            groups,
            None,
            Pipeline([("regressor", LinearRegression())]),
            {},
            observed=0.0,
            config=NullConfig(n_permutations=10),
            inner=BY_SUBJECT,
            seed=0,
            aggregation=AggregationConfig(subject_weighting=weighting),
        ).null
        for weighting in ("equal", "trial_count")
    ]
    assert not np.allclose(nulls[0], nulls[1])


def test_the_p_value_is_taken_over_completed_permutations_only() -> None:
    # A draw that failed to fit is a fault, not a draw that fell short of the observed
    # value. Counting it in the denominator would shrink every p-value.
    result = permutation_test(
        FOLDS,
        X,
        Y,
        GROUPS,
        RUNS,
        Pipeline([("regressor", _DivergesOnSomeTargets(strategy="mean"))]),
        {},
        observed=np.inf,
        config=NullConfig(n_permutations=20, min_complete_fraction=0.0),
        inner=BY_SUBJECT,
        seed=0,
    )
    assert result.n_incomplete > 0 and result.null.size > 0
    assert result.p_value == pytest.approx(1.0 / (result.null.size + 1))


def test_a_draw_whose_inner_search_fails_counts_as_incomplete() -> None:
    # tune reports a failed search as a fit failure, and min_complete_fraction exists to
    # count exactly those; the null must not abort on the first one.
    result = permutation_test(
        FOLDS,
        X,
        Y,
        GROUPS,
        RUNS,
        Pipeline([("regressor", _DivergesOnSomeTargets(strategy="mean"))]),
        {"regressor__strategy": ["mean", "median"]},
        observed=0.0,
        config=NullConfig(n_permutations=20, min_complete_fraction=0.0),
        inner=BY_SUBJECT,
        seed=0,
    )
    assert result.n_incomplete > 0


def test_one_subject_with_constant_predictions_does_not_zero_the_whole_draw() -> None:
    # Subject 0's model predicts a constant, so its correlation is undefined and counts as
    # zero; the other subjects' correlations must still make up the draw.
    rng = np.random.default_rng(2)
    groups = np.repeat([f"s{i}" for i in range(4)], 24).astype(object)
    runs = np.tile(np.repeat(["r1", "r2", "r3"], 8), 4).astype(object)
    x = np.column_stack([np.repeat(np.arange(4.0), 24), rng.normal(size=96)])
    y = x[:, 1] + rng.normal(size=96)
    result = permutation_test(
        within_subject_folds(groups, runs, inner_splits=3, seed=0),
        x,
        y,
        groups,
        runs,
        Pipeline([("regressor", _ConstantForFirstSubject())]),
        {},
        observed=0.0,
        config=NullConfig(n_permutations=10),
        inner=InnerSplit(grouping="run", n_splits=2),
        seed=0,
    )
    assert np.any(result.null != 0.0)


def test_a_subject_with_a_single_run_is_still_shuffled_within_that_run() -> None:
    # One run is still a run whose trials are exchangeable. Skipping the subject would put
    # its real labels into every null draw.
    groups = np.repeat(["s1", "s2"], 8).astype(object)
    runs = np.array(["A"] * 4 + ["B"] * 4 + ["A"] * 8, dtype=object)
    y = np.arange(16, dtype=float)
    draws = [
        permute(
            y,
            groups,
            runs,
            config=NullConfig(scheme="within_subject_within_run"),
            rng=np.random.default_rng(seed),
        )
        for seed in range(20)
    ]
    assert any(not np.array_equal(draw[8:], y[8:]) for draw in draws)


@pytest.mark.parametrize("scheme", ["within_subject_within_run", "circular_shift_within_run"])
def test_a_run_scheme_refuses_trials_without_a_run_label(scheme: str) -> None:
    # Run structure is paradigm-specific. A trial with no run label has no run to be
    # exchanged within, so it is neither guessed into one nor left fixed in every draw.
    groups = np.full(16, "s1", dtype=object)
    runs = np.array(["A"] * 8 + ["B"] * 7 + [np.nan], dtype=object)
    with pytest.raises(ValueError, match="run labels"):
        permute(
            np.arange(16, dtype=float),
            groups,
            runs,
            trial_indices=np.arange(16),
            config=NullConfig(scheme=scheme, min_retained_trials=2),  # type: ignore[arg-type]
            rng=np.random.default_rng(0),
        )
