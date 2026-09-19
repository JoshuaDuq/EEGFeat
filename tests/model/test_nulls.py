from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.nulls import (
    NullConfig,
    changed_fraction,
    circular_shift_group,
    permutation_test,
    permute,
)
from eegfeat.model.splits import InnerSplit, loso_folds

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
    assert result.null.size == 20


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
    one_run = np.full(GROUPS.size, "r1", dtype=object)
    with pytest.raises(ValueError, match="never changes"):
        permutation_test(
            FOLDS,
            X,
            Y,
            GROUPS,
            one_run,
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
            config=NullConfig(scheme="circular_shift_within_run", min_retained_trials=8),
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

