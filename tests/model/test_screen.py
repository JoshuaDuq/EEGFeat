from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.screen import univariate_screen

GROUPS = np.repeat([f"s{i:02d}" for i in range(12)], 30).astype(object)


def _noise(seed: int, n_features: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(GROUPS.size, n_features)), rng.normal(size=GROUPS.size)


def test_a_feature_that_tracks_the_target_within_subjects_survives_the_family() -> None:
    X, y = _noise(0)
    X[:, 0] = y + np.random.default_rng(1).normal(size=GROUPS.size)
    screen = univariate_screen(X, y, GROUPS, n_flips=999, seed=0)
    assert screen["p_fwer"].iloc[0] < 0.01
    assert (screen["p_fwer"].iloc[1:] > 0.05).all()
    assert screen["r"].iloc[0] == pytest.approx(1 / np.sqrt(2), abs=0.1)


def test_differences_between_subjects_do_not_count_as_tracking() -> None:
    # Subjects with a higher mean target also have a higher feature, but within every
    # subject the feature follows nothing: pooled r is large, within-subject r is not.
    X, y = _noise(2, n_features=1)
    offset = np.repeat(np.arange(12) * 5.0, 30)
    y, X[:, 0] = y + offset, X[:, 0] + offset
    screen = univariate_screen(X, y, GROUPS, n_flips=999, seed=0)
    assert np.corrcoef(X[:, 0], y)[0, 1] > 0.9
    assert abs(screen["r"].iloc[0]) < 0.1
    assert screen["p"].iloc[0] > 0.05


def test_the_family_wise_error_is_held_at_its_level_under_the_null() -> None:
    # Features share variance, as channels do, which a max-statistic null keeps.
    hits = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        shared = rng.normal(size=(GROUPS.size, 1))
        X = shared + rng.normal(size=(GROUPS.size, 20))
        y = rng.normal(size=GROUPS.size)
        hits += bool(
            (univariate_screen(X, y, GROUPS, n_flips=199, seed=seed)["p_fwer"] <= 0.05).any()
        )
    assert hits / 200 <= 0.08


def test_a_shared_nuisance_is_removed_from_both_sides_within_each_subject() -> None:
    rng = np.random.default_rng(3)
    stimulus = rng.choice([-1.0, 0.0, 1.0], size=GROUPS.size)
    slope = np.repeat(rng.uniform(0.5, 2.0, 12), 30)
    y = slope * stimulus + 0.5 * rng.normal(size=GROUPS.size)
    X = np.column_stack([slope * stimulus + 0.5 * rng.normal(size=GROUPS.size)])
    raw = univariate_screen(X, y, GROUPS, n_flips=199, seed=0)
    adjusted = univariate_screen(
        X,
        y,
        GROUPS,
        covariates=stimulus.reshape(-1, 1),
        residualize_on=("stimulus",),
        n_flips=199,
        seed=0,
    )
    assert raw["r"].iloc[0] > 0.5
    assert abs(adjusted["r"].iloc[0]) < 0.15


def test_a_feature_constant_within_every_subject_is_not_tested_after_residualizing() -> None:
    X, y = _noise(5, n_features=2)
    X[:, 1] = 3.7
    stimulus = np.random.default_rng(6).choice([-1.0, 0.0, 1.0], size=GROUPS.size)
    screen = univariate_screen(
        X,
        y,
        GROUPS,
        covariates=stimulus.reshape(-1, 1),
        residualize_on=("stimulus",),
        n_flips=99,
        seed=0,
    )
    assert screen["n_subjects"].iloc[1] == 0 and np.isnan(screen["p"].iloc[1])


def test_a_subject_missing_a_feature_drops_out_of_that_feature_only() -> None:
    X, y = _noise(4, n_features=3)
    X[GROUPS == "s00", 1] = np.nan
    X[~np.isin(GROUPS, ["s00", "s01"]), 2] = np.nan
    screen = univariate_screen(X, y, GROUPS, feature_names=["a", "b", "c"], n_flips=99, seed=0)
    assert screen.loc["a", "n_subjects"] == 12
    assert screen.loc["b", "n_subjects"] == 11
    # Two subjects cannot estimate a spread across subjects.
    assert screen.loc["c", "n_subjects"] == 2 and np.isnan(screen.loc["c", "p_fwer"])
