from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.scoring import pearsonr_scorer, safe_pearsonr, scoring_dict


def test_perfect_correlation_is_one() -> None:
    r, _ = safe_pearsonr(np.arange(5.0), 2.0 * np.arange(5.0))
    assert r == pytest.approx(1.0)


def test_a_constant_predictor_gives_nan_rather_than_a_divide_by_zero() -> None:
    # Zero variance makes Pearson undefined. Returning NaN keeps the fold in the record
    # as unscored; returning 0.0 would claim a measured absence of correlation.
    r, p = safe_pearsonr(np.zeros(5), np.arange(5.0))
    assert np.isnan(r) and np.isnan(p)


def test_fewer_than_two_finite_pairs_gives_nan() -> None:
    r, p = safe_pearsonr(np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(r) and np.isnan(p)


def test_non_finite_pairs_are_dropped_not_propagated() -> None:
    r, _ = safe_pearsonr(
        np.array([1.0, 2.0, 3.0, np.nan]), np.array([2.0, 4.0, 6.0, 1.0])
    )
    assert r == pytest.approx(1.0)


def test_scoring_dict_exposes_the_correlation_scorer() -> None:
    scores = scoring_dict()
    assert "r" in scores


def test_pearsonr_scorer_callable() -> None:
    scorer = pearsonr_scorer()
    assert callable(scorer)
