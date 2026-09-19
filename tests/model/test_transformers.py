from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.transformers import (
    DropAllNaNColumns,
    PreprocessingConfig,
    ReplaceInfWithNaN,
    SpatialFeatureSelector,
    validate_subject_missingness,
)


def test_a_subject_above_the_missingness_limit_is_named_in_the_error() -> None:
    values = np.array([[1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan]])
    groups = np.array(["s1", "s2", "s2"], dtype=object)
    with pytest.raises(ValueError, match="s2"):
        validate_subject_missingness(values, groups, maximum=0.2)


def test_missingness_is_checked_per_subject_not_over_the_pooled_matrix() -> None:
    # Pooled missingness here is 25%, under the limit; s2's is 100%. A subject with no
    # usable features must fail even when the cohort average looks acceptable.
    values = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    groups = np.array(["s1", "s1", "s2"], dtype=object)
    with pytest.raises(ValueError, match="s2"):
        validate_subject_missingness(values, groups, maximum=0.3)


def test_missingness_requires_at_least_one_retained_feature() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_subject_missingness(
            np.empty((2, 0)), np.array(["s1", "s2"], dtype=object), maximum=0.5
        )


def test_infinities_become_nan_so_imputation_can_see_them() -> None:
    out = ReplaceInfWithNaN().fit_transform(np.array([[1.0, np.inf], [-np.inf, 2.0]]))
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])


def test_all_nan_columns_are_dropped_and_the_drop_is_learned_on_fit() -> None:
    # The columns to drop are decided by the training block and reapplied to test data,
    # so a column that happens to be present at test time is still dropped.
    train = np.array([[1.0, np.nan], [2.0, np.nan]])
    step = DropAllNaNColumns().fit(train)
    assert step.transform(np.array([[3.0, 9.0]])).shape == (1, 1)


def test_preprocessing_config_validates_missingness_bounds() -> None:
    with pytest.raises(ValueError, match="max_feature_missingness"):
        PreprocessingConfig(max_feature_missingness=1.5)
    with pytest.raises(ValueError, match="max_subject_missingness"):
        PreprocessingConfig(max_subject_missingness=-0.1)


def test_spatial_feature_selector_requires_feature_names_when_regions_are_requested() -> None:
    selector = SpatialFeatureSelector(allowed_regions=("insula",))
    with pytest.raises(ValueError, match="feature names"):
        selector.fit(np.ones((4, 3), dtype=float))
