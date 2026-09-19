from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegfeat.model.design import Selection, build_design, harmonize_fold, select
from eegfeat.table import FeatureMeta, FeatureTable


def test_every_selection_field_is_a_real_feature_meta_field() -> None:
    # Selection keys are metadata field names rather than name fragments, so a measure
    # that gets renamed upstream fails loudly here instead of selecting nothing.
    assert set(Selection.__dataclass_fields__) <= set(FeatureMeta.__dataclass_fields__)


def test_selecting_a_band_keeps_only_that_band(alpha_beta_table: FeatureTable) -> None:
    kept = select(alpha_beta_table, Selection(band=("alpha",)))
    assert {m.band.name for m in kept.meta if m.band is not None} == {"alpha"}


def test_an_empty_field_places_no_restriction(alpha_beta_table: FeatureTable) -> None:
    kept = select(alpha_beta_table, Selection())
    assert kept.values.shape == alpha_beta_table.values.shape


def test_a_selection_that_matches_nothing_raises_rather_than_returning_empty(
    alpha_beta_table: FeatureTable,
) -> None:
    with pytest.raises(ValueError, match="no columns"):
        select(alpha_beta_table, Selection(band=("delta",)))


def test_the_join_uses_the_whole_row_id_not_the_epoch_index(
    alpha_beta_table: FeatureTable,
) -> None:
    # Two recordings both number their epochs from zero. Joining on the epoch index alone
    # matches sub-02's epoch 0 to sub-01's target and raises nothing.
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(alpha_beta_table, targets, target="pain")
    assert design.row_ids == (("sub-01", 0, "stim"), ("sub-02", 0, "stim"))
    np.testing.assert_array_equal(design.y, [1.0, 9.0])


def test_a_target_row_matching_two_feature_rows_is_refused(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-01"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 2.0],
            "subject_id": ["sub-01", "sub-01"],
        }
    )
    with pytest.raises(ValueError, match="one-to-one"):
        build_design(alpha_beta_table, targets, target="pain")


def test_covariate_columns_are_identified_not_merely_counted(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "age": [30.0, 40.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(alpha_beta_table, targets, target="pain", covariates=["age"])
    assert design.column_names[design.covariate_columns[0]] == "age"
    assert design.covariate_columns.size == 1


def test_covariates_including_target_is_rejected_as_leakage(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    with pytest.raises(ValueError, match="leak"):
        build_design(alpha_beta_table, targets, target="pain", covariates=["pain"])


def test_missing_covariates_raises_when_strict(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        build_design(
            alpha_beta_table, targets, target="pain", covariates=["missing_col"],
            strict_covariates=True,
        )


def test_missing_covariates_dropped_when_not_strict(
    alpha_beta_table: FeatureTable,
) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(
        alpha_beta_table, targets, target="pain", covariates=["missing_col"],
        strict_covariates=False,
    )
    assert design.covariate_columns.size == 0


def test_harmonize_fold_drops_features_missing_in_any_training_group() -> None:
    X_tr = np.array(
        [
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=float,
    )
    X_te = np.array([[5.0, 50.0]], dtype=float)
    groups = np.array(["sub-01", "sub-01", "sub-02", "sub-02"], dtype=object)

    X_tr_h, X_te_h, keep = harmonize_fold(X_tr, X_te, groups, mode="intersection")
    assert np.array_equal(keep, [True, True])  # sub-01 has 10.0 in row 0, sub-02 has 30, 40

    # If sub-01 has all NaNs in column 1:
    X_tr[0, 1] = np.nan
    X_tr_h, X_te_h, keep = harmonize_fold(X_tr, X_te, groups, mode="intersection")
    assert np.array_equal(keep, [True, False])
    assert X_tr_h.shape == (4, 1)
    assert X_te_h.shape == (1, 1)


def test_harmonize_fold_rejects_empty_strict_intersection() -> None:
    X_tr = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
            [np.nan, 3.0],
            [np.nan, 4.0],
        ],
        dtype=float,
    )
    X_te = np.array([[1.0, 1.0]], dtype=float)
    groups = np.array(["sub-01", "sub-01", "sub-02", "sub-02"], dtype=object)

    with pytest.raises(ValueError, match="No features are finite for every training group"):
        harmonize_fold(X_tr, X_te, groups, mode="intersection")
