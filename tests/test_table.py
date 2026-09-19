import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable, concat

ALPHA = Band("alpha", 8.0, 13.0)


def _meta(space: str = "C3", measure: str = "power") -> FeatureMeta:
    return FeatureMeta(
        measure=measure,
        band=ALPHA,
        space=space,
        space_kind="channel",
        window="stim",
        normalization="log_ratio",
        unit="log10",
        source="morlet",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("band_power", weighting="trapezoid"),
    )


def _table(spaces: tuple[str, ...] = ("C3", "C4")) -> FeatureTable:
    n = len(spaces)
    return FeatureTable(
        values=np.arange(3 * n, dtype=float).reshape(3, n),
        coverage=np.ones((3, n)),
        meta=tuple(_meta(s) for s in spaces),
    )


def test_name_is_derived_from_metadata_fields() -> None:
    assert _meta().name.startswith("eeg_power_alpha_c3_stim_log-ratio_p")


def test_parameter_hash_is_canonical_and_stable() -> None:
    left = ComputationSpec.create("burst", threshold=0.75, minimum_duration=0.1)
    right = ComputationSpec.create("burst", minimum_duration=0.1, threshold=0.75)

    assert left == right
    assert left.parameter_hash == right.parameter_hash


def test_window_bounds_and_parameters_distinguish_feature_identifiers() -> None:
    base = _meta()
    shifted = FeatureMeta(
        **{
            **base.__dict__,
            "window_bounds": (0.25, 1.25),
        }
    )
    thresholded = FeatureMeta(
        **{
            **base.__dict__,
            "computation": ComputationSpec.create(
                "band_power", weighting="trapezoid", threshold=0.75
            ),
        }
    )

    assert len({base.name, shifted.name, thresholded.name}) == 3


def test_dataframe_columns_are_the_canonical_names() -> None:
    df = _table().to_dataframe()
    assert list(df.columns) == [_meta("C3").name, _meta("C4").name]
    assert df.shape == (3, 2)


def test_select_filters_columns_by_metadata_not_by_string_matching() -> None:
    selected = _table().select(space="C4")
    assert len(selected.meta) == 1
    assert selected.meta[0].space == "C4"
    assert selected.values.shape == (3, 1)


def test_select_on_an_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="not a FeatureMeta field"):
        _table().select(channel="C4")


def test_duplicate_column_names_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FeatureTable(
            values=np.zeros((3, 2)),
            coverage=np.ones((3, 2)),
            meta=(_meta("C3"), _meta("C3")),
        )


def test_mismatched_meta_length_raises() -> None:
    with pytest.raises(ValueError, match="meta"):
        FeatureTable(values=np.zeros((3, 2)), coverage=np.ones((3, 2)), meta=(_meta("C3"),))


def test_flags_must_match_the_value_shape() -> None:
    with pytest.raises(ValueError, match="flag"):
        FeatureTable(
            values=np.zeros((3, 2)),
            coverage=np.ones((3, 2)),
            meta=(_meta("C3"), _meta("C4")),
            flags={"edge_hit": np.zeros((2, 2), dtype=bool)},
        )


def test_concat_joins_columns_and_preserves_flags() -> None:
    identities = tuple(("recording", epoch, "event") for epoch in range(3))
    left = FeatureTable(
        values=np.zeros((3, 1)),
        coverage=np.ones((3, 1)),
        meta=(_meta("C3"),),
        flags={"edge_hit": np.ones((3, 1), dtype=bool)},
        row_ids=identities,
    )
    right = FeatureTable(
        values=np.zeros((3, 1)),
        coverage=np.ones((3, 1)),
        meta=(_meta("C4"),),
        row_ids=identities,
    )
    joined = concat([left, right])
    assert joined.values.shape == (3, 2)
    assert joined.flags["edge_hit"].shape == (3, 2)
    assert joined.flags["edge_hit"][:, 0].all()
    assert not joined.flags["edge_hit"][:, 1].any()


def test_concat_refuses_to_align_mismatched_epoch_counts() -> None:
    small = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),))
    with pytest.raises(ValueError, match="n_rows"):
        concat([small, _table(("C4",))])


def test_concat_refuses_unidentified_epoch_rows() -> None:
    with pytest.raises(ValueError, match="row_ids"):
        concat([_table(("C3",)), _table(("C4",))])


def test_concat_requires_exact_epoch_identity_and_order() -> None:
    identities = (("sub-01_task-rest", 4, "eyes-open"), ("sub-01_task-rest", 9, "eyes-closed"))
    left = FeatureTable(
        np.zeros((2, 1)),
        np.ones((2, 1)),
        (_meta("C3"),),
        row_ids=identities,
    )
    reordered = FeatureTable(
        np.zeros((2, 1)),
        np.ones((2, 1)),
        (_meta("C4"),),
        row_ids=tuple(reversed(identities)),
    )

    with pytest.raises(ValueError, match="row identities"):
        concat([left, reordered])


def test_concat_preserves_matching_epoch_identities() -> None:
    identities = (("sub-01_task-rest", 4, "eyes-open"), ("sub-01_task-rest", 9, "eyes-closed"))
    left = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),), row_ids=identities)
    right = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C4"),), row_ids=identities)

    assert concat([left, right]).row_ids == identities


def test_concat_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        concat([])


# --- row semantics --------------------------------------------------------------------


def test_rows_are_epochs_unless_labelled() -> None:
    assert _table().row_labels is None
    assert _table().n_rows == 3


def test_labelled_rows_index_the_dataframe() -> None:
    table = FeatureTable(
        values=np.zeros((2, 1)),
        coverage=np.ones((2, 1)),
        meta=(_meta("C3"),),
        row_labels=("rest", "task"),
    )
    assert list(table.to_dataframe().index) == ["rest", "task"]


def test_row_labels_must_match_the_row_count() -> None:
    with pytest.raises(ValueError, match="row_labels"):
        FeatureTable(
            values=np.zeros((2, 1)),
            coverage=np.ones((2, 1)),
            meta=(_meta("C3"),),
            row_labels=("only-one",),
        )


def test_concat_refuses_to_join_group_rows_to_epoch_rows() -> None:
    per_epoch = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),))
    per_group = FeatureTable(
        np.zeros((2, 1)), np.ones((2, 1)), (_meta("C4"),), row_labels=("rest", "task")
    )
    with pytest.raises(ValueError, match="row semantics"):
        concat([per_epoch, per_group])


def test_select_preserves_row_labels() -> None:
    table = FeatureTable(
        values=np.zeros((2, 2)),
        coverage=np.ones((2, 2)),
        meta=(_meta("C3"), _meta("C4")),
        row_labels=("rest", "task"),
    )
    assert table.select(space="C4").row_labels == ("rest", "task")
