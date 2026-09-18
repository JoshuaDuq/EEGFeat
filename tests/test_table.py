import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.table import FeatureMeta, FeatureTable, concat

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
    )


def _table(spaces: tuple[str, ...] = ("C3", "C4")) -> FeatureTable:
    n = len(spaces)
    return FeatureTable(
        values=np.arange(3 * n, dtype=float).reshape(3, n),
        coverage=np.ones((3, n)),
        meta=tuple(_meta(s) for s in spaces),
    )


def test_name_is_derived_from_metadata_fields() -> None:
    assert _meta().name == "eeg_power_alpha_c3_stim_log-ratio"


def test_dataframe_columns_are_the_canonical_names() -> None:
    df = _table().to_dataframe()
    assert list(df.columns) == [
        "eeg_power_alpha_c3_stim_log-ratio",
        "eeg_power_alpha_c4_stim_log-ratio",
    ]
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
    left = FeatureTable(
        values=np.zeros((3, 1)),
        coverage=np.ones((3, 1)),
        meta=(_meta("C3"),),
        flags={"edge_hit": np.ones((3, 1), dtype=bool)},
    )
    right = _table(("C4",))
    joined = concat([left, right])
    assert joined.values.shape == (3, 2)
    assert joined.flags["edge_hit"].shape == (3, 2)
    assert joined.flags["edge_hit"][:, 0].all()
    assert not joined.flags["edge_hit"][:, 1].any()


def test_concat_refuses_to_align_mismatched_epoch_counts() -> None:
    small = FeatureTable(np.zeros((2, 1)), np.ones((2, 1)), (_meta("C3"),))
    with pytest.raises(ValueError, match="n_epochs"):
        concat([small, _table(("C4",))])


def test_concat_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        concat([])
