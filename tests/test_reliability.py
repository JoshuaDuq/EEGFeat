from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.test_io import _epoch_table


def test_repeated_session_absolute_agreement_detects_offsets():
    from eegfeat.reliability import intraclass_reliability

    values = np.column_stack([np.array([1, 3, 2, 4, 3, 5], dtype=float)] * 2)
    table = replace(
        _epoch_table(), values=values, coverage=np.ones_like(values), row_ids=None, flags={}
    )
    descriptors = pd.DataFrame(
        {"subject_id": ["a", "a", "b", "b", "c", "c"], "session": ["one", "two"] * 3}
    )
    result = intraclass_reliability(table, descriptors)
    assert result.n_subjects.tolist() == [3, 3]
    assert result.icc_consistency.tolist() == pytest.approx([1.0, 1.0])
    assert result.icc_absolute.tolist() == pytest.approx([1 / 3, 1 / 3])


def test_reliability_rejects_unbalanced_or_duplicate_samples():
    from eegfeat.reliability import intraclass_reliability

    table = _epoch_table()
    descriptors = pd.DataFrame({"subject_id": ["a", "a", "b"], "session": ["one", "two", "one"]})
    with pytest.raises(ValueError, match="balanced"):
        intraclass_reliability(table, descriptors)
    descriptors.loc[1, "session"] = "one"
    with pytest.raises(ValueError, match="duplicate"):
        intraclass_reliability(table, descriptors)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_reliability_is_invariant_to_finite_feature_units(scale):
    from eegfeat.reliability import intraclass_reliability

    values = np.column_stack([np.array([1, 3, 2, 4, 3, 5], dtype=float)] * 2) * scale
    table = replace(
        _epoch_table(), values=values, coverage=np.ones_like(values), row_ids=None, flags={}
    )
    descriptors = pd.DataFrame(
        {"subject_id": ["a", "a", "b", "b", "c", "c"], "session": ["one", "two"] * 3}
    )
    result = intraclass_reliability(table, descriptors)
    assert result.icc_consistency.tolist() == pytest.approx([1.0, 1.0])
    assert result.icc_absolute.tolist() == pytest.approx([1 / 3, 1 / 3])


@pytest.mark.parametrize("offset", [-1e15, 1e15])
def test_reliability_preserves_exact_differences_with_a_large_common_offset(offset):
    from eegfeat.reliability import intraclass_reliability

    values = np.column_stack([np.array([1, 3, 2, 4, 3, 5], dtype=float)] * 2)
    shifted = values + offset
    np.testing.assert_array_equal(shifted - shifted[0], values - values[0])
    table = replace(
        _epoch_table(), values=shifted, coverage=np.ones_like(values), row_ids=None, flags={}
    )
    descriptors = pd.DataFrame(
        {"subject_id": ["a", "a", "b", "b", "c", "c"], "session": ["one", "two"] * 3}
    )
    result = intraclass_reliability(table, descriptors)
    assert result.icc_consistency.tolist() == pytest.approx([1.0, 1.0])
    assert result.icc_absolute.tolist() == pytest.approx([1 / 3, 1 / 3])
