import numpy as np
import pandas as pd
import pytest

from eegfeat import Band
from eegfeat.io import write_table
from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable


def _table(space="C3"):
    return FeatureTable(
        values=np.array([[0.3], [0.7]]),
        coverage=np.ones((2, 1)),
        row_labels=("rest", "task"),
        meta=(
            FeatureMeta(
                measure="coh",
                band=Band("alpha", 8, 13),
                space=space,
                space_kind="channel",
                window="whole",
                normalization="raw",
                unit="unitless",
                source="multitaper",
                window_bounds=(0.0, 1.0),
                computation=ComputationSpec.create("coh"),
            ),
        ),
    )


def test_group_dataset_preserves_recording_and_trial_counts(tmp_path):
    from eegfeat.group import build_group_design, read_group_dataset

    paths = []
    for recording, space in (("sub01", "C3"), ("sub02", "C4")):
        path = tmp_path / f"{recording}.tsv"
        write_table(
            _table(space),
            path,
            rows=pd.DataFrame(
                {
                    "recording": [recording] * 2,
                    "n_trials": [10, 20],
                    "subject": [recording] * 2,
                    "outcome": [0, 1],
                }
            ),
        )
        paths.append(path)
    dataset = read_group_dataset(paths)
    assert dataset.table.row_ids is None
    assert dataset.table.values.shape == (4, 2)
    assert dataset.targets.n_trials.tolist() == [10, 20, 10, 20]
    assert dataset.targets.group.tolist() == ["rest", "task", "rest", "task"]
    design = build_group_design(dataset, target="outcome", groups="subject")
    assert design.sample_ids == (
        ("sub01", "rest"),
        ("sub01", "task"),
        ("sub02", "rest"),
        ("sub02", "task"),
    )
    assert design.groups.tolist() == ["sub01", "sub01", "sub02", "sub02"]
    np.testing.assert_equal(design.X, dataset.table.values)


def test_group_dataset_preserves_numeric_looking_subject_labels(tmp_path):
    from eegfeat.group import build_group_design, read_group_dataset

    path = tmp_path / "features.tsv"
    write_table(
        _table(),
        path,
        rows=pd.DataFrame(
            {
                "recording": ["recording"] * 2,
                "n_trials": [10, 20],
                "subject_id": ["01", "1"],
                "outcome": [0, 1],
            }
        ),
    )

    design = build_group_design(read_group_dataset([path]), target="outcome")

    assert design.groups.tolist() == ["01", "1"]


def test_group_dataset_rejects_duplicate_group_samples(tmp_path):
    from eegfeat.group import read_group_dataset

    path = tmp_path / "features.tsv"
    write_table(
        _table(), path, rows=pd.DataFrame({"recording": ["sub01"] * 2, "n_trials": [10, 20]})
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_group_dataset([path, path])


def test_group_dataset_requires_positive_trial_counts(tmp_path):
    from eegfeat.group import read_group_dataset

    path = tmp_path / "features.tsv"
    write_table(
        _table(), path, rows=pd.DataFrame({"recording": ["sub01"] * 2, "n_trials": [0, 20]})
    )
    with pytest.raises(ValueError, match="n_trials"):
        read_group_dataset([path])


def test_group_design_rejects_packaged_csp_features(tmp_path):
    from dataclasses import replace

    from eegfeat.group import build_group_design, read_group_dataset

    original = _table()
    table = replace(
        original,
        meta=(
            replace(
                original.meta[0],
                measure="csp_log_power",
                computation=ComputationSpec.create("csp_features"),
            ),
        ),
    )
    path = tmp_path / "csp.tsv"
    write_table(
        table,
        path,
        rows=pd.DataFrame(
            {
                "recording": ["sub01"] * 2,
                "n_trials": [10, 20],
                "subject_id": ["sub01"] * 2,
                "outcome": [0, 1],
            }
        ),
    )
    with pytest.raises(ValueError, match="CSP must be fitted inside"):
        build_group_design(read_group_dataset([path]), target="outcome")


def test_group_design_rejects_duplicate_canonical_samples():
    import json
    from dataclasses import replace

    from eegfeat.group import GroupDataset, build_group_design

    table = replace(
        _table(), row_labels=(json.dumps(["sub01", "rest"], separators=(",", ":")),) * 2
    )
    targets = pd.DataFrame(
        {
            "recording": ["sub01"] * 2,
            "group": ["rest"] * 2,
            "subject_id": ["sub01"] * 2,
            "outcome": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_group_design(GroupDataset(table, targets), target="outcome")


def test_group_dataset_rejects_blank_group_identity(tmp_path):
    from dataclasses import replace

    from eegfeat.group import read_group_dataset

    path = tmp_path / "features.tsv"
    write_table(
        replace(_table(), row_labels=("", "task")),
        path,
        rows=pd.DataFrame({"recording": ["sub01"] * 2, "n_trials": [10, 20]}),
    )
    with pytest.raises(ValueError, match="nonempty|blank"):
        read_group_dataset([path])
