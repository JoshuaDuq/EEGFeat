import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eegfeat.io as io_module
from eegfeat.bands import Band
from eegfeat.io import read_table, write_table
from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable

ALPHA = Band("alpha", 8.0, 13.0)


def _meta(**overrides: object) -> FeatureMeta:
    fields: dict[str, object] = {
        "measure": "power",
        "band": ALPHA,
        "space": "Cz",
        "space_kind": "channel",
        "window": "stim",
        "normalization": "log10",
        "unit": "log10(V^2/Hz)",
        "source": "welch",
        "window_bounds": (0.0, 1.0),
        "computation": ComputationSpec.create("welch", n_fft=512, window="hann"),
        "freq_resolution_hz": 0.5,
    }
    fields.update(overrides)
    return FeatureMeta(**fields)  # type: ignore[arg-type]


def _epoch_table() -> FeatureTable:
    return FeatureTable(
        # 1/3 needs all 17 significant digits, so a lossy float format cannot pass.
        values=np.array([[1.25, np.nan], [-3.5, 1.0 / 3.0], [2.0e-12, 7.0]]),
        coverage=np.array([[1.0, 0.0], [0.5, 1.0], [1.0, 0.75]]),
        meta=(
            _meta(),
            _meta(
                measure="slope",
                band=None,
                space="global",
                space_kind="global",
                window=None,
                window_bounds=None,
                normalization="raw",
                unit="a.u.",
                freq_resolution_hz=None,
            ),
        ),
        flags={"cog_fallback": np.array([[False, False], [True, False], [False, True]])},
        row_ids=(
            ("sub-01_task-test", 0, "left"),
            ("sub-01_task-test", 1, "right"),
            ("sub-01_task-test", 2, "left"),
        ),
    )


def _group_table() -> FeatureTable:
    return FeatureTable(
        values=np.array([[0.4], [0.9]]),
        coverage=np.array([[1.0], [0.5]]),
        meta=(_meta(measure="itpc", normalization="raw", unit="a.u."),),
        row_labels=("left", "right"),
    )


def _assert_same_table(actual: FeatureTable, expected: FeatureTable) -> None:
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    assert actual.meta == expected.meta
    assert actual.row_labels == expected.row_labels
    assert actual.row_ids == expected.row_ids
    assert set(actual.flags) == set(expected.flags)
    for key, flag in expected.flags.items():
        np.testing.assert_array_equal(actual.flags[key], flag)


def test_epoch_table_round_trips_exactly(tmp_path) -> None:
    table = _epoch_table()
    write_table(table, tmp_path / "sub-01_features.tsv")

    _assert_same_table(read_table(tmp_path / "sub-01_features.tsv"), table)


def test_group_table_round_trips_with_its_row_labels(tmp_path) -> None:
    table = _group_table()
    write_table(table, tmp_path / "sub-01_crosstrial.tsv")

    _assert_same_table(read_table(tmp_path / "sub-01_crosstrial.tsv"), table)


def test_write_returns_values_coverage_and_sidecar_paths(tmp_path) -> None:
    written = write_table(_epoch_table(), tmp_path / "sub-01_features.tsv")

    assert written == (
        tmp_path / "sub-01_features.tsv",
        tmp_path / "sub-01_features_coverage.tsv",
        tmp_path / "sub-01_features.json",
    )
    assert all(path.is_file() for path in written)


def test_bundle_publish_failure_restores_all_previous_files(tmp_path, monkeypatch) -> None:
    target = tmp_path / "sub-01_features.tsv"
    write_table(_epoch_table(), target)
    paths = (target, target.with_name(f"{target.stem}_coverage.tsv"), target.with_suffix(".json"))
    previous = {path: path.read_bytes() for path in paths}

    import eegfeat.io as io

    real_replace = io.os.replace
    failed = False

    def fail_during_publish(source, destination):
        nonlocal failed
        destination = Path(destination)
        if not failed and destination == paths[1] and Path(source).parent != destination.parent:
            failed = True
            raise OSError("intentional publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(io.os, "replace", fail_during_publish)
    with pytest.raises(OSError, match="intentional"):
        write_table(_epoch_table(), target)
    assert all(path.read_bytes() == content for path, content in previous.items())


def test_values_file_leads_with_the_row_key_then_descriptors(tmp_path) -> None:
    rows = pd.DataFrame({"event": ["left", "right", "left"], "rating": [3, 5, 4]})
    write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)

    frame = pd.read_csv(tmp_path / "t.tsv", sep="\t")
    assert list(frame.columns[:3]) == ["epoch", "event", "rating"]
    assert frame["epoch"].tolist() == [0, 1, 2]
    assert frame["rating"].tolist() == [3, 5, 4]


def test_descriptor_columns_are_not_read_back_as_features(tmp_path) -> None:
    rows = pd.DataFrame({"event": ["left", "right", "left"], "rating": [3, 5, 4]})
    write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)

    assert read_table(tmp_path / "t.tsv").names == _epoch_table().names


def test_group_rows_are_keyed_by_their_labels(tmp_path) -> None:
    write_table(_group_table(), tmp_path / "t.tsv")

    frame = pd.read_csv(tmp_path / "t.tsv", sep="\t")
    assert frame.columns[0] == "group"
    assert frame["group"].tolist() == ["left", "right"]


def test_missing_values_are_written_as_bids_na(tmp_path) -> None:
    write_table(_epoch_table(), tmp_path / "t.tsv")

    second_row = (tmp_path / "t.tsv").read_text().splitlines()[1].split("\t")
    assert second_row[-1] == "n/a"


def test_sidecar_describes_every_column_with_its_band_bounds(tmp_path) -> None:
    write_table(_epoch_table(), tmp_path / "t.tsv", provenance={"input": "sub-01_epo.fif"})

    sidecar = json.loads((tmp_path / "t.json").read_text())
    power, slope = sidecar["columns"]
    assert power["name"] == _meta().name
    assert power["band"] == {"name": "alpha", "fmin": 8.0, "fmax": 13.0}
    assert slope["band"] is None
    assert sidecar["provenance"] == {"input": "sub-01_epo.fif"}


def test_rejects_a_path_that_is_not_tsv(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\.tsv"):
        write_table(_epoch_table(), tmp_path / "t.csv")


def test_rejects_descriptor_rows_of_the_wrong_length(tmp_path) -> None:
    with pytest.raises(ValueError, match="3 rows"):
        write_table(_epoch_table(), tmp_path / "t.tsv", rows=pd.DataFrame({"event": ["a"]}))


def test_rejects_a_descriptor_that_shadows_the_row_key(tmp_path) -> None:
    rows = pd.DataFrame({"epoch": [7, 8, 9]})
    with pytest.raises(ValueError, match="epoch"):
        write_table(_epoch_table(), tmp_path / "t.tsv", rows=rows)


def test_read_fails_when_the_values_file_lacks_a_described_column(tmp_path) -> None:
    write_table(_epoch_table(), tmp_path / "t.tsv")
    frame = pd.read_csv(tmp_path / "t.tsv", sep="\t", keep_default_na=False)
    missing_name = _epoch_table().names[1]
    frame.drop(columns=[missing_name]).to_csv(tmp_path / "t.tsv", sep="\t", index=False)

    with pytest.raises(ValueError, match=missing_name):
        read_table(tmp_path / "t.tsv")


def test_read_dataset_stacks_tables_and_restores_aligned_targets(tmp_path) -> None:
    first = _epoch_table()
    second = FeatureTable(
        values=first.values + 10.0,
        coverage=first.coverage,
        meta=first.meta,
        flags=first.flags,
        row_ids=tuple(("sub-02_task-test", epoch, event) for _, epoch, event in first.row_ids),
    )
    paths = [tmp_path / "sub-01_features.tsv", tmp_path / "sub-02_features.tsv"]
    for path, table, ratings in zip(paths, (first, second), ([3, 5, 4], [2, 1, 0]), strict=True):
        rows = pd.DataFrame({"event": [event for _, _, event in table.row_ids], "rating": ratings})
        write_table(table, path, rows=rows)

    dataset = io_module.read_dataset(paths)

    assert dataset.table.row_ids == first.row_ids + second.row_ids
    np.testing.assert_array_equal(dataset.table.values, np.vstack([first.values, second.values]))
    assert list(dataset.targets.columns) == ["recording", "epoch", "event", "rating"]
    assert dataset.targets["recording"].tolist() == [
        "sub-01_task-test",
        "sub-01_task-test",
        "sub-01_task-test",
        "sub-02_task-test",
        "sub-02_task-test",
        "sub-02_task-test",
    ]
    assert dataset.targets["rating"].tolist() == [3, 5, 4, 2, 1, 0]


def test_read_dataset_refuses_descriptor_event_that_disagrees_with_row_identity(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    rows = pd.DataFrame({"event": ["wrong", "right", "left"]})
    write_table(_epoch_table(), path, rows=rows)

    with pytest.raises(ValueError, match="event.*row_ids"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_epoch_key_that_disagrees_with_row_identity(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path)
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    frame.loc[0, "epoch"] = 99
    frame.to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="epoch.*row_ids"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_fractional_epoch_key_instead_of_truncating_it(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path)
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    frame["epoch"] = frame["epoch"].astype(float)
    frame.loc[0, "epoch"] = 0.5
    frame.to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="epoch.*row_ids"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_cross_trial_tables(tmp_path) -> None:
    path = tmp_path / "crosstrial.tsv"
    write_table(_group_table(), path)

    with pytest.raises(ValueError, match="per-epoch"):
        io_module.read_dataset([path])


def test_read_dataset_refuses_a_missing_descriptor_column(tmp_path) -> None:
    path = tmp_path / "features.tsv"
    write_table(_epoch_table(), path, rows=pd.DataFrame({"rating": [3, 5, 4]}))
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    frame.drop(columns="rating").to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="descriptor columns.*rating"):
        io_module.read_dataset([path])


def test_read_dataset_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        io_module.read_dataset([])
