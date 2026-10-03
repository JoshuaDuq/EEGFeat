import argparse
import hashlib
import json
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import yaml

from eegtable.io import write_table
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable


def _api():
    assert find_spec("eegtable.runner.model_recipe") is not None, "model recipe is not implemented"
    assert find_spec("eegtable.runner.model_run") is not None, "model run is not implemented"
    from eegtable.runner.model_recipe import load_model_recipe
    from eegtable.runner.model_run import check_model, run_model

    return load_model_recipe, check_model, run_model


def _feature_meta(name):
    return FeatureMeta(
        measure=name,
        band=None,
        space="C3",
        space_kind="channel",
        window="all",
        normalization="raw",
        unit="a.u.",
        source="synthetic",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("synthetic"),
    )


def _inputs(tmp_path, task="regression", rows="epochs"):
    paths = []
    rng = np.random.default_rng(31)
    for subject in range(4):
        n_rows = 12
        target = (
            np.tile([0, 1], n_rows // 2) if task == "classification" else rng.normal(size=n_rows)
        )
        values = np.column_stack(
            [target + rng.normal(scale=0.1, size=n_rows), rng.normal(size=n_rows)]
        )
        coverage = np.ones(values.shape)
        coverage[0, 1] = 0.4
        table = FeatureTable(
            values=values,
            coverage=coverage,
            meta=(_feature_meta("signal"), _feature_meta("noise")),
            row_ids=(
                tuple((f"sub-{subject}", row, "event") for row in range(n_rows))
                if rows == "epochs"
                else None
            ),
            row_labels=(
                tuple(f"condition-{row}" for row in range(n_rows)) if rows == "groups" else None
            ),
            flags={"bad": np.zeros(values.shape, dtype=bool)},
        )
        descriptors = pd.DataFrame(
            {
                "recording": [f"sub-{subject}"] * n_rows,
                "subject_id": [f"s{subject}"] * n_rows,
                "outcome": target,
            }
        )
        if rows == "epochs":
            descriptors["event"] = "event"
        else:
            descriptors["n_trials"] = 20
        path = tmp_path / f"sub-{subject}_features.tsv"
        write_table(table, path, rows=descriptors)
        paths.append(path.name)
    return paths


def _recipe(tmp_path, task="regression", rows="epochs"):
    paths = _inputs(tmp_path, task, rows)
    record = {
        "version": 1,
        "inputs": {"paths": paths, "rows": rows},
        "analysis": {"task": task, "target": "outcome", "groups": "subject_id"},
        "model": {
            "estimator": "ridge" if task == "regression" else "logistic",
            "grid": {"regressor__alpha" if task == "regression" else "lr__C": [0.1, 1.0]},
        },
        "validation": {"outer": "loso", "inner_splits": 2, "seed": 13},
        "quality": {"min_coverage": 0.8, "rejected_flags": ["bad"]},
        "output": "results",
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(record))
    return path, record


def test_model_recipe_resolves_paths_and_quality(tmp_path):
    path, _ = _recipe(tmp_path)
    load, _, _ = _api()
    recipe = load(path)
    assert recipe.inputs.paths[0] == tmp_path / "sub-0_features.tsv"
    assert recipe.output == tmp_path / "results"
    assert recipe.quality.min_coverage == 0.8
    assert recipe.validation.scoring == "neg_mean_squared_error"


@pytest.mark.parametrize(
    "change", ["unknown", "duplicate", "bool_seed", "invalid_estimator", "bad_grid"]
)
def test_model_recipe_refuses_invalid_settings(tmp_path, change):
    path, record = _recipe(tmp_path)
    if change == "unknown":
        record["fallback"] = True
    if change == "bool_seed":
        record["validation"]["seed"] = True
    if change == "invalid_estimator":
        record["model"]["estimator"] = "svm"
    if change == "bad_grid":
        record["model"]["grid"] = {"regressor__alpha": []}
    path.write_text(yaml.safe_dump(record))
    if change == "duplicate":
        path.write_text(path.read_text() + "version: 1\n")
    load, _, _ = _api()
    with pytest.raises(ValueError):
        load(path)


def test_model_check_validates_nested_group_splits_without_writing(tmp_path):
    path, _ = _recipe(tmp_path)
    load, check, _ = _api()
    report = check(load(path))
    assert report.n_rows == 48
    assert report.n_groups == 4
    assert report.n_features == 2
    assert report.n_folds == 4
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("rows", ["epochs", "groups"])
def test_model_check_keeps_numeric_looking_groups_distinct(tmp_path, rows):
    from eegtable.io import read_table

    path, settings = _recipe(tmp_path, rows=rows)
    row_key = "epoch" if rows == "epochs" else "group"
    for name, label in zip(settings["inputs"]["paths"], ["01", "1", "02", "2"], strict=True):
        source = tmp_path / name
        table = read_table(source)
        sidecar = json.loads(source.with_suffix(".json").read_text())
        descriptors = pd.read_csv(source, sep="\t", usecols=sidecar["row_columns"])
        descriptors["subject_id"] = label
        write_table(table, source, rows=descriptors.drop(columns=row_key))
    load, check, _ = _api()

    assert check(load(path)).n_groups == 4


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_model_run_writes_verifiable_prediction_and_quality_bundle(tmp_path, task):
    path, _ = _recipe(tmp_path, task)
    load, _, run = _api()
    result = run(load(path))
    predictions = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    assert len(predictions) == 48
    assert not predictions.duplicated(["recording", "epoch", "event"]).any()
    assert predictions.y_pred.notna().all()
    manifest = json.loads((result.output / "manifest.json").read_text())
    assert manifest["task"] == task
    assert manifest["provenance"]["implementation_hash"]
    assert manifest["provenance"]["software"]["scikit-learn"]
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((result.output / name).read_bytes()).hexdigest() == expected_hash
    folds = json.loads((result.output / "folds.json").read_text())
    assert len(folds) == 4
    for fold in folds:
        assert not set(fold["train_groups"]) & set(fold["test_groups"])
        assert fold["best_params"]
        for split in fold["inner_splits"]:
            assert not set(split["train_groups"]) & set(split["test_groups"])
    quality = pd.read_csv(result.output / "quality_ledger.tsv", sep="\t")
    assert len(quality) == 4
    assert (result.output / "design_features_coverage.tsv").is_file()
    assert (result.output / "design_features.json").is_file()
    if task == "classification":
        np.testing.assert_allclose(predictions.probability_0 + predictions.probability_1, 1)
    assert "overall" in json.loads((result.output / "metrics.json").read_text())


def test_model_run_handles_group_rows_without_epoch_broadcast(tmp_path):
    path, _ = _recipe(tmp_path, rows="groups")
    load, _, run = _api()
    result = run(load(path))
    frame = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    assert "group" in frame and "epoch" not in frame
    assert len(frame) == 48


def test_model_run_rejects_existing_results(tmp_path):
    path, _ = _recipe(tmp_path)
    load, _, run = _api()
    (tmp_path / "results").mkdir()
    sentinel = tmp_path / "results" / "sentinel.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        run(load(path))
    assert sentinel.read_text() == "preserve"


def test_model_check_rejects_insufficient_nested_training_groups(tmp_path):
    path, record = _recipe(tmp_path)
    record["validation"]["inner_splits"] = 4
    path.write_text(yaml.safe_dump(record))
    load, check, _ = _api()
    with pytest.raises(ValueError, match="training groups"):
        check(load(path))


def test_model_check_rejects_nonbinary_classification_labels(tmp_path):
    path, _ = _recipe(tmp_path, task="classification")
    file = tmp_path / "sub-0_features.tsv"
    from eegtable.io import read_table

    table = read_table(tmp_path / "sub-1_features.tsv")
    table = FeatureTable(
        values=table.values,
        coverage=table.coverage,
        meta=table.meta,
        row_ids=tuple(("sub-0", index, "event") for index in range(12)),
        flags=table.flags,
    )
    descriptors = pd.DataFrame(
        {
            "recording": ["sub-0"] * 12,
            "subject_id": ["s0"] * 12,
            "event": ["event"] * 12,
            "outcome": np.r_[0.5, np.tile([1, 0], 6)[:11]],
        }
    )
    write_table(table, file, rows=descriptors)
    load, check, _ = _api()
    with pytest.raises(ValueError, match="0/1"):
        check(load(path))


def test_model_cli_init_and_check(tmp_path, capsys):
    assert find_spec("eegtable.runner.model_command") is not None, "model CLI is not implemented"
    from eegtable.runner.model_command import register

    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    output = tmp_path / "starter.yaml"
    args = parser.parse_args(["model", "init", str(output)])
    assert args.handler(args) == 0
    assert output.is_file()
    path, _ = _recipe(tmp_path)
    args = parser.parse_args(["model", "check", str(path)])
    assert args.handler(args) == 0
    assert "48" in capsys.readouterr().out


def test_model_recipe_requires_integer_version(tmp_path):
    path, record = _recipe(tmp_path)
    record["version"] = 1.0
    path.write_text(yaml.safe_dump(record))
    load, _, _ = _api()
    with pytest.raises(ValueError, match="version"):
        load(path)


def test_external_targets_join_by_exact_sample_identity(tmp_path):
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event", "outcome"]].copy()
    targets = targets.rename(columns={"outcome": "external_score"}).iloc[::-1]
    targets.to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["target"] = "external_score"
    path.write_text(yaml.safe_dump(record))
    load, _, run = _api()
    result = run(load(path))
    frame = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    np.testing.assert_allclose(frame.y_true, dataset.targets.outcome)


def test_external_targets_refuse_missing_identity(tmp_path):
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event", "outcome"]].iloc[:-1]
    targets.rename(columns={"outcome": "external_score"}).to_csv(
        tmp_path / "targets.tsv", sep="\t", index=False
    )
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["target"] = "external_score"
    path.write_text(yaml.safe_dump(record))
    load, check, _ = _api()
    with pytest.raises(ValueError, match="exactly"):
        check(load(path))


def test_external_group_labels_preserve_distinct_numeric_strings(tmp_path):
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event"]].copy()
    labels = ["001", "01", "1", "2"]
    targets["external_group"] = np.repeat(labels, 12)
    targets.to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["groups"] = "external_group"
    path.write_text(yaml.safe_dump(record))
    load, check, _ = _api()
    assert check(load(path)).n_groups == 4


def test_model_run_leaves_no_published_bundle_after_backend_error(tmp_path):
    path, record = _recipe(tmp_path)
    record["model"]["grid"] = {"regressor__alpha": [-1.0]}
    path.write_text(yaml.safe_dump(record))
    load, _, run = _api()
    with pytest.raises(ValueError):
        run(load(path))
    assert not (tmp_path / "results").exists()


def test_model_check_refuses_unknown_pipeline_grid_parameter(tmp_path):
    path, record = _recipe(tmp_path)
    record["model"]["grid"] = {"regressor__unknown": [1.0]}
    path.write_text(yaml.safe_dump(record))
    load, check, _ = _api()
    with pytest.raises(ValueError, match="unknown"):
        check(load(path))


def test_group_kfold_predictions_are_complete_and_reproducible(tmp_path):
    path, record = _recipe(tmp_path, task="classification")
    record["validation"].update(outer="group_kfold", outer_splits=2)
    path.write_text(yaml.safe_dump(record))
    load, _, run = _api()
    first = run(load(path))
    frame = pd.read_csv(first.output / "predictions.tsv", sep="\t")
    record["output"] = "second-results"
    path.write_text(yaml.safe_dump(record))
    second = run(load(path))
    repeated = pd.read_csv(second.output / "predictions.tsv", sep="\t")
    pd.testing.assert_frame_equal(frame, repeated)
    assert frame.fold.nunique() == 2


@pytest.mark.parametrize("changed", ["recipe", "implementation", "software"])
def test_model_run_refuses_changed_provenance_during_fitting(tmp_path, monkeypatch, changed):
    from eegtable.runner import model_run

    path, _ = _recipe(tmp_path)
    load, _, run = _api()
    predict = model_run._predictions

    def mutate_after_fitting(recipe, prepared):
        predictions = predict(recipe, prepared)
        if changed == "recipe":
            path.write_text(path.read_text() + "\n# modified during execution\n")
        elif changed == "implementation":
            monkeypatch.setattr(model_run, "implementation_hash", lambda: "modified")
        else:
            monkeypatch.setattr(model_run, "software_versions", lambda: {"python": "modified"})
        return predictions

    monkeypatch.setattr(model_run, "_predictions", mutate_after_fitting)
    with pytest.raises(ValueError, match="changed"):
        run(load(path))
    assert not (tmp_path / "results").exists()


def test_model_run_refuses_recipe_modified_after_loading(tmp_path):
    path, _ = _recipe(tmp_path)
    load, _, run = _api()
    recipe = load(path)
    path.write_text(path.read_text() + "\n# modified after loading\n")
    with pytest.raises(ValueError, match="recipe.*changed"):
        run(recipe)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "rows,role,name",
    [
        ("epochs", "target", "event"),
        ("epochs", "target", "epoch"),
        ("epochs", "groups", "epoch"),
        ("groups", "target", "group"),
        ("groups", "groups", "group"),
    ],
)
def test_model_bundle_preserves_canonical_identity_when_used_in_analysis(
    tmp_path, rows, role, name
):
    from eegtable.group import read_group_dataset
    from eegtable.io import read_dataset

    task = "classification" if name == "event" else "regression"
    path, record = _recipe(tmp_path, task=task, rows=rows)
    reader = read_dataset if rows == "epochs" else read_group_dataset
    paths = [tmp_path / filename for filename in record["inputs"]["paths"]]
    if name in ("event", "group"):
        for source in paths:
            dataset = reader([source])
            descriptors = dataset.targets.copy()
            labels = tuple(
                str(row % 2 if name == "event" else row) for row in range(dataset.table.n_rows)
            )
            descriptors[name] = labels
            if name == "event":
                assert dataset.table.row_ids is not None
                identities = tuple(
                    (recording, epoch, label)
                    for (recording, epoch, _), label in zip(
                        dataset.table.row_ids, labels, strict=True
                    )
                )
                table = replace(dataset.table, row_ids=identities)
            else:
                table = replace(dataset.table, row_labels=labels)
            descriptors = descriptors.drop(columns="epoch" if rows == "epochs" else "group")
            write_table(table, source, rows=descriptors)
    record["analysis"][role] = name
    path.write_text(yaml.safe_dump(record))
    load, check, run = _api()
    recipe = load(path)
    original = reader(paths)
    assert check(recipe).n_rows == original.table.n_rows
    result = run(recipe)
    restored = reader([result.output / "design_features.tsv"])
    keys = ["recording", "epoch", "event"] if rows == "epochs" else ["recording", "group"]
    pd.testing.assert_frame_equal(restored.targets[keys], original.targets[keys])
    matrix = np.load(result.output / "design_matrix.npz")
    np.testing.assert_array_equal(
        matrix["y"], pd.to_numeric(original.targets[record["analysis"]["target"]])
    )
