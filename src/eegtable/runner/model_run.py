"""Execute grouped model recipes and publish auditable result bundles."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml
from filelock import FileLock
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from eegtable.group import GroupDesign, build_group_design, read_group_dataset
from eegtable.io import read_dataset, write_table
from eegtable.model.crossfit import (
    FoldClassification,
    FoldPrediction,
    cross_fit_classification,
    cross_fit_regression,
)
from eegtable.model.design import Design, build_design
from eegtable.model.estimators import (
    elasticnet_pipeline,
    logistic_pipeline,
    random_forest_classifier_pipeline,
    random_forest_pipeline,
    ridge_pipeline,
    svm_pipeline,
)
from eegtable.model.metrics import classification_metrics, regression_metrics
from eegtable.model.splits import Fold, InnerSplit, inner_cv, loso_folds
from eegtable.model.tuning import _validate_grouped_calibration
from eegtable.provenance import (
    canonical_json,
    file_hash,
    implementation_hash,
    serializable,
    software_versions,
)
from eegtable.quality import feature_quality
from eegtable.runner.model_recipe import ModelRecipe
from eegtable.table import FeatureTable


@dataclass(frozen=True)
class ModelCheck:
    n_rows: int
    n_features: int
    n_groups: int
    n_folds: int
    output: Path


@dataclass(frozen=True)
class ModelResult:
    output: Path
    n_rows: int
    n_features: int
    n_folds: int


@dataclass(frozen=True)
class _Prepared:
    design: Design | GroupDesign
    identities: pd.DataFrame
    table: FeatureTable
    pipeline: Pipeline
    folds: tuple[Fold, ...]
    inner: InnerSplit
    evidence: list[dict[str, object]]
    provenance: dict[str, object]


_PIPELINES = {
    ("regression", "ridge"): ridge_pipeline,
    ("regression", "elasticnet"): elasticnet_pipeline,
    ("regression", "random_forest"): random_forest_pipeline,
    ("classification", "logistic"): logistic_pipeline,
    ("classification", "svm"): svm_pipeline,
    ("classification", "random_forest"): random_forest_classifier_pipeline,
}


def _merge_targets(
    descriptors: pd.DataFrame, external: Path, rows: str, groups: str
) -> pd.DataFrame:
    keys = ["recording", "epoch", "event"] if rows == "epochs" else ["recording", "group"]
    targets = pd.read_csv(
        external,
        sep="\t",
        keep_default_na=False,
        na_values=["n/a"],
        dtype={"recording": str, "event": str, "group": str, groups: str},
    )
    if not set(keys) <= set(targets) or targets[keys].isna().any().any():
        raise ValueError(f"external targets require complete identity columns {keys}.")
    if targets.duplicated(keys).any() or descriptors.duplicated(keys).any():
        raise ValueError("target joins require unique sample identities.")
    source_keys = set(map(tuple, descriptors[keys].to_numpy()))
    target_keys = set(map(tuple, targets[keys].to_numpy()))
    if source_keys != target_keys:
        raise ValueError(
            "external targets must match every feature sample exactly, with no extra rows."
        )
    overlap = {str(column) for column in set(targets) & set(descriptors) - set(keys)}
    if overlap:
        raise ValueError(
            f"external targets overlap existing descriptor columns: {sorted(overlap)}."
        )
    return descriptors.merge(targets, how="left", on=keys, sort=False, validate="one_to_one")


def _load_design(recipe: ModelRecipe) -> tuple[Design | GroupDesign, pd.DataFrame]:
    analysis = recipe.analysis
    design: Design | GroupDesign
    if recipe.inputs.rows == "epochs":
        dataset = read_dataset(recipe.inputs.paths, columns=recipe.inputs.columns)
        targets = dataset.targets
        if recipe.inputs.targets is not None:
            targets = _merge_targets(targets, recipe.inputs.targets, "epochs", analysis.groups)
        design = build_design(
            dataset.table,
            targets,
            target=analysis.target,
            groups=analysis.groups,
            covariates=analysis.covariates,
            selection=recipe.selection,
            quality=recipe.quality,
        )
        identities = pd.DataFrame(design.row_ids, columns=["recording", "epoch", "event"])
    else:
        group_dataset = read_group_dataset(recipe.inputs.paths, columns=recipe.inputs.columns)
        if recipe.inputs.targets is not None:
            group_dataset = replace(
                group_dataset,
                targets=_merge_targets(
                    group_dataset.targets, recipe.inputs.targets, "groups", analysis.groups
                ),
            )
        design = build_group_design(
            group_dataset,
            target=analysis.target,
            groups=analysis.groups,
            selection=recipe.selection,
            quality=recipe.quality,
        )
        identities = pd.DataFrame(design.sample_ids, columns=["recording", "group"])
        identities["n_trials"] = group_dataset.targets["n_trials"].to_numpy()
    if pd.isna(design.groups).any() or any(not str(group).strip() for group in design.groups):
        raise ValueError("model groups must have a nonempty label for every sample.")
    if analysis.task == "classification" and set(np.unique(design.y)) != {0, 1}:
        raise ValueError("classification requires both classes with labels coded exactly 0/1.")
    return design, identities


def _pipeline(recipe: ModelRecipe, design: Design | GroupDesign) -> Pipeline:
    factory = _PIPELINES[(recipe.analysis.task, recipe.model.estimator)]
    pipeline = factory(
        recipe.preprocessing, seed=recipe.validation.seed, n_covariates=design.n_covariates
    )
    allowed = pipeline.get_params(deep=True)
    for parameters in ParameterGrid(recipe.model.grid):
        unknown = set(parameters) - set(allowed)
        if unknown:
            raise ValueError(f"unknown estimator grid parameters: {sorted(unknown)}.")
        _validate_grouped_calibration(clone(pipeline).set_params(**parameters))
    return pipeline


def _folds(recipe: ModelRecipe, design: Design | GroupDesign) -> tuple[Fold, ...]:
    settings = recipe.validation
    count = np.unique(design.groups).size
    if count < 3:
        raise ValueError("nested modeling requires at least three distinct groups.")
    if settings.outer == "loso":
        return loso_folds(design.groups)
    assert settings.outer_splits is not None
    if count < settings.outer_splits:
        raise ValueError("outer_splits exceeds the number of model groups.")
    outer = inner_cv(design.groups, InnerSplit("subject", n_splits=settings.outer_splits))
    return tuple(
        Fold(index, np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
        for index, (train, test) in enumerate(
            outer.split(design.X, design.y, groups=design.groups), 1
        )
    )


def _fold_evidence(
    recipe: ModelRecipe, design: Design | GroupDesign, folds: Sequence[Fold], inner: InnerSplit
) -> list[dict[str, object]]:
    evidence = []
    for fold in folds:
        groups = design.groups[fold.train]
        if np.unique(groups).size < inner.n_splits:
            raise ValueError(f"Fold {fold.index} has fewer training groups than inner_splits.")
        labels = design.y[fold.train].astype(np.intp) if inner.stratified else None
        splitter = inner_cv(
            groups,
            inner,
            y_train=labels,
            random_state=recipe.validation.seed + max(fold.index - 1, 0),
        )
        splits = []
        for train, valid in splitter.split(
            design.X[fold.train], design.y[fold.train], groups=groups
        ):
            train_rows, valid_rows = fold.train[train], fold.train[valid]
            if recipe.analysis.task == "classification" and (
                np.unique(design.y[train_rows]).size != 2
                or np.unique(design.y[valid_rows]).size != 2
            ):
                raise ValueError(
                    f"Fold {fold.index}: every inner training/validation split "
                    "must contain both classes."
                )
            splits.append(
                {
                    "train_rows": train_rows,
                    "validation_rows": valid_rows,
                    "train_groups": np.unique(design.groups[train_rows]),
                    "test_groups": np.unique(design.groups[valid_rows]),
                }
            )
        evidence.append(
            {
                "fold": fold.index,
                "train_rows": fold.train,
                "test_rows": fold.test,
                "train_groups": np.unique(design.groups[fold.train]),
                "test_groups": np.unique(design.groups[fold.test]),
                "inner_splits": splits,
            }
        )
    return evidence


def _input_hashes(recipe: ModelRecipe) -> dict[str, str]:
    paths = [recipe.path]
    for path in recipe.inputs.paths:
        paths.extend((path, path.with_suffix(".json"), path.with_name(path.stem + "_coverage.tsv")))
    if recipe.inputs.targets is not None:
        paths.append(recipe.inputs.targets)
    return {str(path): file_hash(path) for path in paths}


def _provenance(recipe: ModelRecipe) -> dict[str, object]:
    hashes = _input_hashes(recipe)
    if hashes[str(recipe.path)] != recipe.recipe_hash:
        raise ValueError("model recipe changed after loading; reload it before execution.")
    return {
        "inputs": hashes,
        "software": software_versions(),
        "implementation_hash": implementation_hash(),
        "recipe_hash": recipe.recipe_hash,
    }


def _verify_provenance(recipe: ModelRecipe, prepared: _Prepared) -> None:
    if _provenance(recipe) != prepared.provenance:
        raise ValueError("model provenance changed during execution; results were not published.")


def _prepare(recipe: ModelRecipe) -> _Prepared:
    provenance = _provenance(recipe)
    design, identities = _load_design(recipe)
    inner = InnerSplit(
        "subject",
        stratified=recipe.analysis.task == "classification",
        n_splits=recipe.validation.inner_splits,
    )
    folds = _folds(recipe, design)
    table = FeatureTable(
        values=design.X[:, design.feature_columns],
        coverage=design.coverage,
        meta=design.meta,
        flags=design.flags,
        row_ids=design.row_ids if isinstance(design, Design) else None,
        row_labels=(
            tuple(group for _, group in design.sample_ids)
            if isinstance(design, GroupDesign)
            else None
        ),
    )
    return _Prepared(
        design,
        identities,
        table,
        _pipeline(recipe, design),
        folds,
        inner,
        _fold_evidence(recipe, design, folds, inner),
        provenance,
    )


def check_model(recipe: ModelRecipe) -> ModelCheck:
    """Validate inputs, design, estimator settings and every nested split; write nothing."""
    prepared = _prepare(recipe)
    _verify_provenance(recipe, prepared)
    design = prepared.design
    return ModelCheck(
        len(design.y),
        len(design.meta),
        np.unique(design.groups).size,
        len(prepared.folds),
        recipe.output,
    )


def _predictions(
    recipe: ModelRecipe, prepared: _Prepared
) -> tuple[pd.DataFrame, tuple[FoldPrediction, ...] | tuple[FoldClassification, ...]]:
    design = prepared.design
    arguments: dict[str, Any] = {
        "inner": prepared.inner,
        "seed": recipe.validation.seed,
        "scoring": recipe.validation.scoring,
    }
    results: tuple[FoldPrediction, ...] | tuple[FoldClassification, ...]
    if recipe.analysis.task == "classification":
        results = cross_fit_classification(
            prepared.folds,
            design.X,
            design.y.astype(np.intp),
            design.groups,
            prepared.pipeline,
            recipe.model.grid,
            **arguments,
        )
    else:
        results = cross_fit_regression(
            prepared.folds,
            design.X,
            design.y,
            design.groups,
            prepared.pipeline,
            recipe.model.grid,
            **arguments,
        )
    frame = prepared.identities.copy()
    frame.insert(0, "row", np.arange(len(frame)))
    frame["model_group"] = design.groups
    frame["y_true"] = design.y
    frame["y_pred"] = np.nan
    frame["fold"] = -1
    for result in results:
        frame.loc[result.rows, "y_pred"] = result.y_pred
        frame.loc[result.rows, "fold"] = result.fold
        if isinstance(result, FoldClassification) and result.y_prob is not None:
            for index, label in enumerate(result.classes):
                frame.loc[result.rows, f"probability_{label}"] = result.y_prob[:, index]
    if (frame.fold < 0).any() or not np.isfinite(frame.y_pred).all():
        raise ValueError("every model sample must have exactly one held-out prediction.")
    return frame, results


def _classification_summary(result: Any) -> dict[str, object]:
    names = (
        "accuracy",
        "balanced_accuracy",
        "auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "specificity",
        "confusion",
    )
    return {name: getattr(result, name) for name in names}


def _metrics(recipe: ModelRecipe, frame: pd.DataFrame) -> dict[str, object]:
    groups = frame.model_group.to_numpy(dtype=object)
    truth, prediction = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    if recipe.analysis.task == "regression":
        overall, subjects = regression_metrics(
            truth, prediction, groups, folds=frame.fold.to_numpy()
        )
        overall.update(
            mean_squared_error=float(mean_squared_error(truth, prediction)),
            mean_absolute_error=float(mean_absolute_error(truth, prediction)),
        )
        return {
            "overall": overall,
            "per_group": subjects,
            "aggregation": "pooled_errors_and_Fisher_z_group_correlation",
        }
    probabilities = (
        frame[["probability_0", "probability_1"]].to_numpy() if "probability_0" in frame else None
    )
    pooled = classification_metrics(
        truth.astype(np.intp), prediction.astype(np.intp), y_prob=probabilities
    )
    grouped = classification_metrics(
        truth.astype(np.intp), prediction.astype(np.intp), y_prob=probabilities, groups=groups
    )
    return {
        "overall": _classification_summary(pooled),
        "group_mean": _classification_summary(grouped),
        "per_group": grouped.per_subject,
        "aggregation": "pooled_and_equal_group_mean",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n")


def _model_descriptors(recipe: ModelRecipe, prepared: _Prepared) -> pd.DataFrame:
    descriptors = prepared.identities.copy()
    for name, values in (
        (recipe.analysis.groups, prepared.design.groups),
        (recipe.analysis.target, prepared.design.y),
    ):
        if name not in descriptors:
            descriptors[name] = values
    row_key = "epoch" if recipe.inputs.rows == "epochs" else "group"
    return descriptors.drop(columns=row_key)


def _write_bundle(
    directory: Path,
    recipe: ModelRecipe,
    prepared: _Prepared,
    frame: pd.DataFrame,
    results: Sequence[FoldPrediction | FoldClassification],
) -> None:
    frame.to_csv(directory / "predictions.tsv", sep="\t", index=False, na_rep="n/a")
    prepared.design.quality_ledger.to_csv(directory / "quality_ledger.tsv", sep="\t", index=False)
    feature_quality(prepared.table).to_csv(directory / "feature_quality.tsv", sep="\t", index=False)
    write_table(
        prepared.table,
        directory / "design_features.tsv",
        rows=_model_descriptors(recipe, prepared),
    )
    np.savez_compressed(
        directory / "design_matrix.npz",
        X=prepared.design.X,
        y=prepared.design.y,
        feature_columns=prepared.design.feature_columns,
        covariate_columns=prepared.design.covariate_columns,
    )
    evidence = [
        dict(record, best_params=result.best_params)
        for record, result in zip(prepared.evidence, results, strict=True)
    ]
    _write_json(directory / "folds.json", evidence)
    _write_json(directory / "metrics.json", _metrics(recipe, frame))
    _write_json(
        directory / "design.json",
        {"column_names": prepared.design.column_names, "row_semantics": recipe.inputs.rows},
    )
    resolved = serializable(recipe)
    (directory / "resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    manifest = {
        "schema_version": 1,
        "task": recipe.analysis.task,
        "rows": recipe.inputs.rows,
        "n_rows": len(frame),
        "n_features": len(prepared.table.meta),
        "provenance": prepared.provenance,
        "files": {
            path.name: file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()
        },
    }
    _write_json(directory / "manifest.json", manifest)


def run_model(recipe: ModelRecipe) -> ModelResult:
    """Fit complete nested grouped CV and atomically publish a new result directory."""
    recipe.output.parent.mkdir(parents=True, exist_ok=True)
    lock = recipe.output.parent / f".{recipe.output.name}.lock"
    with FileLock(lock):
        if recipe.output.exists():
            raise FileExistsError(f"model output already exists: {recipe.output}")
        prepared = _prepare(recipe)
        frame, results = _predictions(recipe, prepared)
        with TemporaryDirectory(
            prefix=f".{recipe.output.name}-", dir=recipe.output.parent
        ) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            _write_bundle(bundle, recipe, prepared, frame, results)
            _verify_provenance(recipe, prepared)
            if recipe.output.exists():
                raise FileExistsError(f"model output already exists: {recipe.output}")
            os.rename(bundle, recipe.output)
    return ModelResult(recipe.output, len(frame), len(prepared.table.meta), len(prepared.folds))
