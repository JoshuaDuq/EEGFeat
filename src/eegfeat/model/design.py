from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from eegfeat.model import _deps as _deps
from eegfeat.table import FeatureTable, RowId

__all__ = [
    "Design",
    "Selection",
    "build_design",
    "compute_train_group_intersection_mask",
    "harmonize_fold",
    "select",
]


@dataclass(frozen=True)
class Selection:
    measure: tuple[str, ...] = ()
    band: tuple[str, ...] = ()
    space_kind: tuple[str, ...] = ()
    window: tuple[str, ...] = ()
    normalization: tuple[str, ...] = ()
    space: tuple[str, ...] = ()


@dataclass(frozen=True)
class Design:
    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    groups: npt.NDArray[np.object_]
    runs: npt.NDArray[np.object_] | None
    row_ids: tuple[RowId, ...]
    column_names: tuple[str, ...]
    feature_columns: npt.NDArray[np.intp]
    covariate_columns: npt.NDArray[np.intp]

    @property
    def n_covariates(self) -> int:
        return int(self.covariate_columns.size)


_DEFAULT_SELECTION = Selection()


def select(table: FeatureTable, selection: Selection) -> FeatureTable:
    kept_indices: list[int] = []
    for i, meta in enumerate(table.meta):
        if selection.measure and meta.measure not in selection.measure:
            continue
        if selection.band and (meta.band is None or meta.band.name not in selection.band):
            continue
        if selection.space_kind and meta.space_kind not in selection.space_kind:
            continue
        if selection.window and (meta.window is None or meta.window not in selection.window):
            continue
        if selection.normalization and meta.normalization not in selection.normalization:
            continue
        if selection.space and meta.space not in selection.space:
            continue
        kept_indices.append(i)

    if not kept_indices:
        msg = f"Selection {selection} matched no columns in FeatureTable."
        raise ValueError(msg)

    if len(kept_indices) == len(table.meta):
        return table

    idx = np.array(kept_indices, dtype=np.intp)
    return FeatureTable(
        values=table.values[:, idx],
        coverage=table.coverage[:, idx],
        meta=tuple(table.meta[i] for i in kept_indices),
        flags={k: v[:, idx] for k, v in table.flags.items()},
        row_labels=table.row_labels,
        row_ids=table.row_ids,
    )


def build_design(
    table: FeatureTable,
    targets: pd.DataFrame,
    *,
    target: str,
    groups: str = "subject_id",
    runs: str | None = None,
    covariates: Sequence[str] = (),
    selection: Selection = _DEFAULT_SELECTION,
    strict_covariates: bool = True,
) -> Design:
    if table.row_ids is None:
        msg = (
            "Modeling is per-epoch only; cross-trial/group-row FeatureTables "
            "have no row_ids and cannot be aligned to targets."
        )
        raise ValueError(msg)
    row_ids = table.row_ids

    if selection != _DEFAULT_SELECTION:
        table = select(table, selection)

    key_columns = ["recording", "epoch", "event"]
    for col in key_columns:
        if col not in targets.columns:
            msg = f"targets DataFrame missing required key column '{col}'."
            raise ValueError(msg)

    if target not in targets.columns:
        msg = f"Target column '{target}' missing from targets."
        raise ValueError(msg)

    if groups not in targets.columns:
        msg = f"Groups column '{groups}' missing from targets."
        raise ValueError(msg)

    if runs is not None and runs not in targets.columns:
        msg = f"Runs column '{runs}' missing from targets."
        raise ValueError(msg)

    target_lower = target.strip().lower()
    leaking = [
        c
        for c in covariates
        if str(c).strip().lower() in {target_lower, "outcome", "target"}
    ]
    if leaking:
        msg = (
            "Covariates include the selected target, which would leak labels into predictors: "
            f"{leaking}. Target={target!r}."
        )
        raise ValueError(msg)

    missing_covs = [c for c in covariates if c not in targets.columns]
    if missing_covs:
        if strict_covariates:
            msg = f"Requested covariates missing from targets: {missing_covs}."
            raise ValueError(msg)
        active_covs = [c for c in covariates if c in targets.columns]
    else:
        active_covs = list(covariates)

    table_keys = list(row_ids)
    if len(set(table_keys)) != len(table_keys):
        msg = "FeatureTable contains duplicate row_ids; join must be one-to-one."
        raise ValueError(msg)

    target_keys = list(zip(targets["recording"], targets["epoch"], targets["event"], strict=True))
    if len(set(target_keys)) != len(target_keys):
        msg = "targets contains duplicate (recording, epoch, event) keys; join must be one-to-one."
        raise ValueError(msg)

    if set(table_keys) != set(target_keys):
        msg = "FeatureTable row_ids and targets keys do not match one-to-one."
        raise ValueError(msg)

    target_key_map = {k: i for i, k in enumerate(target_keys)}
    target_row_indices = [target_key_map[k] for k in table_keys]
    aligned_targets = targets.iloc[target_row_indices]

    y = pd.to_numeric(aligned_targets[target], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y)):
        msg = f"Target column '{target}' contains non-finite values."
        raise ValueError(msg)

    groups_arr = cast(npt.NDArray[np.object_], aligned_targets[groups].to_numpy(dtype=object))
    runs_arr = (
        cast(npt.NDArray[np.object_], aligned_targets[runs].to_numpy(dtype=object))
        if runs is not None
        else None
    )

    n_features = table.values.shape[1]
    feature_names = tuple(m.name for m in table.meta)

    if active_covs:
        cov_df = aligned_targets[active_covs].apply(pd.to_numeric, errors="coerce")
        cov_arr = cov_df.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(cov_arr)):
            msg = "Covariates contain non-finite values."
            raise ValueError(msg)
        X = np.column_stack([table.values, cov_arr])
        column_names = feature_names + tuple(active_covs)
        feature_columns = np.arange(n_features, dtype=np.intp)
        covariate_columns = np.arange(n_features, n_features + len(active_covs), dtype=np.intp)
    else:
        X = table.values
        column_names = feature_names
        feature_columns = np.arange(n_features, dtype=np.intp)
        covariate_columns = np.empty(0, dtype=np.intp)

    return Design(
        X=X,
        y=y,
        groups=groups_arr,
        runs=runs_arr,
        row_ids=row_ids,
        column_names=column_names,
        feature_columns=feature_columns,
        covariate_columns=covariate_columns,
    )


def compute_train_group_intersection_mask(
    X_train: npt.NDArray[np.float64],
    groups_train: npt.NDArray[np.object_] | Sequence[object],
) -> npt.NDArray[np.bool_]:
    X_arr = np.asarray(X_train, dtype=np.float64)
    groups_arr = np.asarray(groups_train)
    if X_arr.ndim != 2:
        msg = f"Expected 2D X_train, got shape={X_arr.shape}"
        raise ValueError(msg)
    if X_arr.shape[0] != len(groups_arr):
        msg = f"Length mismatch for X_train/groups_train: {X_arr.shape[0]} vs {len(groups_arr)}"
        raise ValueError(msg)
    n_features = X_arr.shape[1]
    if n_features == 0:
        return np.zeros(0, dtype=np.bool_)

    keep_mask = np.ones(n_features, dtype=np.bool_)
    unique_groups = np.unique(groups_arr)
    for grp in unique_groups:
        grp_mask = groups_arr == grp
        if not np.any(grp_mask):
            continue
        grp_has = np.any(np.isfinite(X_arr[grp_mask]), axis=0)
        keep_mask &= grp_has

    if not np.any(keep_mask):
        msg = "No features are finite for every training group."
        raise ValueError(msg)
    return keep_mask


def harmonize_fold(
    X_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    groups_train: npt.NDArray[np.object_] | Sequence[object],
    *,
    mode: str | None,
    n_covariates: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    mode_str = (mode or "union_impute").strip().lower()
    if mode_str not in ("intersection", "union_impute"):
        msg = f"Unknown harmonization mode: {mode!r}. Expected 'intersection' or 'union_impute'."
        raise ValueError(msg)
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    if Xtr.shape[1] != Xte.shape[1]:
        msg = f"X_train/X_test feature mismatch: {Xtr.shape[1]} vs {Xte.shape[1]}"
        raise ValueError(msg)

    if mode_str == "union_impute":
        keep = np.ones(Xtr.shape[1], dtype=np.bool_)
        return Xtr, Xte, keep

    Xtr_eeg = Xtr[:, :-n_covariates] if n_covariates > 0 else Xtr

    if Xtr_eeg.shape[1] > 0:
        keep_eeg = compute_train_group_intersection_mask(Xtr_eeg, np.asarray(groups_train))
    else:
        keep_eeg = np.zeros(0, dtype=np.bool_)

    keep = np.ones(Xtr.shape[1], dtype=np.bool_)
    if n_covariates > 0:
        keep[:-n_covariates] = keep_eeg
    else:
        keep = keep_eeg

    if keep.size == 0 or not np.any(keep):
        msg = "Fold-specific intersection harmonization removed all features."
        raise ValueError(msg)
    return Xtr[:, keep], Xte[:, keep], keep
