from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from eegfeat.model.aggregate import AggregationConfig, subject_level_r
from eegfeat.model.scoring import safe_pearsonr

__all__ = [
    "ClassificationResult",
    "classification_metrics",
    "regression_metrics",
    "within_condition_metrics",
    "within_subject_centered_metrics",
]

_DEFAULT_AGGREGATION_CONFIG = AggregationConfig()


@dataclass(frozen=True)
class ClassificationResult:
    y_true: npt.NDArray[np.intp]
    y_pred: npt.NDArray[np.intp]
    y_prob: npt.NDArray[np.float64] | None
    groups: npt.NDArray[np.object_] | None
    accuracy: float
    balanced_accuracy: float
    auc: float
    average_precision: float
    f1: float
    precision: float
    recall: float
    specificity: float
    confusion: npt.NDArray[np.intp]
    per_subject: Mapping[str, Mapping[str, float]]
    mean_subject_auc: float


def _subset_classification_metrics(
    y_true: npt.NDArray[np.intp],
    y_pred: npt.NDArray[np.intp],
    y_prob: npt.NDArray[np.float64] | None = None,
) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else np.nan
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else np.nan
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else np.nan
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else np.nan
    b_acc = (
        float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) >= 2 else np.nan
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _fn, _tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    auc = np.nan
    ap = np.nan
    if y_prob is not None and len(np.unique(y_true)) == 2:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] >= 2:
            y_prob_1d = y_prob_arr[:, 1]
        elif y_prob_arr.ndim == 1:
            y_prob_1d = y_prob_arr
        else:
            y_prob_1d = y_prob_arr.ravel()
        prob_mask = np.isfinite(y_prob_1d) & np.isfinite(y_true)
        if np.sum(prob_mask) >= 2 and len(np.unique(y_true[prob_mask])) == 2:
            with contextlib.suppress(ValueError, TypeError):
                auc = float(roc_auc_score(y_true[prob_mask], y_prob_1d[prob_mask]))
                ap = float(average_precision_score(y_true[prob_mask], y_prob_1d[prob_mask]))

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "auc": auc,
        "average_precision": ap,
    }


def classification_metrics(
    y_true: npt.NDArray[np.intp],
    y_pred: npt.NDArray[np.intp],
    *,
    y_prob: npt.NDArray[np.float64] | None = None,
    groups: npt.NDArray[np.object_] | None = None,
) -> ClassificationResult:
    # The confusion matrix, specificity and the positive class of precision and recall all
    # assume 0/1 coding, so any other coding is refused instead of scored against it.
    labels = np.unique(np.concatenate([np.ravel(y_true), np.ravel(y_pred)]))
    if not set(labels.tolist()) <= {0, 1}:
        msg = f"classification_metrics expects labels coded 0/1, got {labels.tolist()}."
        raise ValueError(msg)
    y_t = np.asarray(y_true, dtype=np.intp)
    y_p = np.asarray(y_pred, dtype=np.intp)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1]).astype(np.intp)

    if groups is not None:
        groups_arr = np.asarray(groups)
        per_subject: dict[str, Mapping[str, float]] = {}
        for subj in np.unique(groups_arr):
            mask = groups_arr == subj
            sub_prob = y_prob[mask] if y_prob is not None else None
            per_subject[str(subj)] = _subset_classification_metrics(y_t[mask], y_p[mask], sub_prob)

        def _mean_metric(key: str) -> float:
            vals = [m[key] for m in per_subject.values() if key in m and np.isfinite(m[key])]
            return float(np.mean(vals)) if vals else np.nan

        mean_auc = _mean_metric("auc")
        return ClassificationResult(
            y_true=y_t,
            y_pred=y_p,
            y_prob=y_prob,
            groups=groups_arr,
            accuracy=_mean_metric("accuracy"),
            balanced_accuracy=_mean_metric("balanced_accuracy"),
            auc=mean_auc,
            average_precision=_mean_metric("average_precision"),
            f1=_mean_metric("f1"),
            precision=_mean_metric("precision"),
            recall=_mean_metric("recall"),
            specificity=_mean_metric("specificity"),
            confusion=cm,
            per_subject=per_subject,
            mean_subject_auc=mean_auc,
        )

    global_m = _subset_classification_metrics(y_t, y_p, y_prob)
    return ClassificationResult(
        y_true=y_t,
        y_pred=y_p,
        y_prob=y_prob,
        groups=None,
        accuracy=global_m["accuracy"],
        balanced_accuracy=global_m["balanced_accuracy"],
        auc=global_m["auc"],
        average_precision=global_m["average_precision"],
        f1=global_m["f1"],
        precision=global_m["precision"],
        recall=global_m["recall"],
        specificity=global_m["specificity"],
        confusion=cm,
        per_subject={},
        mean_subject_auc=np.nan,
    )


def regression_metrics(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_] | None = None,
    *,
    config: AggregationConfig = _DEFAULT_AGGREGATION_CONFIG,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(yt) & np.isfinite(yp)

    if int(finite.sum()) < 2:
        return {
            "pearson_r": np.nan,
            "subject_level_r": np.nan,
            "avg_subject_r_fisher_z": np.nan,
            "r2": np.nan,
            "explained_variance": np.nan,
            "n": float(finite.sum()),
        }, []

    yt_f, yp_f = yt[finite], yp[finite]
    r_val, _ = safe_pearsonr(yt_f, yp_f)
    r2_val = float(r2_score(yt_f, yp_f))
    ev_val = float(explained_variance_score(yt_f, yp_f))

    summary: dict[str, float] = {
        "pearson_r": r_val,
        "r2": r2_val,
        "explained_variance": ev_val,
        "n": float(finite.sum()),
        "subject_level_r": np.nan,
        "avg_subject_r_fisher_z": np.nan,
    }
    per_subject_list: list[dict[str, object]] = []

    if groups is not None:
        groups_arr = np.asarray(groups)[finite]
        pred_df = pd.DataFrame({"subject_id": groups_arr, "y_true": yt_f, "y_pred": yp_f})
        subj_r = subject_level_r(pred_df, config=config)
        summary["subject_level_r"] = subj_r.r
        summary["avg_subject_r_fisher_z"] = subj_r.r
        for s, r in subj_r.per_subject:
            per_subject_list.append({"subject": s, "r": r})

    return summary, per_subject_list


def within_subject_centered_metrics(
    target: npt.NDArray[np.float64],
    full_prediction: npt.NDArray[np.float64],
    nuisance_prediction: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
) -> dict[str, float]:
    grp = np.asarray(groups)
    t = np.asarray(target, dtype=float)
    f = np.asarray(full_prediction, dtype=float)
    n = np.asarray(nuisance_prediction, dtype=float)
    if not (grp.shape == t.shape == f.shape == n.shape):
        raise ValueError("Within-subject prediction metrics require aligned 1D arrays.")

    full_scores: list[float] = []
    nuis_scores: list[float] = []

    for subj in pd.unique(grp):
        if pd.isna(subj):
            continue
        mask = grp == subj
        t_sub = t[mask]
        f_sub = f[mask]
        n_sub = n[mask]
        if len(t_sub) < 2:
            continue
        cent_t = t_sub - np.mean(t_sub)
        denom = float(cent_t @ cent_t)
        if denom <= 1e-12:
            continue
        cent_f = f_sub - np.mean(f_sub)
        cent_n = n_sub - np.mean(n_sub)
        res_f = cent_t - cent_f
        res_n = cent_t - cent_n
        full_scores.append(1.0 - float(res_f @ res_f) / denom)
        nuis_scores.append(1.0 - float(res_n @ res_n) / denom)

    if not full_scores:
        return {
            "within_subject_centered_full_r2": float("nan"),
            "within_subject_centered_nuisance_r2": float("nan"),
            "within_subject_centered_delta_r2": float("nan"),
        }

    full_r2 = float(np.mean(full_scores))
    nuis_r2 = float(np.mean(nuis_scores))
    return {
        "within_subject_centered_full_r2": full_r2,
        "within_subject_centered_nuisance_r2": nuis_r2,
        "within_subject_centered_delta_r2": full_r2 - nuis_r2,
    }


def _within_condition_cells(
    subject_mask: npt.NDArray[np.bool_],
    conditions: npt.NDArray[np.object_],
) -> list[npt.NDArray[np.intp]]:
    subject_rows = np.flatnonzero(subject_mask).astype(np.intp)
    cells: list[npt.NDArray[np.intp]] = []
    sub_conditions = conditions[subject_rows]
    for condition in np.unique(sub_conditions):
        cell_rows = subject_rows[sub_conditions == condition]
        if cell_rows.size >= 2:
            cells.append(cell_rows)
    return cells


def _center_within_cells(
    values: npt.NDArray[np.float64],
    cells: list[npt.NDArray[np.intp]],
) -> npt.NDArray[np.float64]:
    return np.concatenate([values[cell] - values[cell].mean() for cell in cells])


def within_condition_metrics(
    target: npt.NDArray[np.float64],
    full_prediction: npt.NDArray[np.float64],
    nuisance_prediction: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    conditions: npt.NDArray[np.object_],
) -> dict[str, float]:
    grp = np.asarray(groups)
    t = np.asarray(target, dtype=float)
    f = np.asarray(full_prediction, dtype=float)
    n = np.asarray(nuisance_prediction, dtype=float)
    cond = np.asarray(conditions)

    if not (grp.shape == t.shape == f.shape == n.shape == cond.shape):
        raise ValueError("Within-condition prediction metrics require aligned 1D arrays.")

    full_scores: list[float] = []
    nuis_scores: list[float] = []
    n_trials = 0

    for subj in np.unique(grp):
        mask = grp == subj
        cells = _within_condition_cells(mask, cond)
        if not cells:
            continue
        centered_t = _center_within_cells(t, cells)
        denominator = float(centered_t @ centered_t)
        if denominator <= 1e-12:
            continue
        centered_f = _center_within_cells(f, cells)
        centered_n = _center_within_cells(n, cells)
        full_res = centered_t - centered_f
        nuis_res = centered_t - centered_n
        full_scores.append(1.0 - float(full_res @ full_res) / denominator)
        nuis_scores.append(1.0 - float(nuis_res @ nuis_res) / denominator)
        n_trials += int(sum(len(cell) for cell in cells))

    if not full_scores:
        return {
            "within_condition_centered_full_r2": float("nan"),
            "within_condition_centered_nuisance_r2": float("nan"),
            "within_condition_centered_delta_r2": float("nan"),
            "within_condition_centered_n_subjects": 0.0,
            "within_condition_centered_n_trials": 0.0,
        }

    full_r2 = float(np.mean(full_scores))
    nuis_r2 = float(np.mean(nuis_scores))
    return {
        "within_condition_centered_full_r2": full_r2,
        "within_condition_centered_nuisance_r2": nuis_r2,
        "within_condition_centered_delta_r2": full_r2 - nuis_r2,
        "within_condition_centered_n_subjects": float(len(full_scores)),
        "within_condition_centered_n_trials": float(n_trials),
    }
