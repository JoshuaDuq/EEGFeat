from __future__ import annotations

from typing import Any

import sklearn
from packaging.version import parse as parse_version
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eegfeat.model import _deps as _deps
from eegfeat.model.transformers import PreprocessingConfig, base_preprocessing_steps

__all__ = [
    "elasticnet_grid",
    "elasticnet_pipeline",
    "ensemble_pipeline",
    "logistic_grid",
    "logistic_pipeline",
    "random_forest_classifier_grid",
    "random_forest_classifier_pipeline",
    "random_forest_grid",
    "random_forest_pipeline",
    "ridge_grid",
    "ridge_pipeline",
    "svm_grid",
    "svm_pipeline",
]


def _append_classification_resampler(
    steps: list[tuple[str, object]],
    *,
    resampler: str = "none",
    resampler_seed: int = 42,
) -> None:
    resampler_choice = resampler.strip().lower()
    if resampler_choice == "none":
        return
    if resampler_choice == "undersample":
        try:
            from imblearn.under_sampling import (  # type: ignore
                RandomUnderSampler,
            )
        except ImportError as err:
            msg = "imblearn is required for resampling 'undersample'."
            raise ImportError(msg) from err
        steps.append(("resampler", RandomUnderSampler(random_state=resampler_seed)))
    elif resampler_choice == "smote":
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
        except ImportError as err:
            msg = "imblearn is required for resampling 'smote'."
            raise ImportError(msg) from err
        steps.append(("resampler", SMOTE(random_state=resampler_seed)))
    else:
        msg = f"Unknown resampler '{resampler}'."
        raise ValueError(msg)


_LR_PENALTIES = ("l1", "l2", "elasticnet", "none")


def _get_lr_kwargs(penalty: str, l1_ratio: float | None = None) -> dict[str, Any]:
    # Handles penalty argument deprecation in scikit-learn >= 1.8.0. On those versions the
    # penalty is translated into l1_ratio/C rather than passed through, so an unrecognised
    # value would silently fall past every branch and fit a default L2 model instead of the
    # error older scikit-learn raised.
    if penalty not in _LR_PENALTIES:
        raise ValueError(f"penalty must be one of {_LR_PENALTIES}, got {penalty!r}.")

    kwargs: dict[str, Any] = {}
    if parse_version(sklearn.__version__) >= parse_version("1.8.0"):
        if penalty == "l2":
            kwargs["l1_ratio"] = 0.0
        elif penalty == "l1":
            kwargs["l1_ratio"] = 1.0
        elif penalty == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5
        elif penalty == "none":
            kwargs["C"] = float("inf")
    else:
        kwargs["penalty"] = penalty
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5
    return kwargs


def _assemble_pipeline(steps: list[tuple[str, object]], resampler: str) -> Pipeline:
    if resampler.strip().lower() != "none":
        try:
            from imblearn.pipeline import (  # type: ignore
                Pipeline as ImbPipeline,
            )

            return ImbPipeline(steps)
        except ImportError as err:
            msg = "imblearn is required for resampling pipeline."
            raise ImportError(msg) from err
    return Pipeline(steps)


def elasticnet_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    max_iter: int = 10000,
    tol: float = 1e-4,
    selection: str = "cyclic",
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_regression,
    )
    steps.append(
        (
            "regressor",
            ElasticNet(
                random_state=seed,
                max_iter=max_iter,
                tol=tol,
                selection=selection,
            ),
        )
    )
    return Pipeline(steps)


def ridge_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_regression,
    )
    steps.append(("regressor", Ridge(random_state=seed)))
    return Pipeline(steps)


def random_forest_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    n_estimators: int = 500,
    n_jobs: int = 1,
    bootstrap: bool = True,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=False,
        n_covariates=n_covariates,
        score_func=f_regression,
    )
    steps.append(
        (
            "rf",
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=seed,
                bootstrap=bootstrap,
            ),
        )
    )
    return Pipeline(steps)


def svm_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    kernel: str = "rbf",
    n_covariates: int = 0,
    class_weight: str | dict[Any, Any] | None = "balanced",
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    steps.append(
        (
            "svm",
            SVC(
                kernel=kernel,
                probability=True,
                random_state=seed,
                class_weight=class_weight,
            ),
        )
    )
    return _assemble_pipeline(steps, resampler)


def logistic_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    penalty: str = "l2",
    n_covariates: int = 0,
    max_iter: int = 1000,
    class_weight: str | dict[Any, Any] | None = "balanced",
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    lr_kwargs = _get_lr_kwargs(penalty=penalty, l1_ratio=0.5 if penalty == "elasticnet" else None)
    steps.append(
        (
            "lr",
            LogisticRegression(
                solver=solver,
                max_iter=max_iter,
                random_state=seed,
                class_weight=class_weight,
                **lr_kwargs,
            ),
        )
    )
    return _assemble_pipeline(steps, resampler)


def random_forest_classifier_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    n_estimators: int = 500,
    n_jobs: int = 1,
    class_weight: str | dict[Any, Any] | None = "balanced",
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    steps = base_preprocessing_steps(
        config,
        include_scaling=False,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                class_weight=class_weight,
                n_jobs=n_jobs,
            ),
        )
    )
    return _assemble_pipeline(steps, resampler)


def ensemble_pipeline(
    config: PreprocessingConfig,
    *,
    seed: int,
    n_covariates: int = 0,
    svm_kernel: str = "rbf",
    lr_penalty: str = "l2",
    rf_n_estimators: int = 500,
    calibrate_ensemble: bool = False,
    resampler: str = "none",
    resampler_seed: int = 42,
) -> Pipeline:
    svm = SVC(
        kernel=svm_kernel,
        probability=True,
        random_state=seed,
        class_weight="balanced",
    )
    solver = "saga" if lr_penalty in ("l1", "elasticnet") else "lbfgs"
    lr_kwargs = _get_lr_kwargs(
        penalty=lr_penalty,
        l1_ratio=0.5 if lr_penalty == "elasticnet" else None,
    )
    lr = LogisticRegression(
        solver=solver,
        max_iter=1000,
        random_state=seed,
        class_weight="balanced",
        **lr_kwargs,
    )
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        random_state=seed,
        class_weight="balanced",
        n_jobs=1,
    )

    if calibrate_ensemble:
        from sklearn.calibration import CalibratedClassifierCV

        estimators: list[tuple[str, object]] = [
            ("svm", CalibratedClassifierCV(svm, method="sigmoid", cv=2)),
            ("lr", lr),
            ("rf", CalibratedClassifierCV(rf, method="sigmoid", cv=2)),
        ]
    else:
        estimators = [("svm", svm), ("lr", lr), ("rf", rf)]

    steps = base_preprocessing_steps(
        config,
        include_scaling=True,
        n_covariates=n_covariates,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, resampler=resampler, resampler_seed=resampler_seed)
    steps.append(("ensemble", VotingClassifier(estimators=estimators, voting="soft")))
    return _assemble_pipeline(steps, resampler)


# The grids do not tune the variance threshold. It acts on unscaled features, whose units
# differ across a FeatureTable, so any value above zero selects features by unit; the
# pipelines keep it at 0.0, which removes only constant features.


def elasticnet_grid() -> dict[str, list[object]]:
    return {
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "regressor__l1_ratio": [0.2, 0.5, 0.8],
    }


def ridge_grid() -> dict[str, list[object]]:
    return {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}


def random_forest_grid() -> dict[str, list[object]]:
    return {
        "rf__max_depth": [5, 10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }


def svm_grid() -> dict[str, list[object]]:
    return {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", "auto"],
    }


def logistic_grid(*, penalty: str = "l2") -> dict[str, list[object]]:
    grid: dict[str, list[object]] = {"lr__C": [0.01, 0.1, 1.0, 10.0]}
    if penalty == "elasticnet":
        grid["lr__l1_ratio"] = [0.1, 0.5, 0.9]
    return grid


def random_forest_classifier_grid() -> dict[str, list[object]]:
    return random_forest_grid()
