from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

__all__ = [
    "Deconfounder",
    "DropAllNaNColumns",
    "MissingnessThreshold",
    "PreprocessingConfig",
    "ReplaceInfWithNaN",
    "SpatialFeatureSelector",
    "VarianceThreshold",
    "base_preprocessing_steps",
    "transform_feature_names",
    "transform_feature_names_through_steps",
    "validate_subject_missingness",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    max_feature_missingness: float = 0.2
    max_subject_missingness: float = 0.5
    feature_selection_percentile: float | None = None
    deconfound: bool = False
    spatial_regions_allowed: tuple[str, ...] = ()
    pca_enabled: bool = False
    pca_n_components: int | float | None = None
    pca_whiten: bool = False
    pca_svd_solver: str = "auto"
    pca_random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_feature_missingness <= 1.0:
            raise ValueError(
                f"max_feature_missingness must be in [0, 1], got {self.max_feature_missingness}"
            )
        if not 0.0 <= self.max_subject_missingness <= 1.0:
            raise ValueError(
                f"max_subject_missingness must be in [0, 1], got {self.max_subject_missingness}"
            )


def validate_subject_missingness(
    values: npt.NDArray[np.float64],
    groups: npt.NDArray[np.object_],
    *,
    maximum: float,
) -> None:
    x_arr = np.asarray(values, dtype=float)
    groups_arr = np.asarray(groups)

    if x_arr.shape[1] == 0:
        raise ValueError("validate_subject_missingness requires at least one retained feature.")

    for grp in np.unique(groups_arr):
        mask = groups_arr == grp
        sub_x = x_arr[mask]
        sub_missingness = float(np.isnan(sub_x).mean())
        if sub_missingness > maximum:
            raise ValueError(
                f"Subject {grp} has missingness {sub_missingness:.2f} > maximum {maximum:.2f}."
            )


class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def fit(self, X: Any, y: Any = None) -> ReplaceInfWithNaN:
        _ = (X, y)
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        out = np.asarray(X, dtype=float).copy()
        out[np.isinf(out)] = np.nan
        return out


class DropAllNaNColumns(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, min_finite: int = 1) -> None:
        self.min_finite = min_finite

    def fit(self, X: Any, y: Any = None) -> DropAllNaNColumns:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        finite_counts = np.sum(np.isfinite(x_arr), axis=0)
        self.support_mask_ = finite_counts >= self.min_finite
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class VarianceThreshold(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> VarianceThreshold:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        self.variances_ = np.nanvar(x_arr, axis=0)
        self.support_mask_ = self.variances_ > self.threshold
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class MissingnessThreshold(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(
        self,
        max_feature_missingness: float = 0.2,
        max_subject_missingness: float = 0.5,
    ) -> None:
        self.max_feature_missingness = max_feature_missingness
        self.max_subject_missingness = max_subject_missingness

    def fit(self, X: Any, y: Any = None, groups: Any = None) -> MissingnessThreshold:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        feat_missingness = np.isnan(x_arr).mean(axis=0)
        self.support_mask_ = feat_missingness <= self.max_feature_missingness
        if groups is not None and np.any(self.support_mask_):
            validate_subject_missingness(
                x_arr[:, self.support_mask_],
                groups,
                maximum=self.max_subject_missingness,
            )
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class SpatialFeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(
        self,
        allowed_regions: Sequence[str] = (),
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.allowed_regions = tuple(allowed_regions)
        self.feature_names = tuple(feature_names) if feature_names is not None else None

    def fit(self, X: Any, y: Any = None) -> SpatialFeatureSelector:
        _ = y
        x_arr = np.asarray(X, dtype=float)
        n_features = x_arr.shape[1]

        if not self.allowed_regions:
            self.support_mask_ = np.ones(n_features, dtype=bool)
            return self

        names = self.feature_names
        if names is None and hasattr(X, "columns"):
            names = tuple(str(c) for c in X.columns)

        if names is None or len(names) != n_features:
            raise ValueError("SpatialFeatureSelector requires feature names when regions are set.")

        allowed_set = {r.strip().lower() for r in self.allowed_regions if r.strip()}
        # Match feature name containing any of the allowed region tokens
        support: list[bool] = []
        for name in names:
            name_lower = name.lower()
            matched = any(r in name_lower for r in allowed_set)
            support.append(matched)

        self.support_mask_ = np.array(support, dtype=bool)
        if not np.any(self.support_mask_):
            raise ValueError(f"No features matched allowed regions: {self.allowed_regions}")
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        return x_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False) -> npt.NDArray[Any]:
        if indices:
            return cast(npt.NDArray[np.intp], np.flatnonzero(self.support_mask_))
        return cast(npt.NDArray[np.bool_], self.support_mask_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return [f for f, keep in zip(input_features, self.support_mask_, strict=True) if keep]


class Deconfounder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, n_covariates: int = 0) -> None:
        self.n_covariates = int(n_covariates)

    def fit(self, X: Any, y: Any = None) -> Deconfounder:
        _ = y
        if self.n_covariates <= 0:
            return self

        x_arr = np.asarray(X, dtype=float)
        n_features = x_arr.shape[1] - self.n_covariates
        if n_features <= 0:
            raise ValueError(
                f"X has {x_arr.shape[1]} columns, but n_covariates={self.n_covariates}."
            )

        features = x_arr[:, :n_features]
        covariates = x_arr[:, n_features:]

        cov_design = np.column_stack([np.ones(len(x_arr)), covariates])
        self.coef_, *_ = np.linalg.lstsq(cov_design, features, rcond=None)
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        x_arr = np.asarray(X, dtype=float)
        if self.n_covariates <= 0:
            return x_arr

        n_features = x_arr.shape[1] - self.n_covariates
        features = x_arr[:, :n_features]
        covariates = x_arr[:, n_features:]

        cov_design = np.column_stack([np.ones(len(x_arr)), covariates])
        residuals = features - (cov_design @ self.coef_)
        return cast(npt.NDArray[np.float64], residuals)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        if self.n_covariates <= 0:
            return list(input_features)
        n_features = len(input_features) - self.n_covariates
        return list(input_features[:n_features])


def base_preprocessing_steps(
    config: PreprocessingConfig,
    *,
    include_scaling: bool,
    n_covariates: int = 0,
    score_func: Callable[..., object] | None = None,
) -> list[tuple[str, object]]:
    feature_steps: list[tuple[str, object]] = [
        ("finite", ReplaceInfWithNaN()),
        ("drop_all_nan", DropAllNaNColumns()),
    ]

    if config.spatial_regions_allowed:
        feature_steps.append(
            (
                "spatial_filter",
                SpatialFeatureSelector(allowed_regions=config.spatial_regions_allowed),
            )
        )

    feature_steps.extend(
        [
            (
                "missingness",
                MissingnessThreshold(
                    max_feature_missingness=config.max_feature_missingness,
                    max_subject_missingness=config.max_subject_missingness,
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
        ]
    )

    if (
        config.feature_selection_percentile is not None
        and config.feature_selection_percentile < 100.0
    ):
        from sklearn.feature_selection import SelectPercentile, f_regression

        chosen_score_func = score_func or f_regression
        feature_steps.append(
            (
                "k_best",
                SelectPercentile(
                    chosen_score_func,
                    percentile=config.feature_selection_percentile,
                ),
            )
        )

    if include_scaling or config.pca_enabled:
        feature_steps.append(("scaler", StandardScaler()))

    if config.pca_enabled:
        pca_components = 0.95 if config.pca_n_components is None else config.pca_n_components
        feature_steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    whiten=config.pca_whiten,
                    random_state=config.pca_random_state,
                    svd_solver=config.pca_svd_solver,
                ),
            )
        )

    if n_covariates > 0:
        cov_steps: list[tuple[str, object]] = [
            ("finite", ReplaceInfWithNaN()),
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
        if include_scaling:
            cov_steps.append(("scaler", StandardScaler()))
        else:
            cov_steps.append(("passthrough", FunctionTransformer(func=None, validate=False)))

        def feature_idx(X: npt.NDArray[Any]) -> list[int]:
            return list(range(X.shape[1] - n_covariates))

        def cov_idx(X: npt.NDArray[Any]) -> list[int]:
            return list(range(X.shape[1] - n_covariates, X.shape[1]))

        preprocessor = ColumnTransformer(
            transformers=[
                ("eeg", Pipeline(feature_steps), feature_idx),
                ("cov", Pipeline(cov_steps), cov_idx),
            ],
            remainder="drop",
        )
        steps: list[tuple[str, object]] = [("preprocessing", preprocessor)]
        if config.deconfound:
            steps.append(("deconfound", Deconfounder(n_covariates=n_covariates)))
        return steps

    return feature_steps


def transform_feature_names(
    steps: Sequence[tuple[str, object]],
    feature_names: Sequence[str],
) -> list[str]:
    names = list(feature_names)
    for _name, step in steps:
        if hasattr(step, "get_feature_names_out"):
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                names = list(step.get_feature_names_out(names))
        elif hasattr(step, "get_support"):
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                support = step.get_support()
                names = [f for f, keep in zip(names, support, strict=True) if keep]
    return names


transform_feature_names_through_steps = transform_feature_names
