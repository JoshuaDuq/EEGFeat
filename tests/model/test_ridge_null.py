from __future__ import annotations

import numpy as np
import pytest

import eegfeat.model.nulls as nulls
from eegfeat.model import _ridge_null
from eegfeat.model.aggregate import AggregationConfig
from eegfeat.model.crossfit import cross_fit_regression
from eegfeat.model.estimators import ridge_grid, ridge_pipeline
from eegfeat.model.nulls import NullConfig, _prediction_statistic, permutation_test
from eegfeat.model.splits import InnerSplit, loso_folds, within_subject_folds
from eegfeat.model.transformers import PreprocessingConfig

GROUPS = np.repeat([f"s{i}" for i in range(6)], 12).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2", "r3"], 4), 6).astype(object)


def _design(missing: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    nuisance = rng.normal(size=GROUPS.size)
    X = rng.normal(size=(GROUPS.size, 20)) + np.outer(nuisance, rng.normal(size=20))
    y = X[:, 0] - 0.5 * X[:, 1] + 2.0 * nuisance + rng.normal(size=GROUPS.size)
    if missing:
        X[rng.random(X.shape) < 0.05] = np.nan
        X[GROUPS == "s2", 3] = np.nan
    return X, y, nuisance.reshape(-1, 1)


def _case(name: str) -> dict[str, object]:
    X, y, nuisance = _design(missing=name == "missing")
    case: dict[str, object] = {
        "folds": loso_folds(GROUPS),
        "X": X,
        "y": y,
        "runs": None,
        "pipeline": ridge_pipeline(PreprocessingConfig(), seed=0),
        "grid": {"regressor__alpha": ridge_grid(X)["regressor__alpha"][2:7:2]},
        "config": NullConfig(n_permutations=9),
        "inner": InnerSplit(grouping="subject", n_splits=2),
        "options": {},
    }
    if name == "missing":
        case["options"] = {"harmonization": "intersection"}
    elif name == "pooled_nuisance":
        case["options"] = {"covariates": nuisance, "residualize_on": ("n",)}
    elif name == "subject_nuisance":
        case["options"] = {
            "covariates": nuisance,
            "residualize_on": ("n",),
            "residualize_within": "subject",
        }
    elif name == "within_subject_folds":
        case["folds"] = within_subject_folds(GROUPS, RUNS, inner_splits=2, outer_splits=3)
        case["runs"] = RUNS
        case["inner"] = InnerSplit(grouping="run", n_splits=2)
        case["config"] = NullConfig(scheme="within_subject_within_run", n_permutations=9)
    elif name == "trial_count":
        case["aggregation"] = AggregationConfig(subject_weighting="trial_count")
    elif name == "deconfounded":
        case["X"] = np.column_stack([X, nuisance])
        case["pipeline"] = ridge_pipeline(
            PreprocessingConfig(deconfound=True), seed=0, n_covariates=1
        )
    elif name == "untuned":
        case["grid"] = {}
    return case


def _null(case: dict[str, object]):
    options = dict(case["options"])  # type: ignore[call-overload]
    aggregation = case.get("aggregation", AggregationConfig())
    predictions = cross_fit_regression(
        case["folds"],
        case["X"],
        case["y"],
        GROUPS,
        case["pipeline"],
        case["grid"],
        inner=case["inner"],
        seed=3,
        runs=case["runs"],
        **options,
    )
    observed = _prediction_statistic(predictions, GROUPS, aggregation, None)
    return permutation_test(
        case["folds"],
        case["X"],
        case["y"],
        GROUPS,
        case["runs"],
        case["pipeline"],
        case["grid"],
        observed,
        config=case["config"],
        inner=case["inner"],
        seed=3,
        aggregation=aggregation,
        **options,
    )


CASES = [
    "loso",
    "missing",
    "pooled_nuisance",
    "subject_nuisance",
    "within_subject_folds",
    "trial_count",
    "deconfounded",
    "untuned",
]


@pytest.mark.parametrize("name", CASES)
def test_the_closed_form_ridge_null_equals_refitting_every_draw(name: str, monkeypatch) -> None:
    # The null is a claim about the procedure that produced the observed statistic, so the
    # shortcut must reproduce every refitted draw, not just a similar distribution.
    fast = _null(_case(name))
    monkeypatch.setattr(_ridge_null, "ridge_penalty", lambda *args, **kwargs: None)
    refitted = _null(_case(name))
    np.testing.assert_allclose(fast.null, refitted.null, rtol=0, atol=1e-9)
    assert fast.p_value == refitted.p_value


def _count_refits(monkeypatch) -> list[int]:
    calls: list[int] = []
    engine = nulls._cross_fit_engine

    def counting(*args, **kwargs):
        calls.append(1)
        return engine(*args, **kwargs)

    monkeypatch.setattr(nulls, "_cross_fit_engine", counting)
    return calls


def test_a_ridge_null_refits_the_procedure_once_not_once_per_draw(monkeypatch) -> None:
    case = _case("loso")
    calls = _count_refits(monkeypatch)
    _null(case)
    assert len(calls) == 1  # the observed statistic's own check


def test_a_step_that_learns_from_the_target_falls_back_to_refitting(monkeypatch) -> None:
    # Univariate selection scores features against y, so preprocessing differs by draw.
    case = _case("loso")
    case["pipeline"] = ridge_pipeline(PreprocessingConfig(feature_selection_percentile=50), seed=0)
    calls = _count_refits(monkeypatch)
    _null(case)
    assert len(calls) == 1 + 9
