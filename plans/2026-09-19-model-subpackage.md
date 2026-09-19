# `eegfeat.model` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `eegfeat.model`, a scikit-learn-based subpackage that fits and evaluates predictive models on `FeatureTable`s, by re-founding the methodology currently in EEG_fMRI_Pipeline.

**Architecture:** Fourteen focused modules, each taking plain arrays and returning values. Nothing in `model/` reads a file or a TOML. The leave-one-subject-out path is ported from array-callable functions that already exist upstream; the within-subject path is extracted from inlined loops and carries the heaviest verification.

**Tech Stack:** Python 3.11+, numpy, scipy, pandas, scikit-learn (as the `model` extra), shap (as the `importance` extra), pytest, mypy strict, ruff, black.

**Spec:** `specs/2026-09-19-model-subpackage-design.md`

**Reference source:** `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/analysis/machine_learning/`. This repository is not modified. Read it; do not edit it.

**Out of scope for this plan:** the recipe runner (`runner/fit_recipe.py`, `runner/fit.py`, `runner/provenance.py`, the `eegfeat fit` command). That is a second plan, written after this one lands.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Code style — this is the point of the exercise, not a formality.**

- No docstring that restates the signature. The upstream `"""Create leave-one-subject-out folds."""` on `create_loso_folds` is exactly what to delete. A public function earns a docstring only when it has something to say that the signature does not: why the method is the one chosen, what invariant it maintains, what a caller would otherwise get wrong. Private helpers (`_name`) get none.
- Keep load-bearing why-comments. `power.py:158` — `# A frequency that produced nothing leaves the average rather than entering it as zero.` — is the model.
- `from __future__ import annotations` first in every module.
- Modern typing only: `str | None`, `dict[str, int]`, `tuple[Fold, ...]`, `Sequence`/`Mapping` from `collections.abc`. Never `Optional`, `Dict`, `List`, `Tuple` from `typing`. The upstream source uses the old spellings throughout; convert on the way over.
- Arrays are `npt.NDArray[np.float64]`, `npt.NDArray[np.intp]`, `npt.NDArray[np.object_]`.
- Options are keyword-only, after `*`.
- Configuration dataclasses are `@dataclass(frozen=True)` with `__post_init__` validation, and live in the module that consumes them — there is no central config module. Follow `bands.Band`.
- Error messages name the offending value: `f"Band {self.name!r} requires fmin < fmax, got {self.fmin} >= {self.fmax}."`

**Never in `model/`:** an import from `eegfeat.runner`, a TOML read, a file write, a `logging` call, or a parameter typed `Any` standing in for a configuration object. Every `config: Any` upstream becomes explicit typed parameters or a frozen dataclass.

**Tests:** pytest. `def test_...() -> None:` with the return annotation. Named for the claim being made, not the function under test — `test_loso_folds_never_share_a_subject_between_train_and_test`, not `test_loso_folds`. No docstrings on tests; the name carries it. Why-comments where the reasoning is not obvious. Shared builders go in `tests/synthetic.py`.

**Limits:** no module over 500 lines. Python >= 3.11. Line length 100 (ruff and black).

**Two contracts that cut across tasks** (spec §3.7, §3.8):

- *Inner splits are declared, never inferred.* `tune` takes an `InnerSplit` naming the
  grouping (`"subject"` or `"run"`) and whether to stratify, plus the array to group by.
  A within-subject fold trains on one subject, so subject-grouping is impossible there —
  upstream groups on runs. Classification stratifies; regression does not.
- *Every operation states its independent unit.* Outer folds: subject, or run for
  within-subject. Subject-level metrics, bootstrap CIs and sign-flip tests: subject.
  Permutations: whatever the scheme declares. Conformal intervals: trial, giving marginal
  coverage only. Put it in the docstring where a reader could assume otherwise.

**One correction the port makes deliberately** (spec §3.6): the permutation effectiveness
filter is removed. Upstream skips a permutation that changed too few labels
(`cv.py:1238`); that conditions the null on the permuted values and breaks the
randomization argument `circular_shift.py` itself states. Every sampled permutation
counts; the changed fraction is reported, not acted on. P-values will not match the
reference on this path, by design.

**Green at every commit:**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
```

Baseline at `6f22bd4`: 485 passed, 4 skipped in 8.05s; mypy clean on 31 source files; ruff clean. The suite runs in eight seconds — run all of it, every time.

**Porting discipline.** Most implementation steps port a named function. The step names the source file and function; you read it there and apply the stated transformation. Do not paste upstream code unchanged — the old typing spellings, the `config: Any` parameters and the signature-restating docstrings all have to go. Where a step shows code, the code is the target, not a sketch.

---

## File Structure

```
src/eegfeat/model/
  __init__.py       public surface and the scikit-learn guard
  _deps.py          raises a clear error when scikit-learn is absent
  splits.py         Fold, and every fold constructor
  scoring.py        correlation scorers and scoring dictionaries
  aggregate.py      subject-level aggregation, bootstrap CI, sign-flip test
  transformers.py   scikit-learn transformers and the base step list
  metrics.py        metric computation and ClassificationResult
  estimators.py     estimator pipelines and parameter grids
  residualize.py    fold-local nuisance regression on targets
  execution.py      seeds and fold parallelism
  design.py         FeatureTable -> design matrix, and fold harmonization
  tuning.py         inner-CV tuning of one estimator on one fold
  crossfit.py       the outer fold loop
  nulls.py          permutation schemes and effectiveness
  uncertainty.py    conformal prediction intervals
  importance.py     SHAP and permutation importance

tests/
  model/test_*.py           one per module
  model/test_boundary.py    the model/ import invariant
  test_equivalence_model.py equivalence against the reference pipeline
scripts/make_fixtures.py    extended with a --model mode
```

Dependency order, which is the task order: `splits`, `scoring`, `aggregate`, `transformers` depend on nothing in the subpackage; `metrics`, `estimators`, `residualize`, `execution` depend on those; `design` on `eegfeat.table`; `tuning` on `splits` and `estimators`; `crossfit` on everything; `nulls`, `uncertainty`, `importance` on `crossfit`.

---

### Task 1: Subpackage scaffolding and the scikit-learn guard

**Files:**
- Create: `src/eegfeat/model/__init__.py`, `src/eegfeat/model/_deps.py`
- Create: `tests/model/__init__.py` (empty), `tests/model/test_boundary.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: importing `eegfeat.model` without scikit-learn raises `ModuleNotFoundError` naming the extra. `eegfeat.model` never appears in `eegfeat/__init__.py`.

- [ ] **Step 1: Write the failing boundary tests**

`tests/model/test_boundary.py`:

```python
from __future__ import annotations

import ast
import pathlib

import pytest

MODEL = pathlib.Path(__file__).parents[2] / "src" / "eegfeat" / "model"


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(MODEL.glob("*.py")), ids=lambda p: p.name)
def test_no_model_module_imports_the_runner(path: pathlib.Path) -> None:
    assert not {n for n in _imported_modules(path) if n.startswith("eegfeat.runner")}


@pytest.mark.parametrize("path", sorted(MODEL.glob("*.py")), ids=lambda p: p.name)
def test_no_model_module_reads_toml_or_writes_files(path: pathlib.Path) -> None:
    # The science half takes arrays and returns values. Reading a recipe or writing a
    # result is the runner's job, and keeping that true is what makes every module here
    # testable with arrays alone.
    forbidden = {"tomllib", "tomli", "logging"}
    assert not _imported_modules(path) & forbidden
    source = path.read_text()
    assert "open(" not in source
    assert ".write_text(" not in source


def test_top_level_eegfeat_does_not_import_model() -> None:
    # Importing eegfeat must not cost scikit-learn.
    init = (MODEL.parent / "__init__.py").read_text()
    assert "eegfeat.model" not in init
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_boundary.py -v`
Expected: collection produces no parametrized cases for the two `glob` tests (the directory does not exist), and `test_top_level_eegfeat_does_not_import_model` passes vacuously. Create the directory first if collection errors.

- [ ] **Step 3: Create the guard and the package**

`src/eegfeat/model/_deps.py`:

```python
from __future__ import annotations

from importlib.util import find_spec

if find_spec("sklearn") is None:
    raise ModuleNotFoundError(
        "eegfeat.model requires scikit-learn. Install it with: pip install 'eegfeat[model]'"
    )
```

`src/eegfeat/model/__init__.py`:

```python
from __future__ import annotations

# Imported for its side effect: it reports the missing extra by name, before a submodule
# fails on `import sklearn` with a message that does not say what to install.
from eegfeat.model import _deps as _deps

__all__: list[str] = []
```

- [ ] **Step 4: Add the extras**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
model = ["scikit-learn>=1.3"]
importance = ["scikit-learn>=1.3", "shap>=0.44"]
```

Add `"shap.*"` to the existing `[[tool.mypy.overrides]]` module list that already carries `sklearn.*`.

- [ ] **Step 5: Verify green**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
```
Expected: 485 passed plus the new boundary tests; mypy and ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/eegfeat/model tests/model pyproject.toml
git commit -m "feat(model): subpackage scaffolding and the scikit-learn guard"
```

---

### Task 2: `splits.py` — folds

**Files:**
- Create: `src/eegfeat/model/splits.py`, `tests/model/test_splits.py`
- Source: `cv.py` — `create_loso_folds`, `create_inner_cv`, `create_run_aware_cv`, `create_within_subject_folds`, `create_run_aware_inner_cv`, `get_inner_cv_splits`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Fold:
      index: int
      train: npt.NDArray[np.intp]
      test: npt.NDArray[np.intp]
      subject: str | None = None

  def loso_folds(groups: npt.NDArray[np.object_]) -> tuple[Fold, ...]
  def within_subject_folds(
      groups: npt.NDArray[np.object_],
      blocks: npt.NDArray[np.object_] | None,
      *, inner_splits: int, seed: int,
      outer_splits: int | None = None, ordered_runs: bool = False,
  ) -> tuple[Fold, ...]
  @dataclass(frozen=True)
  class InnerSplit:
      grouping: Literal["subject", "run"]
      stratified: bool = False
      n_splits: int = 5

  def inner_cv(
      train_groups: npt.NDArray[np.object_], split: InnerSplit,
      y_train: npt.NDArray[np.intp] | None = None,
  ) -> GroupKFold | StratifiedGroupKFold
  def run_aware_cv(
      blocks: npt.NDArray[np.object_], *, n_splits: int | None = None, default_splits: int = 5
  ) -> tuple[GroupKFold | None, int]
  def run_aware_inner_cv(
      blocks_train: npt.NDArray[np.object_], n_splits: int, seed: int, fold: int, subject: str
  ) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] | None
  def inner_cv_splits(n_unique_groups: int, *, default: int = 5) -> int
  ```

`Fold` is the uniform record that lets one outer loop consume either fold source (Task 13). `subject` is set by `within_subject_folds` and left `None` by `loso_folds`.

`InnerSplit` is the declaration that makes Task 11 and Task 13 possible at all. A
within-subject outer fold trains on **one subject**, so a subject-grouped inner split
cannot yield two groups; upstream tunes those folds with
`create_run_aware_inner_cv(blocks_train, ...)` and passes `groups=blocks_train`
(`orchestration.py:868`). Classification needs `StratifiedGroupKFold`, pinned upstream by
`nested_loso_classification_requires_stratified_inner_cv_support`. `inner_cv` returns the
stratified splitter when `split.stratified`, which is why it takes `y_train`.

Three upstream parameters are dropped: `create_loso_folds`'s `X` (only its length was used — derive from `groups`), and `create_within_subject_folds`'s `epochs` and `apply_hygiene` (fold-local IAF is deferred, spec §3.6). `config` becomes `ordered_runs` and `default_splits`.

- [ ] **Step 1: Write the failing tests**

`tests/model/test_splits.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.splits import Fold, InnerSplit, inner_cv, inner_cv_splits, loso_folds

GROUPS = np.array(["s1", "s1", "s2", "s2", "s3"], dtype=object)


def test_loso_folds_never_share_a_subject_between_train_and_test() -> None:
    for fold in loso_folds(GROUPS):
        assert not set(GROUPS[fold.train]) & set(GROUPS[fold.test])


def test_loso_folds_hold_out_exactly_one_subject_each() -> None:
    folds = loso_folds(GROUPS)
    assert len(folds) == 3
    assert [set(GROUPS[f.test]) for f in folds] == [{"s1"}, {"s2"}, {"s3"}]


def test_loso_folds_cover_every_trial_exactly_once_across_test_sets() -> None:
    tested = np.concatenate([f.test for f in loso_folds(GROUPS)])
    np.testing.assert_array_equal(np.sort(tested), np.arange(GROUPS.size))


def test_loso_folds_carry_no_subject_label() -> None:
    assert all(f.subject is None for f in loso_folds(GROUPS))


def test_inner_cv_refuses_a_single_training_group() -> None:
    # Tuning inside a fold needs at least two groups to split on; with one, the inner
    # split is not group-disjoint and the tuned hyperparameters leak the held-out subject.
    with pytest.raises(ValueError, match="at least 2"):
        inner_cv(
            np.array(["s1", "s1", "s1"], dtype=object),
            InnerSplit(grouping="run", n_splits=3),
        )


def test_inner_cv_splits_are_capped_by_available_groups() -> None:
    assert inner_cv_splits(3, default=5) == 3
    assert inner_cv_splits(10, default=5) == 5


def test_a_within_subject_fold_cannot_be_grouped_by_subject() -> None:
    # Its training rows are one subject, so a subject-grouped inner split has one group.
    # Refusing the combination here beats a puzzling "at least 2 groups" from inside tuning.
    with pytest.raises(ValueError, match="within-subject"):
        inner_cv(
            np.array(["s1"] * 6, dtype=object),
            InnerSplit(grouping="subject", n_splits=3),
        )


def test_a_stratified_inner_split_requires_labels() -> None:
    with pytest.raises(ValueError, match="y_train"):
        inner_cv(
            np.array(["r1", "r2", "r3"], dtype=object),
            InnerSplit(grouping="run", stratified=True, n_splits=2),
        )


def test_fold_is_frozen() -> None:
    fold = Fold(index=1, train=np.array([0], dtype=np.intp), test=np.array([1], dtype=np.intp))
    with pytest.raises(AttributeError):
        fold.index = 2  # type: ignore[misc]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eegfeat.model.splits'`

- [ ] **Step 3: Implement**

Port the six functions from `cv.py`. Apply the transformations above, convert the typing spellings, and delete the signature-restating docstrings. `inner_cv` keeps its `ValueError` — it is safeguard 1 and the test above pins it.

`within_subject_folds` is the long one (121 lines upstream). Port it faithfully; its ordering and minimum-block rules are pinned by validity tests ported in Step 5.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining split constraints**

Add tests for these constraints, from the upstream validity suite (`tests/machine_learning/test_machine_learning_validity_fixes.py`), reading each one there for the exact behaviour and rewriting it in the style above:

- `create_within_subject_folds_respects_outer_cv_splits`
- `create_within_subject_folds_requires_runs`
- `create_within_subject_folds_rejects_insufficient_subject_blocks`
- `create_within_subject_folds_supports_forward_ordering`
- `create_within_subject_folds_raises_when_ordered_runs_cannot_be_formed`
- `create_inner_cv_requires_two_groups` (already covered above — confirm no gap)
- `find_run_column_parses_run_prefixed_labels`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/splits.py tests/model/test_splits.py
git commit -m "feat(model): fold construction with a uniform Fold record"
```

---

### Task 3: `scoring.py` — correlation scorers

**Files:**
- Create: `src/eegfeat/model/scoring.py`, `tests/model/test_scoring.py`
- Source: `cv.py` — `safe_pearsonr`, `make_pearsonr_scorer`, `create_scoring_dict`

**Interfaces:**
- Produces:
  ```python
  def safe_pearsonr(
      x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *, min_variance: float = 1e-10
  ) -> tuple[float, float]
  def pearsonr_scorer() -> Callable[..., float]
  def scoring_dict() -> dict[str, object]
  ```

`make_pearsonr_scorer` has no upstream return annotation; annotate it.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.scoring import safe_pearsonr, scoring_dict


def test_perfect_correlation_is_one() -> None:
    r, _ = safe_pearsonr(np.arange(5.0), 2.0 * np.arange(5.0))
    assert r == pytest.approx(1.0)


def test_a_constant_predictor_gives_nan_rather_than_a_divide_by_zero() -> None:
    # Zero variance makes Pearson undefined. Returning NaN keeps the fold in the record
    # as unscored; returning 0.0 would claim a measured absence of correlation.
    r, p = safe_pearsonr(np.zeros(5), np.arange(5.0))
    assert np.isnan(r) and np.isnan(p)


def test_fewer_than_two_finite_pairs_gives_nan() -> None:
    r, p = safe_pearsonr(np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(r) and np.isnan(p)


def test_non_finite_pairs_are_dropped_not_propagated() -> None:
    r, _ = safe_pearsonr(
        np.array([1.0, 2.0, 3.0, np.nan]), np.array([2.0, 4.0, 6.0, 1.0])
    )
    assert r == pytest.approx(1.0)


def test_scoring_dict_exposes_the_correlation_scorer() -> None:
    assert "r" in scoring_dict() or "pearson_r" in scoring_dict()
```

The last assertion accepts either key; read `create_scoring_dict` in `cv.py` and pin the real key before committing.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_scoring.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the three functions. `safe_pearsonr`'s `min_variance` becomes keyword-only. Keep the clip to `[-1, 1]`; add a why-comment saying it guards floating-point overshoot, not a modelling choice.

- [ ] **Step 4: Run to verify they pass, fixing the scoring-dict key assertion to the real key**

Run: `.venv/bin/python -m pytest tests/model/test_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/scoring.py tests/model/test_scoring.py
git commit -m "feat(model): correlation scorers with explicit undefined-case handling"
```

---

### Task 4: `aggregate.py` — subject-level aggregation

**Files:**
- Create: `src/eegfeat/model/aggregate.py`, `tests/model/test_aggregate.py`
- Source: `cv.py` — `aggregate_fold_results`, `compute_subject_level_r`, `compute_subject_level_errors`; `orchestration.py` — `_subject_mean_metric`, `_count_finite_subject_metric`, `_subject_metric_values`, `_bootstrap_mean_ci`, `_paired_signflip_p_value`, `_subject_weighted_r2_scores`

**Interfaces:**
- Consumes: `eegfeat.model.scoring.safe_pearsonr`
- Produces:
  ```python
  @dataclass(frozen=True)
  class AggregationConfig:
      subject_weighting: Literal["equal", "trial_count"] = "equal"
      bootstrap_iterations: int = 10_000
      ci_method: Literal["fixed_effects", "bootstrap"] = "fixed_effects"

  @dataclass(frozen=True)
  class SubjectLevelR:
      r: float
      per_subject: tuple[tuple[str, float], ...]
      ci_low: float
      ci_high: float

  def subject_level_r(
      predictions: pd.DataFrame, *, config: AggregationConfig = AggregationConfig()
  ) -> SubjectLevelR
  def subject_level_errors(
      predictions: pd.DataFrame, *, config: AggregationConfig = AggregationConfig()
  ) -> dict[str, float]
  def fold_results(results: Sequence[dict[str, object]]) -> tuple[
      npt.NDArray[np.float64], npt.NDArray[np.float64], list[str], list[int], list[int]
  ]
  def bootstrap_mean_ci(
      values: npt.NDArray[np.float64], *, iterations: int, seed: int
  ) -> tuple[float, float]
  def paired_signflip_p_value(
      differences: npt.NDArray[np.float64], *, iterations: int, seed: int
  ) -> float
  ```

`AggregationConfig` replaces the `config: Optional[Any]` on both upstream functions; its two fields are the only keys they read (`machine_learning.evaluation.subject_weighting`, `machine_learning.evaluation.bootstrap_iterations`). `compute_subject_level_r` returns a bare 4-tuple upstream; `SubjectLevelR` names the parts.

`predictions` is a DataFrame with columns `subject_id`, `y_true`, `y_pred`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegfeat.model.aggregate import AggregationConfig, subject_level_r


def _predictions(per_subject: dict[str, tuple[list[float], list[float]]]) -> pd.DataFrame:
    rows = [
        {"subject_id": subject, "y_true": t, "y_pred": p}
        for subject, (truths, preds) in per_subject.items()
        for t, p in zip(truths, preds, strict=True)
    ]
    return pd.DataFrame(rows)


def test_each_subject_counts_once_regardless_of_trial_count() -> None:
    # A subject with forty trials and one with four contribute equally. Pooling instead
    # would let the largest subject decide the cohort result.
    small = ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    large = ([float(i) for i in range(40)], [float(-i) for i in range(40)])
    result = subject_level_r(_predictions({"s1": small, "s2": large}))
    assert result.r == pytest.approx(0.0, abs=1e-9)


def test_per_subject_correlations_are_reported_individually() -> None:
    result = subject_level_r(
        _predictions(
            {
                "s1": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                "s2": ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
            }
        )
    )
    assert dict(result.per_subject) == pytest.approx({"s1": 1.0, "s2": -1.0})


def test_a_subject_whose_correlation_is_undefined_does_not_become_zero() -> None:
    # A flat predictor within a subject is unscored, not scored as no correlation.
    result = subject_level_r(
        _predictions(
            {
                "s1": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                "s2": ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
            }
        )
    )
    assert result.r == pytest.approx(1.0)


def test_aggregation_config_rejects_an_unknown_weighting() -> None:
    with pytest.raises(ValueError, match="subject_weighting"):
        AggregationConfig(subject_weighting="by_vibes")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_aggregate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the functions. Add `AggregationConfig.__post_init__` validating `subject_weighting`, `ci_method` and `bootstrap_iterations > 0`. Fisher-z aggregation is the upstream method for `subject_level_r`; keep it and add a why-comment saying correlations are averaged in z rather than in r because r is not additive.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_aggregate.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining aggregation constraints**

From the upstream validity suite:

- `subject_level_r_uses_equal_subject_weighting_by_default`
- `subject_level_r_rejects_invalid_subjects`
- `subject_level_errors_reject_invalid_subjects`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/aggregate.py tests/model/test_aggregate.py
git commit -m "feat(model): subject-level aggregation with equal subject weight"
```

---

### Task 5: `transformers.py` — scikit-learn transformers

**Files:**
- Create: `src/eegfeat/model/transformers.py`, `tests/model/test_transformers.py`
- Source: `preprocessing.py` — all of it

**Interfaces:**
- Produces:
  ```python
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

  class VarianceThreshold(BaseEstimator, TransformerMixin)
  class MissingnessThreshold(BaseEstimator, TransformerMixin)
  class ReplaceInfWithNaN(BaseEstimator, TransformerMixin)
  class DropAllNaNColumns(BaseEstimator, TransformerMixin)
  class SpatialFeatureSelector(BaseEstimator, TransformerMixin)
  class Deconfounder(BaseEstimator, TransformerMixin)

  def validate_subject_missingness(
      values: npt.NDArray[np.float64], groups: npt.NDArray[np.object_], *, maximum: float
  ) -> None
  def base_preprocessing_steps(
      config: PreprocessingConfig, *, include_scaling: bool, n_covariates: int = 0,
      score_func: Callable[..., object] | None = None,
  ) -> list[tuple[str, object]]
  def transform_feature_names(
      steps: Sequence[tuple[str, object]], feature_names: Sequence[str]
  ) -> list[str]
  ```

`PreprocessingConfig`'s ten fields are exactly the keys `build_base_preprocessing_steps` reads from its `cfg` dict; it replaces both that dict and the separate `config: Any`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.transformers import (
    DropAllNaNColumns,
    ReplaceInfWithNaN,
    validate_subject_missingness,
)


def test_a_subject_above_the_missingness_limit_is_named_in_the_error() -> None:
    values = np.array([[1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan]])
    groups = np.array(["s1", "s2", "s2"], dtype=object)
    with pytest.raises(ValueError, match="s2"):
        validate_subject_missingness(values, groups, maximum=0.2)


def test_missingness_is_checked_per_subject_not_over_the_pooled_matrix() -> None:
    # Pooled missingness here is 25%, under the limit; s2's is 100%. A subject with no
    # usable features must fail even when the cohort average looks acceptable.
    values = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    groups = np.array(["s1", "s1", "s2"], dtype=object)
    with pytest.raises(ValueError, match="s2"):
        validate_subject_missingness(values, groups, maximum=0.3)


def test_missingness_requires_at_least_one_retained_feature() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_subject_missingness(
            np.empty((2, 0)), np.array(["s1", "s2"], dtype=object), maximum=0.5
        )


def test_infinities_become_nan_so_imputation_can_see_them() -> None:
    out = ReplaceInfWithNaN().fit_transform(np.array([[1.0, np.inf], [-np.inf, 2.0]]))
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])


def test_all_nan_columns_are_dropped_and_the_drop_is_learned_on_fit() -> None:
    # The columns to drop are decided by the training block and reapplied to test data,
    # so a column that happens to be present at test time is still dropped.
    train = np.array([[1.0, np.nan], [2.0, np.nan]])
    step = DropAllNaNColumns().fit(train)
    assert step.transform(np.array([[3.0, 9.0]])).shape == (1, 1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_transformers.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port all six transformers and three functions. Every transformer keeps its learned state on `fit` and reapplies it on `transform` — that is safeguard 2 and the last test pins it. Add `PreprocessingConfig.__post_init__` validating both missingness fractions are in `[0, 1]`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_transformers.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining transformer constraints**

- `spatial_feature_selector_requires_feature_names_when_regions_are_requested`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/transformers.py tests/model/test_transformers.py
git commit -m "feat(model): fold-local preprocessing transformers"
```

---

### Task 6: `metrics.py` — metric computation

**Files:**
- Create: `src/eegfeat/model/metrics.py`, `tests/model/test_metrics.py`
- Source: `cv.py` — `compute_metrics`; `classification.py` — `ClassificationResult`; `orchestration.py` — `_within_subject_centered_prediction_metrics`, `_within_condition_prediction_metrics`, `_within_condition_cells`, `_center_within_cells`

**Interfaces:**
- Consumes: `aggregate.subject_level_r`, `aggregate.AggregationConfig`, `scoring.safe_pearsonr`
- Produces:
  ```python
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

  def classification_metrics(
      y_true: npt.NDArray[np.intp], y_pred: npt.NDArray[np.intp],
      *, y_prob: npt.NDArray[np.float64] | None = None,
      groups: npt.NDArray[np.object_] | None = None,
  ) -> ClassificationResult
  def regression_metrics(
      y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64],
      groups: npt.NDArray[np.object_] | None = None,
      *, config: AggregationConfig = AggregationConfig(),
  ) -> tuple[dict[str, float], list[dict[str, object]]]
  def within_subject_centered_metrics(...) -> dict[str, float]
  def within_condition_metrics(...) -> dict[str, float]
  ```

Three changes from upstream's `ClassificationResult` (`classification.py:385`), which mixes
metrics, ROC curve data and fold bookkeeping in one mutable object:

- It becomes **frozen**, built by `classification_metrics` rather than computing itself in
  `__post_init__`. Frozen containers built by functions is the house pattern — see
  `table.FeatureTable`.
- `fold_ids`, `test_indices`, `failed_fold_count` and `n_folds_total` do not cross. That is
  fold bookkeeping and belongs to `crossfit.FoldPrediction` (Task 13).
- `fpr`, `tpr` and `thresholds` do not cross. They are plotting data, recomputable from
  `y_true` and `y_prob`, and plotting stays in the pipeline.

`per_subject` is upstream's `per_subject_metrics`. Read `_compute_metrics`
(`classification.py:457`, 42 lines) for how the per-subject aggregation works. `balanced_accuracy`
is the subject-level figure, never the pooled one — safeguard 4.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.metrics import classification_metrics, regression_metrics


def test_a_single_class_fold_does_not_report_a_balanced_accuracy() -> None:
    # One class in the held-out subject makes balanced accuracy undefined. Reporting NaN
    # keeps the fold visible as unscored; reporting 0.5 would invent a chance result.
    result = classification_metrics(
        np.array([1, 1, 1]),
        np.array([1, 1, 0]),
        groups=np.array(["s1", "s1", "s1"], dtype=object),
    )
    assert np.isnan(result.balanced_accuracy)


def test_a_single_class_confusion_matrix_still_has_both_axes() -> None:
    # A 1x1 matrix would silently reindex downstream, turning "no negatives were seen"
    # into "no negatives exist".
    result = classification_metrics(np.array([1, 1]), np.array([1, 1]))
    assert result.confusion.shape == (2, 2)


def test_regression_metrics_report_subject_level_r() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    groups = np.array(["s1"] * 3 + ["s2"] * 3, dtype=object)
    summary, _ = regression_metrics(y_true, y_true.copy(), groups)
    assert summary["subject_level_r"] == pytest.approx(1.0)
```

The metric key `subject_level_r` is a target name; read `cv.py:919` (`compute_metrics`) for the upstream key spellings and reconcile before Step 4.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the functions. `compute_metrics`'s `config: Optional[Any]` becomes the `AggregationConfig` from Task 4. Primary classification metrics are subject-level with equal subject weight, never pooled — safeguard 4.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining metric constraints**

- `classification_result_handles_single_class_confusion_matrix`
- `classification_result_marks_single_class_subject_balanced_accuracy_nan`
- `classification_primary_balanced_accuracy_stays_subject_level_only`
- `classification_primary_precision_recall_f1_are_subject_level`
- `classification_calibration_uses_finite_probabilities_only`
- `classification_reports_subject_level_confidence_intervals`
- `group_classification_permutations_do_not_fallback_to_pooled_auc`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/metrics.py tests/model/test_metrics.py
git commit -m "feat(model): subject-level regression and classification metrics"
```

---

### Task 7: `estimators.py` — pipelines and grids

**Files:**
- Create: `src/eegfeat/model/estimators.py`, `tests/model/test_estimators.py`
- Source: `pipelines.py` — all; `classification.py` — `create_svm_pipeline`, `create_logistic_pipeline`, `create_rf_classification_pipeline`, `create_ensemble_pipeline`, `build_svm_param_grid`, `build_logistic_param_grid`, `build_rf_classification_param_grid`, `_append_classification_resampler`, `_variance_param_prefix`, `_get_lr_kwargs`

**Interfaces:**
- Consumes: `transformers.PreprocessingConfig`, `transformers.base_preprocessing_steps`
- Produces:
  ```python
  def elasticnet_pipeline(config: PreprocessingConfig, *, seed: int, n_covariates: int = 0) -> Pipeline
  def ridge_pipeline(...) -> Pipeline
  def random_forest_pipeline(...) -> Pipeline
  def svm_pipeline(...) -> Pipeline
  def logistic_pipeline(...) -> Pipeline
  def random_forest_classifier_pipeline(...) -> Pipeline
  def ensemble_pipeline(...) -> Pipeline
  def elasticnet_grid(*, n_covariates: int = 0) -> dict[str, list[object]]
  def ridge_grid(...) -> dict[str, list[object]]
  def random_forest_grid(...) -> dict[str, list[object]]
  def svm_grid(...) -> dict[str, list[object]]
  def logistic_grid(...) -> dict[str, list[object]]
  def random_forest_classifier_grid(...) -> dict[str, list[object]]
  ```

Every `build_*_param_grid` takes `config: Any = None` upstream but reads no `machine_learning.*` key (verified: `pipelines.py` has zero config lookups); drop the parameter rather than typing it.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from eegfeat.model.estimators import elasticnet_grid, elasticnet_pipeline
from eegfeat.model.transformers import PreprocessingConfig

CONFIG = PreprocessingConfig()


def test_every_grid_key_names_a_step_that_exists_in_its_pipeline() -> None:
    # A grid key that does not resolve is not an error in GridSearchCV until fit time,
    # and then it reports a parameter name rather than the typo that caused it.
    pipe = elasticnet_pipeline(CONFIG, seed=0)
    valid = set(pipe.get_params())
    assert set(elasticnet_grid()) <= valid


def test_the_pipeline_scales_before_it_regularizes() -> None:
    # ElasticNet penalizes coefficients on their own scale, so an unscaled feature in
    # different units is regularized differently from an identical one in volts.
    names = [name for name, _ in elasticnet_pipeline(CONFIG, seed=0).steps]
    assert names.index("scaler") < names.index("regressor")


def test_the_seed_reaches_the_estimator() -> None:
    pipe = elasticnet_pipeline(CONFIG, seed=17)
    assert pipe.named_steps["regressor"].random_state == 17


def test_pipelines_are_built_fresh_not_shared() -> None:
    assert elasticnet_pipeline(CONFIG, seed=0) is not elasticnet_pipeline(CONFIG, seed=0)
```

Step names `scaler` and `regressor` are the targets; read `pipelines.py:24` for the upstream spellings and reconcile before Step 4.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_estimators.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the seven pipeline builders and six grid builders. Keep `_variance_param_prefix`'s covariate-aware prefixing — it is why the grids take `n_covariates`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_estimators.py -v`
Expected: PASS

- [ ] **Step 5: Extend the grid-key test across every estimator**

Parametrize `test_every_grid_key_names_a_step_that_exists_in_its_pipeline` over all six (pipeline, grid) pairs. This is cheap and catches the most common porting slip.

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/estimators.py tests/model/test_estimators.py
git commit -m "feat(model): estimator pipelines and their parameter grids"
```

---

### Task 8: `residualize.py` — fold-local nuisance regression

**Files:**
- Create: `src/eegfeat/model/residualize.py`, `tests/model/test_residualize.py`
- Source: `target_residualization.py` — all except `configured_target_residualization_columns`; `orchestration.py` — `_StagedResidualPreprocessor`, `_fit_staged_residual_preprocessor`, `reconstruct_staged_permutation_target_for_fold`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class FoldNuisanceFit:
      train_target: npt.NDArray[np.float64]
      test_target: npt.NDArray[np.float64]
      train_prediction: npt.NDArray[np.float64]
      test_prediction: npt.NDArray[np.float64]
      train_residual: npt.NDArray[np.float64]
      test_residual: npt.NDArray[np.float64]
      details: Mapping[str, float]

  def fit_nuisance_model(
      y: npt.NDArray[np.float64], covariates: npt.NDArray[np.float64],
      train: npt.NDArray[np.intp], *, columns: Sequence[str],
  ) -> FoldNuisanceFit
  def residualize_targets(
      y: npt.NDArray[np.float64], covariates: npt.NDArray[np.float64],
      train: npt.NDArray[np.intp], test: npt.NDArray[np.intp], *, columns: Sequence[str],
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
  ```

`configured_target_residualization_columns` does not cross; the caller passes `columns`.
Upstream `details` is `dict[str, Any]`; read its construction site and narrow it to the
concrete value type before committing — `Any` does not pass mypy strict here.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from eegfeat.model.residualize import residualize_targets

TRAIN = np.arange(8, dtype=np.intp)
TEST = np.arange(8, 12, dtype=np.intp)


def test_the_nuisance_model_is_fitted_on_training_rows_only() -> None:
    # If the test rows entered the fit, the test residuals would be centred by
    # construction and the held-out score would be optimistic.
    y = np.concatenate([np.arange(8, dtype=float), np.full(4, 100.0)])
    covariates = np.concatenate([np.arange(8, dtype=float), np.zeros(4)]).reshape(-1, 1)
    _, y_test = residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])
    assert np.all(y_test > 50.0)


def test_a_covariate_that_explains_the_target_leaves_near_zero_residuals() -> None:
    y = np.arange(12, dtype=float)
    covariates = np.arange(12, dtype=float).reshape(-1, 1)
    y_train, _ = residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])
    assert np.allclose(y_train, 0.0, atol=1e-9)


def test_a_rank_deficient_training_design_is_refused() -> None:
    # A constant covariate adds nothing to the intercept; silently inverting a singular
    # design would return residuals that depend on the pseudo-inverse's tie-breaking.
    y = np.arange(12, dtype=float)
    covariates = np.ones((12, 1))
    with pytest.raises(ValueError):
        residualize_targets(y, covariates, TRAIN, TEST, columns=["c"])


def test_train_and_test_indices_may_not_overlap() -> None:
    y = np.arange(12, dtype=float)
    covariates = np.arange(12, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError):
        residualize_targets(y, covariates, TRAIN, TRAIN, columns=["c"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_residualize.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the functions including `_validate_indices` and `_validate_training_nuisance_rank`, which the last two tests pin.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_residualize.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining residualization constraints**

From `tests/machine_learning/test_target_residualization.py` and `test_staged_residual_nesting.py` — read both files and rewrite every test they contain in the style above. The staged-residual nesting tests are the ones that pin fold-local ordering.

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/residualize.py tests/model/test_residualize.py
git commit -m "feat(model): fold-local target residualization"
```

---

### Task 9: `execution.py` — seeds and parallelism

**Files:**
- Create: `src/eegfeat/model/execution.py`, `tests/model/test_execution.py`
- Source: `cv.py` — `set_random_seeds`, `determine_inner_n_jobs`, `should_parallelize_folds`, `execute_folds_parallel`

**Interfaces:**
- Consumes: `splits.Fold`
- Produces:
  ```python
  def set_random_seeds(seed: int, fold: int) -> None
  def inner_n_jobs(outer_n_jobs: int, n_jobs: int) -> int
  def should_parallelize(outer_n_jobs: int, n_folds: int) -> bool
  def run_folds(
      folds: Sequence[Fold], work: Callable[[Fold], dict[str, object]], *, outer_n_jobs: int
  ) -> list[dict[str, object]]
  ```

`execute_folds_parallel` takes `List[Tuple]` upstream; it takes `Sequence[Fold]` here.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np

from eegfeat.model.execution import run_folds, set_random_seeds, should_parallelize
from eegfeat.model.splits import Fold

FOLDS = tuple(
    Fold(index=i, train=np.array([0], dtype=np.intp), test=np.array([1], dtype=np.intp))
    for i in range(3)
)


def test_fold_results_come_back_in_fold_order_not_completion_order() -> None:
    # Downstream aggregation indexes fold records positionally, so a parallel backend
    # returning them as they finish would silently misattribute per-fold results.
    results = run_folds(FOLDS, lambda fold: {"index": fold.index}, outer_n_jobs=2)
    assert [r["index"] for r in results] == [0, 1, 2]


def test_the_same_seed_and_fold_reproduce_the_same_draw() -> None:
    set_random_seeds(42, 1)
    first = np.random.random()
    set_random_seeds(42, 1)
    assert np.random.random() == first


def test_different_folds_do_not_share_a_draw() -> None:
    set_random_seeds(42, 1)
    first = np.random.random()
    set_random_seeds(42, 2)
    assert np.random.random() != first


def test_a_single_fold_is_not_parallelized() -> None:
    assert not should_parallelize(outer_n_jobs=4, n_folds=1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_execution.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the four functions. `run_folds` must preserve input order — the first test pins it.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_execution.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining execution constraint**

- `execute_folds_parallel_handles_userwarning_filter`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/execution.py tests/model/test_execution.py
git commit -m "feat(model): seeded, order-preserving fold execution"
```

---

### Task 10: `design.py` — FeatureTable to design matrix

**Files:**
- Create: `src/eegfeat/model/design.py`, `tests/model/test_design.py`
- Source: `cv.py` — `compute_train_group_intersection_mask`, `apply_fold_feature_harmonization`; `orchestration.py` — `_target_covariate_aliases`, `_warn_or_raise_if_binary_like_regression_target`, `_apply_fold_feature_harmonization_foldwise`; the `[select]` query is new

**Interfaces:**
- Consumes: `eegfeat.table.FeatureTable`, `eegfeat.table.FeatureMeta`
- Produces:
  ```python
  @dataclass(frozen=True)
  class Selection:
      measure: tuple[str, ...] = ()
      band: tuple[str, ...] = ()
      space_kind: tuple[str, ...] = ()
      window: tuple[str, ...] = ()
      normalization: tuple[str, ...] = ()

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

  def select(table: FeatureTable, selection: Selection) -> FeatureTable
  def build_design(
      table: FeatureTable, targets: pd.DataFrame, *, target: str,
      groups: str = "subject_id", runs: str | None = None,
      covariates: Sequence[str] = (), selection: Selection = Selection(),
  ) -> Design
  def harmonize_fold(
      X_train: npt.NDArray[np.float64], X_test: npt.NDArray[np.float64],
      groups_train: npt.NDArray[np.object_], *, mode: str | None, n_covariates: int = 0,
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]
  ```

Every `Selection` field is a `FeatureMeta` field name. An empty tuple means no restriction on that field.

Two things `Design` must get right that the upstream loader handled implicitly:

- **Join on the composite key.** `RowId = tuple[str, int, str]` is `(recording, epoch, event)`
  (`table.py:18`). An epoch index is unique only within a recording, so `targets` must carry
  all three and the join must be validated one-to-one. Joining on an epoch index alone
  mismatches rows across subjects without raising.
- **Keep covariate identity.** Upstream's convention is `n_covariates: int` — covariates are
  the last *n* columns — which is why `_variance_param_prefix` recomputes positions in
  `estimators.py`. Explicit `feature_columns` and `covariate_columns` index arrays make the
  leakage check and the nuisance design directly inspectable. `Design.n_covariates` remains
  available as `covariate_columns.size` for the estimator builders that still want a count.

`runs` is carried because Task 11 and Task 13 need it as the inner grouping for
within-subject folds.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

import pandas as pd

from eegfeat.model.design import Selection, build_design, select
from eegfeat.table import FeatureMeta


def test_every_selection_field_is_a_real_feature_meta_field() -> None:
    # Selection keys are metadata field names rather than name fragments, so a measure
    # that gets renamed upstream fails loudly here instead of selecting nothing.
    assert set(Selection.__dataclass_fields__) <= set(FeatureMeta.__dataclass_fields__)


def test_selecting_a_band_keeps_only_that_band(alpha_beta_table) -> None:
    kept = select(alpha_beta_table, Selection(band=("alpha",)))
    assert {m.band.name for m in kept.meta} == {"alpha"}


def test_an_empty_field_places_no_restriction(alpha_beta_table) -> None:
    kept = select(alpha_beta_table, Selection())
    assert kept.values.shape == alpha_beta_table.values.shape


def test_a_selection_that_matches_nothing_raises_rather_than_returning_empty(
    alpha_beta_table,
) -> None:
    with pytest.raises(ValueError, match="no columns"):
        select(alpha_beta_table, Selection(band=("delta",)))


def test_the_join_uses_the_whole_row_id_not_the_epoch_index(alpha_beta_table) -> None:
    # Two recordings both number their epochs from zero. Joining on the epoch index alone
    # matches sub-02's epoch 0 to sub-01's target and raises nothing.
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(alpha_beta_table, targets, target="pain")
    assert design.row_ids == (("sub-01", 0, "stim"), ("sub-02", 0, "stim"))
    np.testing.assert_array_equal(design.y, [1.0, 9.0])


def test_a_target_row_matching_two_feature_rows_is_refused(alpha_beta_table) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-01"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 2.0],
            "subject_id": ["sub-01", "sub-01"],
        }
    )
    with pytest.raises(ValueError, match="one-to-one"):
        build_design(alpha_beta_table, targets, target="pain")


def test_covariate_columns_are_identified_not_merely_counted(alpha_beta_table) -> None:
    targets = pd.DataFrame(
        {
            "recording": ["sub-01", "sub-02"],
            "epoch": [0, 0],
            "event": ["stim", "stim"],
            "pain": [1.0, 9.0],
            "age": [30.0, 40.0],
            "subject_id": ["sub-01", "sub-02"],
        }
    )
    design = build_design(alpha_beta_table, targets, target="pain", covariates=["age"])
    assert design.column_names[design.covariate_columns[0]] == "age"
    assert design.covariate_columns.size == 1
```

Add an `alpha_beta_table` fixture to `tests/model/conftest.py` building a two-band
`FeatureTable` with two rows whose `row_ids` are `("sub-01", 0, "stim")` and
`("sub-02", 0, "stim")` — two recordings both numbering epochs from zero, which is what
makes the composite-key test meaningful. Follow `tests/test_table.py` for construction.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_design.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Write `Selection`, `select`, `Design` and `build_design` new; port `harmonize_fold` and the covariate-alias guard. `build_design` aligns the table's rows to `targets` on trial index and raises when they disagree.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_design.py -v`
Expected: PASS

- [ ] **Step 5: Port the leakage constraints**

These seven are safeguard 3 and are the reason this module exists. Read each upstream and rewrite it against `build_design`:

- `load_active_matrix_blocks_target_covariate_leakage`
- `load_active_matrix_blocks_target_covariate_leakage_for_explicit_target_column`
- `load_active_matrix_raises_when_requested_covariates_are_missing_in_strict_mode`
- `load_active_matrix_raises_when_requested_covariates_are_missing_by_default`
- `load_active_matrix_drops_missing_covariates_when_not_strict`
- `load_active_matrix_preserves_canonical_trial_ids`
- `group_intersection_harmonization_rejects_empty_strict_intersection`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/design.py tests/model/test_design.py tests/model/conftest.py
git commit -m "feat(model): design matrices from feature tables, selected by metadata"
```

---

### Task 11: `tuning.py` — inner-CV tuning

**Files:**
- Create: `src/eegfeat/model/tuning.py`, `tests/model/test_tuning.py`
- Source: `cv.py` — `fit_with_warning_logging`, `grid_search_with_warning_logging`, `_raise_for_nonfinite_grid_search_scores`, `create_best_params_record`, `_fit_default_pipeline`; `orchestration.py` — `_fit_tuned_regression_estimator`, `_fit_within_subject_fold`, `_fit_estimator_with_optional_groups`, `_fit_subject_weighted_inner_cv_estimator`, `_InnerSplitData`

**Interfaces:**
- Consumes: `splits.InnerSplit`, `splits.inner_cv`, `splits.inner_cv_splits`, `execution.set_random_seeds`
- Produces:
  ```python
  @dataclass(frozen=True)
  class TunedFit:
      estimator: Pipeline
      best_params: dict[str, object]

  def tune(
      pipeline: Pipeline, grid: Mapping[str, Sequence[object]],
      X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.float64],
      inner_groups_train: npt.NDArray[np.object_],
      *, split: InnerSplit, seed: int, fold: int, n_jobs: int = 1,
  ) -> TunedFit
  def fit_untuned(pipeline: Pipeline, X: ..., y: ..., *, seed: int) -> Pipeline
  ```

`inner_groups_train` is whichever array `split.grouping` names, already sliced to the fold's
training rows — subjects for leave-one-subject-out, runs for within-subject. `tune` never
sees the subject array when grouping on runs, so it cannot pick the wrong one.

`fit_untuned` is `_fit_default_pipeline`. Upstream reaches it when a fold has **no run
structure to split on** (`orchestration.py:919`) and raises when the inner `GridSearchCV`
*fails* (`orchestration.py:912`). Keep both: an absent splitter is a property of the
design, a failed search is a fault, and safeguard 6 governs only the second.

`logger` and `fold_info` parameters upstream are for logging; `model/` does not log. Raise
with the fold in the message instead.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.splits import InnerSplit
from eegfeat.model.tuning import tune

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean", "median"]}
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)
BY_RUN = InnerSplit(grouping="run", n_splits=2)
TWO_SUBJECTS = np.array(["s1"] * 4 + ["s2"] * 4, dtype=object)
TWO_RUNS = np.array(["r1"] * 4 + ["r2"] * 4, dtype=object)
X8 = np.arange(8.0).reshape(-1, 1)
Y8 = np.arange(8.0)


def test_tuning_refuses_a_training_fold_with_one_group() -> None:
    # With one group the inner split cannot be group-disjoint, so the chosen
    # hyperparameters would be selected on data from the subject being predicted.
    with pytest.raises(ValueError, match="at least 2"):
        tune(
            PIPE, GRID, X8, Y8, np.array(["s1"] * 8, dtype=object),
            split=BY_SUBJECT, seed=0, fold=1,
        )


def test_a_within_subject_fold_tunes_on_runs_not_subjects() -> None:
    # The training block is one subject, so runs are the only split that exists. This is
    # what upstream does at orchestration.py:868, and the reason tune takes an InnerSplit
    # rather than "the groups".
    assert tune(PIPE, GRID, X8, Y8, TWO_RUNS, split=BY_RUN, seed=0, fold=1).best_params


def test_the_fold_is_named_when_tuning_fails() -> None:
    with pytest.raises(ValueError, match="fold 7"):
        tune(
            PIPE, GRID, X8, Y8, np.array(["s1"] * 8, dtype=object),
            split=BY_SUBJECT, seed=0, fold=7,
        )


def test_tuning_does_not_fall_back_to_an_untuned_fit() -> None:
    # A failed inner search must surface. Falling back to a default fit would report a
    # score for a model nobody selected, indistinguishable from a tuned one downstream.
    with pytest.raises(ValueError):
        tune(
            PIPE, {"regressor__nonexistent": [1]}, X8, Y8, TWO_SUBJECTS,
            split=BY_SUBJECT, seed=0, fold=1,
        )


class _NaNScoringRegressor(DummyRegressor):
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: object = None) -> float:
        return float("nan")


def test_non_finite_inner_scores_are_refused() -> None:
    # A grid point scoring NaN on every inner split is not the best model; selecting it
    # by argmax over NaN picks whichever the sort happened to put first.
    pipe = Pipeline([("regressor", _NaNScoringRegressor(strategy="mean"))])
    with pytest.raises(ValueError, match="finite"):
        tune(pipe, GRID, X8, Y8, TWO_SUBJECTS, split=BY_SUBJECT, seed=0, fold=1)
```

Read `cv.py:362` (`_raise_for_nonfinite_grid_search_scores`) for the exact message and reconcile the `match=` pattern.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_tuning.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the functions. Drop `logger`; raise with the fold index in the message.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_tuning.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining tuning constraints**

- `tuned_loso_regression_requires_inner_cv_groups`
- `within_subject_inner_cv_failure_raises_instead_of_default_fit`
- `grid_search_wrapper_rejects_nonfinite_test_scores`
- `fit_default_pipeline_sets_available_random_state_parameter`
- `within_subject_regression_passes_rf_param_grid_to_inner_cv_helper`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/tuning.py tests/model/test_tuning.py
git commit -m "feat(model): inner-CV tuning that surfaces its failures"
```

---

### Task 12: The equivalence harness

**Files:**
- Modify: `scripts/make_fixtures.py`
- Create: `tests/test_equivalence_model.py`, `tests/fixtures/model_reference.npz`

**Interfaces:**
- Consumes: everything built so far
- Produces: a fixture file and a test asserting the port reproduces the reference pipeline

This task exists before the riskiest module so that the harness is proven on code already known to be correct. It covers `splits` and `tuning`; Task 13 extends it to `crossfit`.

- [ ] **Step 1: Extend the fixture generator**

Add a `--model` mode to `scripts/make_fixtures.py`, following the file's existing structure and its "never run from the test suite" contract. It imports the reference pipeline directly — no BIDS tree, no patching, because these units take arrays:

```python
REFERENCE_ROOT = Path("/Users/joduq24/Desktop/EEG_fMRI_Pipeline")


def model_fixtures(rng: np.random.Generator) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REFERENCE_ROOT))
    from eeg_pipeline.analysis.machine_learning import cv

    n_subjects, n_trials, n_features = 6, 10, 4
    groups = np.repeat([f"sub-{i:04d}" for i in range(n_subjects)], n_trials).astype(object)
    X = rng.normal(size=(groups.size, n_features))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=groups.size)

    folds = cv.create_loso_folds(X, groups)
    # Fold index arrays are ragged. Flattening them with an offsets array keeps the
    # fixture free of object arrays, so it loads with allow_pickle=False as
    # tests/test_equivalence.py:43 already requires.
    train = np.concatenate([f[1] for f in folds])
    test = np.concatenate([f[2] for f in folds])
    return {
        "X": X,
        "y": y,
        "groups": groups.astype("U16"),
        "loso_train": train,
        "loso_train_offsets": np.cumsum([0] + [f[1].size for f in folds]),
        "loso_test": test,
        "loso_test_offsets": np.cumsum([0] + [f[2].size for f in folds]),
        "reference_commit": np.array(_reference_commit()),
    }
```

`_reference_commit()` runs `git -C <REFERENCE_ROOT> rev-parse HEAD` and returns the hash, so
a fixture that stops matching can be traced to a specific upstream state.

Two properties are deliberate and match the existing feature-side harness rather than
being oversights: the hard-coded `REFERENCE_ROOT` (`scripts/make_fixtures.py:20`) and the
skip when fixtures are absent (`tests/test_equivalence.py:31`). Generating fixtures needs
the reference repository checked out, which is not a CI property. Equivalence is a
developer-run check; the per-module constraint tests in every task's Step 5 are what runs
everywhere, and they are the specification. Equivalence only confirms the port agrees with
an implementation that is itself wrong in one known place (spec §3.6).

- [ ] **Step 2: Generate the fixture**

```bash
.venv/bin/python scripts/make_fixtures.py --model --out tests/fixtures
```
Expected: `tests/fixtures/model_reference.npz` written.

- [ ] **Step 3: Write the equivalence test**

`tests/test_equivalence_model.py`:

```python
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from eegfeat.model.splits import loso_folds

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "model_reference.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="model fixtures not generated")


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    return dict(np.load(FIXTURE, allow_pickle=False))


def _unflatten(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def test_loso_folds_match_the_reference_pipeline(reference: dict[str, np.ndarray]) -> None:
    train = _unflatten(reference["loso_train"], reference["loso_train_offsets"])
    test = _unflatten(reference["loso_test"], reference["loso_test_offsets"])
    folds = loso_folds(reference["groups"].astype(object))
    assert len(folds) == len(train)
    for fold, expected_train, expected_test in zip(folds, train, test, strict=True):
        np.testing.assert_array_equal(fold.train, expected_train)
        np.testing.assert_array_equal(fold.test, expected_test)
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest tests/test_equivalence_model.py -v`
Expected: PASS. A failure here means the port changed fold assignment — stop and reconcile before continuing.

- [ ] **Step 5: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add scripts/make_fixtures.py tests/test_equivalence_model.py tests/fixtures/model_reference.npz
git commit -m "test(model): equivalence harness against the reference pipeline"
```

---

### Task 13: `crossfit.py` — the outer fold loop, regression

**This is the one task that is an extraction rather than a port. Read spec §4.1 before starting.**

**Files:**
- Create: `src/eegfeat/model/crossfit.py`, `tests/model/test_crossfit.py`
- Modify: `scripts/make_fixtures.py`, `tests/test_equivalence_model.py`
- Source: `cv.py` — `nested_loso_predictions_matrix`; `orchestration.py` — `compute_baseline_predictions`, and the inlined fold loop inside `run_within_subject_regression_ml` (line 1329, 565 lines)

**Interfaces:**
- Consumes: `splits.Fold`, `splits.InnerSplit`, `tuning.tune`, `execution.run_folds`, `design.harmonize_fold`, `residualize.residualize_targets`
- Produces:
  ```python
  @dataclass(frozen=True)
  class FoldPrediction:
      fold: int
      subject: str | None
      rows: npt.NDArray[np.intp]
      y_true: npt.NDArray[np.float64]
      y_pred: npt.NDArray[np.float64]
      best_params: dict[str, object]

  def cross_fit_regression(
      folds: Sequence[Fold], X: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
      groups: npt.NDArray[np.object_], pipeline: Pipeline,
      grid: Mapping[str, Sequence[object]],
      *, inner: InnerSplit, seed: int, runs: npt.NDArray[np.object_] | None = None,
      outer_n_jobs: int = 1, harmonization: str | None = None,
      covariates: npt.NDArray[np.float64] | None = None,
      residualize_on: Sequence[str] = (),
  ) -> tuple[FoldPrediction, ...]
  ```

One private loop serves every path. The leave-one-subject-out versus within-subject
distinction is entirely in which `folds` are passed and which array `inner.grouping`
names; `cross_fit_regression` resolves that array **once, before the loop**, so the loop
body branches on neither. Task 14 adds the classification entry point over the same loop.

Two validations belong at the entry point, not inside the loop:

- `inner.grouping == "run"` requires `runs`; raise naming both if it is absent.
- Folds carrying a `subject` (within-subject) with `inner.grouping == "subject"` is
  incoherent — a one-subject training block cannot be subject-grouped. Raise here rather
  than letting `tune` fail with "at least 2 groups".

**If a conditional on fold source or task appears inside the loop body, the abstraction is
wrong — stop and split the function rather than adding the branch** (spec §10).

Per spec §5, harmonization and residualization are fitted once per **outer** training fold,
matching upstream. They never see the outer test fold, so the reported estimate is clean;
they do touch the inner-validation rows, which makes hyperparameter selection mildly
optimistic. That is a stated limitation, not an oversight — do not "fix" it here, because
it would change numbers and leave the equivalence net.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_regression
from eegfeat.model.splits import InnerSplit, loso_folds, within_subject_folds

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean", "median"]}
GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 6).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2"], 3), 4).astype(object)
X = np.arange(GROUPS.size, dtype=float).reshape(-1, 1)
Y = np.arange(GROUPS.size, dtype=float)
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)
BY_RUN = InnerSplit(grouping="run", n_splits=2)


def test_every_trial_is_predicted_exactly_once_across_folds() -> None:
    predictions = cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0
    )
    rows = np.concatenate([p.rows for p in predictions])
    np.testing.assert_array_equal(np.sort(rows), np.arange(GROUPS.size))


def test_no_fold_is_fitted_on_the_subject_it_predicts() -> None:
    for prediction in cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0
    ):
        assert len(set(GROUPS[prediction.rows])) == 1


def test_the_same_loop_serves_within_subject_folds_grouped_on_runs() -> None:
    # The only things that change between designs are the folds and the inner grouping.
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2, seed=0)
    predictions = cross_fit_regression(
        folds, X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0, runs=RUNS
    )
    assert all(p.subject is not None for p in predictions)


def test_within_subject_folds_cannot_be_grouped_by_subject() -> None:
    # Caught at the entry point, where the message can name the real problem, rather than
    # surfacing as "at least 2 groups" from inside tuning.
    folds = within_subject_folds(GROUPS, RUNS, inner_splits=2, seed=0)
    with pytest.raises(ValueError, match="within-subject"):
        cross_fit_regression(folds, X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0)


def test_run_grouping_without_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="runs"):
        cross_fit_regression(loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_RUN, seed=0)


def test_a_failing_fold_raises_rather_than_being_dropped() -> None:
    # Safeguard 6: a fold that cannot be fitted is a fault, not a measurement. Dropping it
    # would silently compute the cohort result on a subset nobody chose.
    with pytest.raises(ValueError):
        cross_fit_regression(
            loso_folds(GROUPS), X, Y, GROUPS, PIPE,
            {"regressor__nonexistent": [1]}, inner=BY_SUBJECT, seed=0,
        )


def test_results_are_ordered_by_fold_not_by_completion() -> None:
    predictions = cross_fit_regression(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=BY_SUBJECT, seed=0, outer_n_jobs=2
    )
    assert [p.fold for p in predictions] == sorted(p.fold for p in predictions)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_crossfit.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Write the private loop and the regression wrapper. Per fold: `harmonize_fold`, then
`residualize_targets` when covariates are present, then `tune`, then predict. Read
`nested_loso_predictions_matrix` (`cv.py:995`) for the leave-one-out ordering and
`run_within_subject_regression_ml` (`orchestration.py:1329`, the loop at its line 142) for
the within-subject specifics, and satisfy both with the one loop.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_crossfit.py -v`
Expected: PASS

- [ ] **Step 5: Extend the equivalence harness to cover this module**

This is the verification that matters. Add to `scripts/make_fixtures.py`:

- **Leave-one-out**: call `cv.nested_loso_predictions_matrix` on the synthetic arrays from
  Task 12 and store its predictions.
- **Within-subject**: drive the upstream shell with the idiom its own tests use, capturing
  predictions from the written outputs:

```python
from unittest.mock import patch

from eeg_pipeline.analysis.machine_learning import orchestration as orch
from tests.utils.pipelines_test_utils import DotConfig

with (
    patch.object(orch, "load_active_matrix", return_value=(X, y, groups, names, meta)),
    patch.object(orch, "create_within_subject_folds", return_value=folds),
):
    out_dir = orch.run_within_subject_regression_ml(..., results_root=tmp, config=DotConfig(...))
```

Store predictions as plain float arrays with `allow_pickle=False`. Then assert in
`tests/test_equivalence_model.py` that `cross_fit_regression` reproduces both vectors.

- [ ] **Step 6: Port the remaining regression cross-fit constraints**

- `within_subject_regression_uses_fold_baseline_and_effective_subjects`
- `within_subject_regression_applies_fold_feature_harmonization`
- `cv_hygiene_surfaces_fold_context_failures`

- [ ] **Step 7: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/crossfit.py tests/model/test_crossfit.py scripts/make_fixtures.py tests/test_equivalence_model.py tests/fixtures/model_reference.npz
git commit -m "feat(model): one cross-fitting loop, regression entry point"
```

---

### Task 14: `crossfit.py` — the classification entry point

**Files:**
- Modify: `src/eegfeat/model/crossfit.py`, `tests/model/test_crossfit.py`
- Create: `tests/model/test_crossfit_classification.py`
- Source: `classification.py` — `decode_binary_outcome`, `nested_loso_classification`; `orchestration.py` — the inlined fold loop inside `run_within_subject_classification_ml` (line 2532, 666 lines)

**Interfaces:**
- Consumes: everything Task 13 consumes, plus `metrics.ClassificationResult`
- Produces:
  ```python
  @dataclass(frozen=True)
  class FoldClassification:
      fold: int
      subject: str | None
      rows: npt.NDArray[np.intp]
      y_true: npt.NDArray[np.intp]
      y_pred: npt.NDArray[np.intp]
      y_prob: npt.NDArray[np.float64] | None
      classes: tuple[int, ...]
      best_params: dict[str, object]

  def cross_fit_classification(
      folds: Sequence[Fold], X: npt.NDArray[np.float64], y: npt.NDArray[np.intp],
      groups: npt.NDArray[np.object_], pipeline: Pipeline,
      grid: Mapping[str, Sequence[object]],
      *, inner: InnerSplit, seed: int, runs: npt.NDArray[np.object_] | None = None,
      outer_n_jobs: int = 1, harmonization: str | None = None,
  ) -> tuple[FoldClassification, ...]
  ```

A separate entry point rather than a `predict_proba: bool` on Task 13's, because
classification is not regression with a flag flipped. It differs in four ways the type
system should carry: integer labels rather than floats, a **stratified** inner splitter,
a one-class training fold being an error, and a probability matrix whose column order must
be pinned to `classes` rather than assumed.

`classes` is recorded per fold and taken from the fitted estimator, not from
`np.unique(y)`. `y_prob` column *j* is the probability of `classes[j]`; that mapping is
what makes the stored probabilities interpretable when a fold's training block happens to
order its labels differently.

`y_prob` is `None` when the estimator has no `predict_proba`. It is never silently
replaced with a decision function — those are not probabilities and metrics that expect
one would be wrong.

The private loop from Task 13 is unchanged and shared. `InnerSplit(stratified=True)` is
what routes `inner_cv` to `StratifiedGroupKFold`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from eegfeat.model.crossfit import cross_fit_classification
from eegfeat.model.splits import InnerSplit, loso_folds

GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 6).astype(object)
Y = np.tile([0, 1], GROUPS.size // 2).astype(np.intp)
X = np.random.default_rng(0).normal(size=(GROUPS.size, 3)) + Y[:, None]
PIPE = Pipeline([("classifier", LogisticRegression(max_iter=500))])
GRID = {"classifier__C": [0.1, 1.0]}
STRATIFIED = InnerSplit(grouping="subject", stratified=True, n_splits=2)


def test_probability_columns_are_pinned_to_the_recorded_classes() -> None:
    # Column order comes from the fitted estimator, not from np.unique(y). A fold whose
    # training block orders its labels differently would otherwise have its columns
    # silently transposed, inverting every AUC computed from them.
    for prediction in cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
    ):
        assert prediction.y_prob is not None
        assert prediction.y_prob.shape[1] == len(prediction.classes)
        np.testing.assert_allclose(prediction.y_prob.sum(axis=1), 1.0)


def test_a_single_class_training_fold_raises() -> None:
    # Nothing can be learned from one class, and a classifier fitted on it predicts that
    # class everywhere — which scores as chance rather than as the failure it is.
    y_degenerate = np.where(GROUPS == "s1", 1, 0).astype(np.intp)
    with pytest.raises(ValueError, match="one class"):
        cross_fit_classification(
            [f for f in loso_folds(GROUPS) if set(GROUPS[f.test]) != {"s1"}][:1],
            X, y_degenerate, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0,
        )


def test_an_estimator_without_predict_proba_reports_none_not_a_decision_function() -> None:
    pipe = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    predictions = cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, pipe, {}, inner=STRATIFIED, seed=0
    )
    assert all(p.y_prob is None for p in predictions)


def test_labels_stay_integers_through_the_fold_loop() -> None:
    for prediction in cross_fit_classification(
        loso_folds(GROUPS), X, Y, GROUPS, PIPE, GRID, inner=STRATIFIED, seed=0
    ):
        assert prediction.y_pred.dtype == np.intp
```

Construct the degenerate-fold case so the *training* block is one class; read
`nested_loso_classification` (`classification.py:668`) for where upstream raises, and
reconcile the `match=` pattern to its message.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_crossfit_classification.py -v`
Expected: FAIL — `cross_fit_classification` not defined

- [ ] **Step 3: Implement**

Add the wrapper over Task 13's private loop. The loop itself does not change; if it needs
to, the abstraction was wrong and the two paths split.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_crossfit_classification.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining classification constraints**

- `nested_loso_classification_raises_when_training_fold_has_one_class`
- `nested_loso_classification_requires_inner_cv_groups_for_tuning`
- `nested_loso_classification_requires_stratified_inner_cv_support`
- `run_within_subject_classification_raises_when_training_fold_has_one_class`
- `run_within_subject_classification_raises_when_inner_cv_fails`
- `within_subject_classification_applies_fold_feature_harmonization`
- `within_subject_classification_exports_probabilities_when_available`
- `decode_binary_outcome_requires_groups_for_numeric_cv`
- `decode_binary_outcome_uses_stratified_group_kfold_for_grouped_numeric_cv`

- [ ] **Step 6: Extend the equivalence harness**

Add a classification fixture from `cv.nested_loso_classification` on synthetic arrays, and
assert `cross_fit_classification` reproduces its predictions and probabilities.

- [ ] **Step 7: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/crossfit.py tests/model/test_crossfit_classification.py scripts/make_fixtures.py tests/test_equivalence_model.py tests/fixtures/model_reference.npz
git commit -m "feat(model): classification entry point over the shared fold loop"
```

---

### Task 15: `nulls.py` — permutations

**Files:**
- Create: `src/eegfeat/model/nulls.py`, `tests/model/test_nulls.py`
- Source: `circular_shift.py` — all; `cv.py` — `permutation_changed_fraction`, `is_effective_permutation`, `run_permutation_test`; `orchestration.py` — `_resolve_permutation_scheme`, `_validate_permutation_runs`, `_validate_permutation_trial_indices`, `_circular_shift_group`, `_trial_index_ordered_indices`, `_permutation_indices_by_scheme`, `_permute_labels_by_scheme`, `_generate_effective_permutation`, `filter_circular_shift_permutation_rows`, `_run_classification_permutations`

**Interfaces:**
- Consumes: `crossfit.cross_fit`, `splits.Fold`
- Produces:
  ```python
  Scheme = Literal["within_subject", "run_wise", "circular_shift_within_run"]

  @dataclass(frozen=True)
  class NullConfig:
      scheme: Scheme = "within_subject"
      n_permutations: int = 1000
      min_complete_fraction: float = 0.9
      min_retained_trials: int = 8

  @dataclass(frozen=True)
  class NullResult:
      p_value: float
      observed: float
      null: npt.NDArray[np.float64]
      changed_fractions: npt.NDArray[np.float64]
      n_incomplete: int

  def circular_shift_group(n_retained: int) -> tuple[int, ...]
  def changed_fraction(
      y_original: npt.NDArray[np.float64], y_permuted: npt.NDArray[np.float64]
  ) -> float
  def permute(
      y: npt.NDArray[np.float64], groups: npt.NDArray[np.object_],
      runs: npt.NDArray[np.object_] | None, *, config: NullConfig, rng: np.random.Generator,
  ) -> npt.NDArray[np.float64]
  def permutation_test(
      folds: Sequence[Fold], X: ..., y: ..., groups: ..., runs: ...,
      pipeline: Pipeline, grid: Mapping[str, Sequence[object]],
      observed: float, *, config: NullConfig, inner: InnerSplit, seed: int,
  ) -> NullResult
  ```

`permutation_test` calls `cross_fit_regression` per permutation rather than carrying its
own loop.

**This module makes the one deliberate correction in the migration** (spec §3.6). Upstream
computes a permutation, measures how many labels it changed, and `continue`s past it when
that fraction is too low (`cv.py:1238`) — so the null is conditioned on the permuted
values, and the p-value's denominator is the surviving count. `circular_shift.py`'s own
docstring argues that exactly this kind of subsetting destroys the randomization argument.

So: **every sampled permutation enters the null.** The changed fraction is recorded in
`NullResult.changed_fractions` and reported, never acted on. `min_changed_fraction` is
gone from `NullConfig`.

One guard survives in a different form. Upstream raised when `n_effective == 0`; here,
`permutation_test` raises when *no* permutation changed anything, because a scheme under
which the labels are invariant cannot test the hypothesis. That is a property of the
design — run-wise permutation with one run per subject, say — not a property of an
individual draw.

`min_complete_fraction` is untouched. It governs folds that **failed to fit**, which are
faults rather than measurements, and dropping those is safeguard 6 working correctly.

Upstream's `is_effective_permutation` does **not** cross. Exporting a predicate whose only
purpose was to decide which permutations to discard would invite the filter straight back;
`changed_fraction` crosses as the measurement, and `NullResult.changed_fractions` is where
it is reported.

The p-value is `(C + 1) / (B + 1)` with `C` the count at least as extreme as `observed`
and `B` the number of permutations — matching `orchestration.py:4292`, where `B` is now
the number sampled rather than the number surviving a filter.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.nulls import NullConfig, changed_fraction, circular_shift_group, permutation_test
from eegfeat.model.splits import InnerSplit, loso_folds

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
GRID = {"regressor__strategy": ["mean"]}
GROUPS = np.repeat(["s1", "s2", "s3", "s4"], 6).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2"], 3), 4).astype(object)
X = np.arange(GROUPS.size, dtype=float).reshape(-1, 1)
Y = np.arange(GROUPS.size, dtype=float)
FOLDS = loso_folds(GROUPS)
BY_SUBJECT = InnerSplit(grouping="subject", n_splits=2)


def test_the_shift_set_is_the_whole_cycle_including_the_identity() -> None:
    # The upper-tail p-value is justified by the shifts forming a group under
    # composition, with the observed statistic as the identity. Any strict subset
    # generally is not closed: composing shift 5 with itself on an 11-trial run gives
    # shift 10, so dropping 10 loses the randomization argument and its calibration.
    assert circular_shift_group(11) == tuple(range(11))


def test_an_empty_run_yields_no_shifts() -> None:
    assert circular_shift_group(0) == ()


def test_a_permutation_that_changed_little_still_enters_the_null() -> None:
    # The admissible transformations are fixed by the design before the permuted values
    # are looked at. Dropping a draw for changing too little conditions the null on the
    # thing it is meant to hold fixed, and the upper-tail p-value loses its warrant.
    result = permutation_test(
        FOLDS, X, Y, GROUPS, RUNS, PIPE, GRID, observed=0.0,
        config=NullConfig(n_permutations=20), inner=BY_SUBJECT, seed=0,
    )
    assert result.null.size == 20


def test_the_changed_fraction_is_reported_not_acted_on() -> None:
    result = permutation_test(
        FOLDS, X, Y, GROUPS, RUNS, PIPE, GRID, observed=0.0,
        config=NullConfig(n_permutations=20), inner=BY_SUBJECT, seed=0,
    )
    assert result.changed_fractions.size == result.null.size


def test_a_scheme_under_which_nothing_ever_changes_is_refused() -> None:
    # Run-wise permutation with one run per subject permutes nothing, ever. That is a
    # property of the design, not of a draw, and it cannot test the hypothesis.
    one_run = np.full(GROUPS.size, "r1", dtype=object)
    with pytest.raises(ValueError, match="never changes"):
        permutation_test(
            FOLDS, X, Y, GROUPS, one_run, PIPE, GRID, observed=0.0,
            config=NullConfig(scheme="run_wise", n_permutations=10),
            inner=BY_SUBJECT, seed=0,
        )


def test_the_p_value_counts_the_observed_statistic_in_both_terms() -> None:
    # (C + 1) / (B + 1): the observed statistic is the identity element of the
    # transformation group, so it belongs in the numerator and the denominator alike.
    result = permutation_test(
        FOLDS, X, Y, GROUPS, RUNS, PIPE, GRID, observed=np.inf,
        config=NullConfig(n_permutations=19), inner=BY_SUBJECT, seed=0,
    )
    assert result.p_value == pytest.approx(1.0 / 20.0)


def test_changed_fraction_refuses_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        changed_fraction(np.arange(4.0), np.arange(5.0))


def test_an_unknown_scheme_is_refused_rather_than_downgraded() -> None:
    # A typo'd scheme that silently fell back to within-subject would produce a null
    # that answers a different question than the one the analysis claims to ask.
    with pytest.raises(ValueError, match="scheme"):
        NullConfig(scheme="shuffle_everything")  # type: ignore[arg-type]


def test_null_config_has_no_changed_fraction_filter() -> None:
    # Guards the correction in spec 3.6 against being reintroduced by a later port.
    assert "min_changed_fraction" not in NullConfig.__dataclass_fields__
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_nulls.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port everything **except the skip**. `circular_shift.py` comes over whole, docstring
included — its group-under-composition argument is load-bearing, the first test pins it,
and it is the reason the filter goes.

`_generate_effective_permutation` becomes `permute`, returning the permutation and its
changed fraction with no effectiveness verdict attached. The `continue` at `cv.py:1238`
has no counterpart here. `NullConfig.__post_init__` validates the scheme against `Scheme`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_nulls.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining null constraints**

These are safeguard 5, the largest ported block. **One upstream test is deliberately not
ported**: `regression_permutation_requires_effective_label_shuffling` encodes the filter
this module removes. Its replacements are `test_a_permutation_that_changed_little_still_enters_the_null`
and `test_a_scheme_under_which_nothing_ever_changes_is_refused` above.

The completion-threshold tests below concern *failed fits*, not changed labels, and port
unchanged:

- `resolve_permutation_scheme_rejects_invalid_value`
- `resolve_permutation_scheme_defaults_to_within_subject`
- `generate_effective_permutation_requires_runs_for_runwise_scheme`
- `generate_effective_permutation_rejects_run_length_mismatch`
- `generate_effective_permutation_rejects_all_missing_runs`
- `generate_effective_permutation_does_not_downgrade_runwise_scheme`
- `run_permutation_test_rejects_all_missing_runs_for_runwise_scheme`
- `run_permutation_test_surfaces_permutation_scheme_config_errors`
- `regression_permutation_test_handles_nonfinite_predictions`
- `regression_permutation_completion_threshold_uses_config`
- `within_subject_regression_permutation_completion_threshold_uses_config`
- `within_subject_regression_skips_partial_fold_permutations`
- `within_subject_classification_permutation_failures_do_not_abort_inference`
- `group_classification_permutations_enforce_failed_fold_fraction`
- `group_classification_permutation_failures_do_not_abort_inference`

Also port `tests/machine_learning/test_circular_shift_group.py` and `test_permutation_complete_folds.py` whole.

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/nulls.py tests/model/test_nulls.py
git commit -m "feat(model): permutation nulls that refuse to answer a different question"
```

---

### Task 16: `uncertainty.py` — conformal intervals

**Files:**
- Create: `src/eegfeat/model/uncertainty.py`, `tests/model/test_uncertainty.py`
- Source: `uncertainty.py` — all

**Interfaces:**
- Consumes: `splits.Fold`
- Produces:
  ```python
  Method = Literal["split", "cv_plus", "quantile"]

  @dataclass(frozen=True)
  class PredictionIntervals:
      lower: npt.NDArray[np.float64]
      upper: npt.NDArray[np.float64]
      alpha: float
      method: Method
      calibration_unit: Literal["trial", "subject"]

  def prediction_intervals(
      model: Pipeline,
      X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.float64],
      X_test: npt.NDArray[np.float64],
      *, alpha: float = 0.1, method: Method = "cv_plus", cv_splits: int = 5,
      seed: int = 42, groups: npt.NDArray[np.object_] | None = None,
  ) -> PredictionIntervals
  ```

All three methods rest on exchangeability, and trials nested within a participant are not
exchangeable with trials from a participant the model has never seen. So the interval is
**marginal over the calibration unit** and nothing more (spec §3.9):

- `groups=None` calibrates on pooled trials and gives marginal coverage over trials. It
  does **not** promise 90% coverage within each subject, nor for a new subject.
- `groups=<subjects>` makes the calibration split subject-disjoint, which is the right
  choice when the estimand is a new subject.

`calibration_unit` records which was used so a downstream reader cannot mistake one for the
other. The docstring states the limitation; neither it nor the API claims
subject-conditional coverage, because none of the three methods delivers it without
adaptation.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.uncertainty import prediction_intervals

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])


def test_intervals_bracket_their_point_prediction() -> None:
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    result = prediction_intervals(PIPE, X, y, X[:5], alpha=0.1, cv_splits=4)
    assert np.all(result.lower <= result.upper)


def test_a_smaller_alpha_gives_wider_intervals() -> None:
    # Ninety-five percent coverage cannot be narrower than ninety percent on the same
    # residuals; if it is, the quantile is being taken from the wrong tail.
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    wide = prediction_intervals(PIPE, X, y, X[:5], alpha=0.05, cv_splits=4)
    narrow = prediction_intervals(PIPE, X, y, X[:5], alpha=0.10, cv_splits=4)
    assert np.all((wide.upper - wide.lower) >= (narrow.upper - narrow.lower))


def test_the_calibration_unit_is_recorded_so_coverage_cannot_be_over_read() -> None:
    # A trial-calibrated interval and a subject-calibrated one make different promises.
    # Storing which was used keeps a reader from assuming the stronger of the two.
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10).astype(object)
    assert prediction_intervals(PIPE, X, y, X[:5], cv_splits=4).calibration_unit == "trial"
    assert (
        prediction_intervals(PIPE, X, y, X[:5], cv_splits=4, groups=groups).calibration_unit
        == "subject"
    )


def test_no_calibration_data_raises_rather_than_returning_an_infinite_interval() -> None:
    with pytest.raises(ValueError):
        prediction_intervals(
            PIPE, np.arange(2.0).reshape(-1, 1), np.arange(2.0),
            np.zeros((1, 1)), alpha=0.1, cv_splits=5,
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_uncertainty.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port all three conformal methods and the quantile helpers.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_uncertainty.py -v`
Expected: PASS

- [ ] **Step 5: Port the remaining uncertainty constraints**

- `split_conformal_handles_small_training_sets`
- `cv_plus_uses_fold_specific_prediction_bands`
- `cv_plus_raises_when_no_calibration_chunks_survive`
- `uncertainty_requires_minimum_valid_fold_fraction`
- `uncertainty_exports_fold_and_subject_provenance`

- [ ] **Step 6: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model/uncertainty.py tests/model/test_uncertainty.py
git commit -m "feat(model): conformal prediction intervals"
```

---

### Task 17: `importance.py` — SHAP and permutation importance

**Files:**
- Create: `src/eegfeat/model/importance.py`, `tests/model/test_importance.py`
- Source: `shap_importance.py` — all; `feature_metadata.py` — `aggregate_importance`; `orchestration.py` — the computation inside `_run_permutation_importance_stage`

**Interfaces:**
- Consumes: `splits.Fold`, `crossfit.cross_fit`, `eegfeat.table.FeatureMeta`
- Produces:
  ```python
  @dataclass(frozen=True)
  class Importance:
      feature_names: tuple[str, ...]
      values: npt.NDArray[np.float64]
      per_fold: npt.NDArray[np.float64]

  def shap_importance(
      model: Pipeline, X: npt.NDArray[np.float64], feature_names: Sequence[str], *, seed: int = 42
  ) -> Importance
  def shap_importance_over_folds(...) -> Importance
  def permutation_importance(
      model: Pipeline, X: ..., y: ..., *, n_repeats: int = 10, seed: int = 42
  ) -> Importance
  def aggregate_by(
      importance: Importance, meta: Sequence[FeatureMeta], field: str
  ) -> dict[str, float]
  ```

`aggregate_by` replaces `feature_metadata.aggregate_importance`. It groups by a `FeatureMeta` field rather than by parsing column names — the rest of `feature_metadata.py` does not cross, because `FeatureMeta` already holds what it reconstructed.

shap is optional. Guard it as upstream's `_check_shap_available` does, and mark **only the
SHAP tests** with `@needs_shap`. A module-scope `pytest.importorskip("shap")` would also
skip `permutation_importance` and `aggregate_by`, neither of which touches shap — the
same shape of mistake as `tests/test_microstates.py` avoids for scikit-learn.

Permutation importance is computed on **held-out** rows, not on the training block. The
question is what the model generalizes on; training-set importance answers a different one
and is systematically inflated for high-cardinality features.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from eegfeat.model.importance import Importance, aggregate_by, permutation_importance

PIPE = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
needs_shap = pytest.mark.skipif(
    importlib.util.find_spec("shap") is None, reason="shap not installed"
)


def test_permutation_importance_names_every_feature_it_scores() -> None:
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = X[:, 0] * 2.0
    result = permutation_importance(PIPE.fit(X, y), X, y, n_repeats=3, seed=0)
    assert len(result.feature_names) == result.values.size == 3


def test_importance_aggregates_by_a_metadata_field_not_a_name_fragment(alpha_beta_meta) -> None:
    # Grouping on FeatureMeta.band means a column renamed upstream stops matching loudly
    # rather than quietly falling out of its band's total.
    importance = Importance(
        feature_names=("a", "b"), values=np.array([1.0, 3.0]), per_fold=np.empty((0, 2))
    )
    assert aggregate_by(importance, alpha_beta_meta, "band") == {"alpha": 1.0, "beta": 3.0}
```

Add the `alpha_beta_meta` fixture to `tests/model/conftest.py` beside `alpha_beta_table` from Task 10: two `FeatureMeta` records differing only in band (alpha, beta).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_importance.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Port the SHAP functions and write `permutation_importance` and `aggregate_by`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_importance.py -v`
Expected: PASS. Then confirm the skip is scoped correctly — with shap absent, the two
tests above must still run:

Run: `.venv/bin/python -m pytest tests/model/test_importance.py -v -p no:cacheprovider`
Expected: no test in this file is skipped unless its name mentions shap.

- [ ] **Step 5: Port the remaining importance constraints**

- `shap_kernel_uses_estimator_predict_fn_for_transformed_features`
- `cv_shap_raises_when_inner_grid_search_fails`
- `cv_shap_raises_when_fold_explanation_fails`
- `shap_stage_requires_min_valid_fold_fraction`
- `permutation_importance_stage_requires_min_valid_fold_fraction`

- [ ] **Step 6: Export the public surface**

Fill `src/eegfeat/model/__init__.py`'s `__all__` with the names a user calls directly — `Fold`, `Design`, `Selection`, `build_design`, `cross_fit`, `loso_folds`, `within_subject_folds`, `permutation_test`, `prediction_intervals`, `shap_importance`, `regression_metrics`, `ClassificationResult`, and the config dataclasses. Add `tests/model/test_public_api.py` asserting every name in `__all__` is importable, following `tests/test_public_api.py`.

- [ ] **Step 7: Verify green and commit**

```bash
.venv/bin/python -m pytest && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
git add src/eegfeat/model tests/model
git commit -m "feat(model): SHAP and permutation importance, aggregated by metadata"
```

---

## Verification at the end

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy
.venv/bin/python -m ruff check .
wc -l src/eegfeat/model/*.py | sort -rn | head -3     # none over 500
.venv/bin/python -c "import eegfeat"                   # must not import sklearn
```

The last check matters: `import eegfeat` must still work, and still cost nothing beyond numpy, scipy, pandas and mne.

## What this plan does not deliver

The recipe runner. After this lands you can fit models from Python; `eegfeat fit model.toml` comes in the second plan, along with `provenance.py` and the output files in spec §7.

Spec §11's bridge is also still open: `model/design.py` reads `FeatureTable`s, so pointing this at the existing study's pipeline-format TSVs needs either a cohort-wide feature recomputation through `eegfeat` or a transitional reader. That decision is not blocking — every task above is testable on synthetic tables — but it determines when the subpackage is usable on real data.
