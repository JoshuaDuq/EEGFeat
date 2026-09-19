# The `eegfeat.model` subpackage

Migrating the machine-learning methodology out of EEG_fMRI_Pipeline.

| | |
|---|---|
| Date | 2026-09-19 |
| Status | Approved, not yet implemented |
| Source | `EEG_fMRI_Pipeline/eeg_pipeline/analysis/machine_learning/` |
| Destination | `eegfeat/src/eegfeat/model/`, `eegfeat/src/eegfeat/runner/` |

---

## 1. Goal

`eegfeat` computes features. It should also fit and evaluate models on them, so that one
repository carries a study from preprocessed epochs to a defensible result.

The methodology to do this already exists in EEG_fMRI_Pipeline: roughly 12,000 lines
across fifteen modules, guarded by 123 validity tests that encode years of accumulated
corrections. That methodology is repository-independent and worth keeping. The wiring
around it — BIDS derivative trees, subject discovery, an untyped configuration object
threaded through every function — is not.

### The seam already exists

The strongest evidence that this is the right boundary is that the pipeline's own code
already draws it for the leave-one-subject-out path. Every unit that crosses is already
callable on plain arrays:

```
nested_loso_predictions_matrix(X, y, groups, pipe, param_grid, ...)
run_permutation_test(X, y, groups, blocks, pipe, ...)
nested_loso_classification(X, y, groups, model=..., ...)
decode_binary_outcome(X, y, cv=..., groups=..., ...)
compute_prediction_intervals(model, X_train, y_train, X_test, ...)
```

Only the four `run_*_ml` shells require `subjects`, `task`, `deriv_root` and `config`,
and those are exactly what does not cross. This migration is not inventing a boundary;
it is removing the wrapper from one that is already there.

### Why this is a re-founding, not a copy

`orchestration.py` is 5,668 lines, seven and a half times the largest module in
`eegfeat`. The cause is measurable: **83 configuration lookups in that file alone, and 56
functions across the machine-learning modules that take an untyped `config: Any`.**

`eegfeat` runs `mypy --strict`. `config: Any` does not survive contact with it. Parsing a
typed recipe once at the boundary and passing dataclasses downward is what collapses the
volume, and strict mode enforces that rather than leaving it to discipline.

A second symptom points the same way. Many of the 123 validity tests reach through the
underscore — `orch._fit_tuned_regression_estimator`, `orch._resolve_permutation_scheme`.
When a module is 5,668 lines its real units are not reachable from the public surface, so
tests trespass to get at them. Splitting the module makes those units public.

### Non-goals

EEG_fMRI_Pipeline is not modified. Nothing deleted, no imports repointed, no behaviour
changed. Whether the pipeline should later import `eegfeat` is a separate decision that
waits on the equivalence fixtures; see section 11.

---

## 2. Scope

### Crosses

Cross-validation construction, estimator pipelines, fold-local preprocessing and target
residualization, permutation and circular-shift nulls, subject-level metrics, nested
cross-validated fitting for regression and classification, SHAP and permutation
importance, and conformal prediction intervals.

### Stays behind

| | Reason |
|---|---|
| `cnn.py` | Adds a torch dependency; least reusable |
| `time_generalization.py` | Deferred with the CNN |
| `run_model_comparison_ml`, `run_incremental_validity_ml` | Study-specific analysis modes |
| `plotting.py` | Tied to the pipeline's output conventions |
| `load_active_matrix` and the loading layer | BIDS trees, subject discovery, events |
| `config.py`, `get_ml_config` | Replaced by typed recipe dataclasses |
| `cv_hygiene.py` (fold-local IAF) | Deferred; see section 3.6 |

---

## 3. Architecture

### 3.1 The seam

The pipeline keeps everything behind `load_active_matrix`. `eegfeat` never learns what a
subject directory is. Its entry point is a `FeatureTable` plus a target and a groups
vector, aligned on trial index.

### 3.2 Layout and size budget

**No module exceeds 500 lines.** If one would, it splits. This is the checkable form of
"clean", and it is the constraint that prevents a second `orchestration.py`. For
reference, the largest module in `eegfeat` today is `runner/recipe.py` at 753.

```
src/eegfeat/model/
  __init__.py
  design.py         FeatureTable(s) + target + groups -> design matrix    ~300
  splits.py         LOSO, within-subject, run-aware, inner CV             ~250
  transformers.py   missingness, variance, inf->nan, deconfound, spatial  ~450
  estimators.py     pipelines and grids, regression and classification    ~500
  residualize.py    fold-local nuisance regression on targets             ~350
  nulls.py          permutation schemes, circular shift, effectiveness    ~400
  scoring.py        scorers and scoring dictionaries                      ~100
  aggregate.py      subject-level aggregation, bootstrap CI, sign-flip    ~300
  metrics.py        metric computation, ClassificationResult              ~350
  tuning.py         inner-CV tuning of one estimator on one fold          ~250
  crossfit.py       the outer fold loop                                   ~250
  execution.py      seeds and parallelism                                 ~120
  importance.py     SHAP and permutation importance                       ~450
  uncertainty.py    conformal prediction intervals                        ~400

src/eegfeat/runner/
  fit_recipe.py     parses a model recipe                                 ~400
  fit.py            executes one                                          ~350
  provenance.py     hashes, seeds, versions, git state                    ~200
```

Fourteen modules in the subpackage is consistent with the house style; the feature half
is twenty-two flat modules.

### 3.3 The boundary invariant

**No module under `model/` imports from `runner/`, reads a TOML file, or writes a file.**
Science takes arrays and returns values; orchestration reads recipes and writes results.

This is the boundary `runner/compute.py` already holds against the feature modules. It
makes every module in `model/` unit-testable with arrays alone, and it is checkable, so
`tests/test_packaging.py` gains a test asserting it by inspecting imports.

### 3.4 Input contract

Core functions in `model/` take plain `(X, y, groups)` arrays. One adapter — `design.py`
— turns a `FeatureTable` plus target and groups into a design matrix.

The payoff is that selection becomes a structured query over `FeatureMeta` rather than
substring matching on column names. The pipeline's four string selectors map one-to-one
onto real metadata fields:

| Pipeline selector | `FeatureMeta` field |
|---|---|
| `feature_families`, `feature_stats` | `measure` |
| `feature_bands` | `band` |
| `feature_scopes` | `space_kind` |
| `feature_segments` | `window` |

A renamed measure then raises instead of silently selecting nothing.

### 3.5 Dependencies, and how `model` is imported

`scikit-learn` becomes a `model` extra and `shap` an `importance` extra.

**`eegfeat/__init__.py` does not import `eegfeat.model`.** `import eegfeat as ef` gives
features and costs numpy, scipy, pandas and mne; `from eegfeat import model` gives the
machine learning and requires scikit-learn, failing with a clear message if it is absent.

This deliberately differs from the `_require_sklearn()` lazy-import pattern in
`microstates.py`, and the difference is justified: microstates uses scikit-learn for one
KMeans call, so a function-level guard is proportionate. Every transformer and estimator
in `model/` subclasses `BaseEstimator`. Lazy-importing inside every function would fight
the design, so the guard moves up to the subpackage boundary.

### 3.6 Fold-local IAF is deferred, deliberately

`analysis/features/cv_hygiene.py` (340 lines) estimates individual alpha frequency from
training trials only and derives band edges from it. It is a natural fit for `eegfeat`
and the clearest argument for joining the two halves: a band definition estimated on
training trials only is a feature-computation concern that exists solely inside a fold.

It is nonetheless **out of scope for this migration**, because porting it honestly would
break the verification story. `cv_hygiene._compute_aperiodic_residual` calls the
pipeline's `_robust_aperiodic_fit`; `eegfeat` already has its own aperiodic fit with
iterative positive-residual rejection. Reusing `eegfeat`'s — which is the right thing to
do — changes the IAF numbers, so this one component could not be verified by equivalence
while every other component is. Mixing a deliberate behaviour change into a migration
whose entire premise is "prove nothing changed" is how a migration loses its warrant.

`design.py` therefore accepts fold-specific band parameters when supplied, honouring
safeguard 2, while their estimation lands in a follow-up with its own verification.

---

## 4. Module mapping

Every symbol that crosses, and where it lands. Anything in `orchestration.py` not listed
here stays behind.

### `model/splits.py` — from `cv.py`

`create_loso_folds`, `create_inner_cv`, `create_run_aware_cv`,
`create_within_subject_folds`, `create_run_aware_inner_cv`, `get_inner_cv_splits`.

Gains a uniform `Fold` record carrying `train_idx`, `test_idx`, and optional `subject_id`
and `params`, so that one outer loop can consume either fold source. See section 4.1.

`get_min_channels_required` does not cross; it becomes a recipe field.

### `model/scoring.py` — from `cv.py`

`safe_pearsonr`, `make_pearsonr_scorer`, `create_scoring_dict`.

### `model/aggregate.py` — from `cv.py`, `orchestration.py`

From `cv.py`: `aggregate_fold_results`, `compute_subject_level_r`,
`compute_subject_level_errors`.

From `orchestration.py`: `_subject_mean_metric`, `_count_finite_subject_metric`,
`_subject_metric_values`, `_bootstrap_mean_ci`, `_paired_signflip_p_value`,
`_subject_weighted_r2_scores`.

`compute_subject_level_r` currently takes `config: Optional[Any]` to resolve its CI
method and subject weighting; both become typed parameters.

### `model/metrics.py` — from `cv.py`, `classification.py`, `orchestration.py`

From `cv.py`: `compute_metrics`. From `classification.py`: `ClassificationResult`.

From `orchestration.py`: `_within_subject_centered_prediction_metrics`,
`_within_condition_prediction_metrics`, `_within_condition_cells`, `_center_within_cells`.

### `model/transformers.py` — from `preprocessing.py`

`VarianceThreshold`, `MissingnessThreshold`, `ReplaceInfWithNaN`, `DropAllNaNColumns`,
`SpatialFeatureSelector`, `Deconfounder`, `build_base_preprocessing_steps`,
`transform_feature_names_through_steps`, `validate_subject_missingness`.

### `model/estimators.py` — from `pipelines.py`, `classification.py`

`create_elasticnet_pipeline`, `create_ridge_pipeline`, `create_rf_pipeline`,
`create_svm_pipeline`, `create_logistic_pipeline`, `create_rf_classification_pipeline`,
`create_ensemble_pipeline`, and the six `build_*_param_grid` functions.

Each grid function currently takes `config: Any`; each becomes a typed dataclass.

### `model/nulls.py` — from `circular_shift.py`, `cv.py`, `orchestration.py`

`circular_shift.py` merges here whole — at 43 lines its concern is nulls, and its
docstring already carries the group-under-composition argument justifying the upper-tail
p-value.

From `cv.py`: `permutation_changed_fraction`, `is_effective_permutation`,
`run_permutation_test`.

From `orchestration.py`: `_resolve_permutation_scheme`, `_validate_permutation_runs`,
`_validate_permutation_trial_indices`, `_circular_shift_group`,
`_trial_index_ordered_indices`, `_permutation_indices_by_scheme`,
`_permute_labels_by_scheme`, `_generate_effective_permutation`,
`filter_circular_shift_permutation_rows`, `_run_classification_permutations`.

The permutation loop reuses `crossfit.py` with a permuted target rather than carrying its
own copy of the outer loop.

### `model/residualize.py` — from `target_residualization.py`, `orchestration.py`

`FoldNuisanceFit`, `residualize_targets_for_fold`, `fit_nuisance_model_for_fold`,
`_design_matrix`, `_validate_indices`, `_validate_training_nuisance_rank`,
`_StagedResidualPreprocessor`, `_fit_staged_residual_preprocessor`,
`reconstruct_staged_permutation_target_for_fold`.

`configured_target_residualization_columns` does not cross; it becomes a recipe field.

### `model/design.py` — from `cv.py`, `orchestration.py`, and new

New: the `[select]` query over `FeatureMeta`, and assembly of a `FeatureTable` plus
target and groups into a design matrix.

From `cv.py`: `compute_train_group_intersection_mask`, `apply_fold_feature_harmonization`.

From `orchestration.py`: `_target_covariate_aliases`,
`_warn_or_raise_if_binary_like_regression_target`,
`_apply_fold_feature_harmonization_foldwise`.

`design.py` owns the feature axis — which columns are in `X`, globally through `[select]`
and per fold through harmonization.

### `model/execution.py` — from `cv.py`

`set_random_seeds`, `determine_inner_n_jobs`, `should_parallelize_folds`,
`execute_folds_parallel`.

### `model/tuning.py` — from `cv.py`, `orchestration.py`

From `cv.py`: `fit_with_warning_logging`, `grid_search_with_warning_logging`,
`_raise_for_nonfinite_grid_search_scores`, `create_best_params_record`.

From `orchestration.py`: `_fit_tuned_regression_estimator`, `_fit_within_subject_fold`,
`_fit_estimator_with_optional_groups`, `_fit_subject_weighted_inner_cv_estimator`,
`_InnerSplitData`.

`_fit_default_pipeline` crosses as an explicit, named path. It must never be reachable as
a silent fallback from a failed fit; see safeguard 6.

### `model/crossfit.py` — from `cv.py`, `classification.py`, `orchestration.py`

From `cv.py`: `nested_loso_predictions_matrix`. From `classification.py`:
`decode_binary_outcome`, `nested_loso_classification`. From `orchestration.py`:
`compute_baseline_predictions`.

Plus the extracted within-subject loops; see 4.1.

### 4.1 The one part that is extraction, not port

This is the highest-risk item in the migration and it should be read before anything is
scheduled.

The leave-one-subject-out path has a clean array-level core:
`nested_loso_predictions_matrix` and `nested_loso_classification`. The within-subject
path has none. `run_within_subject_regression_ml` (565 lines) and
`run_within_subject_classification_ml` (666 lines) **inline their fold loops** — the
fold iteration, the permutation re-iteration, tuning, metric assembly and result writing
all interleaved in one function. Measured against those 1,231 lines, configuration and
I/O account for only 22 and 20 lines respectively. They are not thin config wrappers
around a hidden core. The core does not exist.

So `crossfit.py` extracts it. The four paths differ on two axes only:

- **Fold source** — `create_loso_folds` versus `create_within_subject_folds`, which
  `splits.py` already owns.
- **Task** — which estimator is tuned, whether prediction is `predict` or
  `predict_proba`, and which metric set is computed afterwards.

Neither axis needs a branch inside the loop. One outer loop takes a fold iterator, a
tuner and a predictor; task-specific metrics are computed after it returns. Four inlined
loops become one parameterized loop, and the leave-one-out versus within-subject
distinction returns to `splits.py` where it belongs.

That is a genuine improvement and it is also a redesign. Redesign during migration is how
behaviour silently changes, so it carries the strictest verification burden in section
8.3, and it is scheduled last among the science modules so that everything it depends on
is already proven.

### `model/importance.py` — from `shap_importance.py`, `feature_metadata.py`, `orchestration.py`

`SHAPResult`, `compute_shap_values`, `compute_shap_importance`,
`compute_shap_for_cv_folds`, `aggregate_importance`, and the permutation-importance
computation inside `_run_permutation_importance_stage`.

`feature_metadata.py` largely dissolves: `build_feature_metadata` and
`_channel_to_roi_map` reconstruct by hand what `FeatureMeta` already holds. Only
`aggregate_importance` survives, as a group-by over metadata fields.

### `model/uncertainty.py` — from `uncertainty.py`

`PredictionIntervalResult`, `compute_prediction_intervals`, `_conformal_split`,
`_conformal_cv_plus`, `_conformalized_quantile_regression`, `_compute_conformal_quantile`,
`_order_stat_quantile`, `_get_cv_splitter`.

### `runner/fit_recipe.py` — new

Nothing maps to it. The pipeline's equivalent is 83 scattered configuration lookups; this
parses the grammar in section 6 once into typed dataclasses, reporting every problem at
once in the manner `runner/recipe.py` already establishes.

### `runner/provenance.py` — from `orchestration.py`

`_sha256_file`, `_sha256_array`, `_sha256_json`, `build_ml_input_hashes`,
`_git_metadata`, `write_reproducibility_info`, `_json_safe`, `_json_safe_scalar`,
`_normalize_subject_ids`.

Adds one link the pipeline cannot currently make: the upstream feature run's manifest
hash, so a fit records exactly which feature recipe produced its inputs.

### `runner/fit.py` — from `orchestration.py`

`export_subject_selection_report`, `export_baseline_predictions`, and the orchestration
shells of `_run_shap_importance_stage`, `_run_uncertainty_stage` and
`_run_permutation_importance_stage` — minimum-valid-fold-fraction enforcement and result
writing, with computation delegated to `model/`.

---

## 5. The safeguards contract

The README sells `eegfeat` on eliminating known failure modes. The model subpackage earns
the same section. These seven are what the 123 validity tests encode.

1. **Group-disjoint splits, always.** No subject or run appears in both train and test.
   Inner cross-validation refuses fewer than two training groups. Validation splits stay
   disjoint even when the group splitter falls back.

2. **Fold-local everything.** Feature harmonization, target residualization, imputation
   and scaling are fit on training trials only. Where fold-specific band edges are
   supplied, they are honoured as fold-local; their estimation is deferred (section 3.6).

3. **No target leakage through covariates.** Covariates that alias the target are
   blocked, by name and by explicit column.

4. **Subject-level primary metrics.** Balanced accuracy, precision, recall, F1 and r are
   computed per subject and aggregated with equal subject weight — never pooled across
   trials, and with no fallback to a pooled AUC when the subject-level computation fails.

5. **Honest nulls.** A permutation must actually change labels. The circular-shift set is
   the full group under composition, not a filtered subset. A scheme mismatch raises
   rather than silently downgrading. An incomplete permutation run fails a configured
   completion threshold instead of being averaged over.

6. **Failures surface.** A fold that fails to fit raises; it does not fall back to a
   default fit. Every stage enforces a minimum valid-fold fraction.

7. **Non-finite guards.** Grid search rejects non-finite scores. A single-class training
   fold is an error, not a silent NaN.

### A distinction to preserve in safeguard 6

Safeguard 6 governs fits, not measurements, and the port must not flatten the two.

A detector that resolves nothing has *made a measurement*: it must be reported and must
not abort a cohort run. A fold that fails to fit has *failed*: it must raise and must not
be averaged into a result. Both behaviours are correct in their own domain. The
documentation states the distinction so that neither is later "fixed" into the other.

---

## 6. The model recipe

A separate file from the feature recipe, not new sections within it. The two runs have
different shapes: a feature run is expensive and parallel over recordings, a fit is one
cohort-level operation whose parallelism is over folds and permutations. Many models are
fitted against one feature set, so welding them together forces duplication or
recomputation. `runner/recipe.py` is already the largest module in the repository at 753
lines and should not absorb a second grammar.

Command: `eegfeat fit model.toml`. Validation stays under `eegfeat check`, which
dispatches on the kind of recipe the file declares; `recipe.py:196` already rejects an
unknown section with an explicit message, so the error is clear either way.

```toml
[inputs]
root = "derivatives/eegfeat"        # a feature run's output tree
targets = "targets.tsv"             # trial index, target, groups, covariates

[select]                            # every key is a FeatureMeta field name,
measure = ["integrated_band_power", "aperiodic"]   # each taking a list of
band = ["alpha", "beta"]                           # accepted values
space_kind = ["roi"]                # channel | roi | global | pair | state
window = ["stimulus"]
normalization = ["log10"]

[target]
name = "pain_rating"
kind = "continuous"                 # or "binary"
residualize_on = ["trial_number"]   # fold-local nuisance regression

[cv]
scheme = "loso"                     # or "within_subject", "run_aware"
inner_splits = 5
seed = 42

[model]
estimator = "elasticnet"
[model.grid]
alpha = [0.01, 0.1, 1.0]
l1_ratio = [0.1, 0.5, 0.9]

[nulls]
scheme = "circular_shift"           # or "within_subject", "run_wise"
n_permutations = 1000
min_complete_fraction = 0.9

[importance]                        # optional; absent means not run
method = "shap"

[uncertainty]                       # optional; absent means not run
method = "cv_plus"
alpha = 0.1
```

One seed in `[cv]` derives everything, through the existing per-fold `set_random_seeds`.

---

## 7. Outputs

```
predictions.tsv   one row per trial: subject, run, trial, fold, y_true, y_pred
folds.tsv         fold assignments
metrics.json      subject-level and aggregate
nulls.tsv         the null distribution and its p-value
importance.tsv    written when [importance] is present
intervals.tsv     written when [uncertainty] is present
provenance.json   input hashes, seeds, package versions, git state, upstream feature run
```

`folds.tsv` is not optional. It makes group-disjointness auditable by a reader after the
fact rather than guaranteed only by a unit test at development time, and it is the
artifact a skeptical reviewer asks for. The pipeline already exports fold and trial
provenance; this preserves that.

---

## 8. Test strategy

### 8.1 What the 123 validity tests become

They are `unittest.TestCase` methods averaging 41 lines, each building its synthetic
arrays inline, with no shared fixtures. They are rewritten pytest-style against the
public API, with shared builders in the existing `tests/synthetic.py`.

Categories overlap, so the counts do not sum to 123.

| Topic | Count | Destination |
|---|---:|---|
| CV, folds, splits | 42 | `test_splits.py`, `test_crossfit.py`, `test_tuning.py` |
| Loading layer | 26 | Stays, except the leakage constraints |
| Permutations, nulls | 23 | `test_nulls.py` |
| Model comparison, incremental validity | 14 | Stays |
| Time generalization | 11 | Stays |
| Metrics | 11 | `test_metrics.py`, `test_aggregate.py` |
| Leakage, covariates | 7 | `test_design.py` |
| Uncertainty, conformal | 6 | `test_uncertainty.py` |
| SHAP | 4 | `test_importance.py` |
| CNN | 4 | Stays |

Roughly 85 to 90 tests' worth of constraints cross.

### 8.2 Equivalence fixtures for the array-callable units

Unit tests prove the new code is self-consistent. They do not prove it computes what the
old code computed. The feature half already solved this: `scripts/make_fixtures.py` runs
the reference pipeline once, writes npz fixtures, and `tests/test_equivalence.py` asserts
`eegfeat` reproduces them. It is never run from the test suite.

Because every crossing unit is array-callable (section 1), this is straightforward for
most of the migration: `make_fixtures.py` imports the pipeline's functions and calls them
directly on synthetic arrays — no BIDS tree, no patching, no mounted drive. Seeds are set
per fold, so the deterministic paths must match exactly.

### 8.3 Equivalence for the within-subject path

The four `run_*_ml` shells call `load_active_matrix` themselves and cannot be driven on
arrays. Since the within-subject fold loop exists only inside them (section 4.1), its
fixtures need the shell.

The pipeline's own tests already establish the idiom, and it is exactly what the fixture
script needs:

```python
patch.object(orch, "load_active_matrix", return_value=(X, y, groups, ["f1"], meta)),
patch.object(orch, "create_within_subject_folds", return_value=folds),
...
out_dir = orch.run_within_subject_regression_ml(...)
```

A `DotConfig` from `tests/utils/pipelines_test_utils` supplies the configuration and
`results_root` points at a temporary directory. `make_fixtures.py` captures fold
assignments, per-fold predictions, and the assembled metrics from the written outputs.

This path gets the most fixture coverage in the migration, not the least, because it is
the only part where the new code is an extraction rather than a port.

---

## 9. Sequencing

Bottom-up, test-driven. Each module's constraints are written as tests before its code.

| Tier | Modules | Depends on |
|---|---|---|
| 0 | `splits.py`, `scoring.py`, `aggregate.py`, `transformers.py` | numpy, scikit-learn |
| 1 | `metrics.py`, `estimators.py`, `residualize.py`, `execution.py` | tier 0 |
| 2 | `design.py` | `eegfeat.table` |
| 3 | `tuning.py` | tiers 0–1 |
| 4 | `crossfit.py` | tiers 0–3 |
| 5 | `nulls.py`, `uncertainty.py`, `importance.py` | `crossfit.py` |
| 6 | `fit_recipe.py`, `runner/fit.py`, `provenance.py`, CLI | tiers 2–5 |

`crossfit.py` is deliberately late. It is the only extraction rather than port, so
everything it composes is proven before it is written.

**Gate at every module:** the full suite, `mypy --strict`, `ruff`.
**Gate at the end:** the equivalence fixtures.

The suite runs in eight seconds, so there is no reason to gate on anything narrower. This
differs from EEG_fMRI_Pipeline, where a nine-minute suite makes targeted subsets the only
practical choice.

### Baseline

Measured at `6f22bd4`, before any model module exists:

```
485 passed, 4 skipped in 8.05s
mypy --strict: no issues found in 31 source files
ruff: all checks passed
```

Every gate is measured against this.

---

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Extracting the within-subject loop changes behaviour | High | Section 8.3 fixtures; scheduled last; `folds.tsv` makes fold assignment diffable |
| One outer loop becomes conditional soup serving four paths | Medium | The two axes are fold source and task; neither branches inside the loop. If a conditional appears in the loop body, the abstraction is wrong and the paths split again |
| `crossfit.py` grows past budget | Medium | 500-line ceiling; split by fold source if breached |
| SHAP's optional dependency leaks into core tests | Low | `importance` extra, skipped like the existing microstate tests |
| Recipe grammar drifts from `FeatureMeta` | Low | `[select]` keys are field names; a test asserts every key resolves to a real field |
| Equivalence fixtures silently stop being regenerated | Low | Fixtures carry the pipeline commit hash; the test reports it on failure |

### The honest summary

The leave-one-subject-out path is a low-risk move: the units already take arrays, the
fixtures are direct, and the work is mostly deleting configuration plumbing. The
within-subject path is a genuine rewrite of 1,231 lines that have no extractable core,
and that is where this migration can go wrong. Everything in the sequencing and
verification above is arranged around that single fact.

---

## 11. Adoption: what this does not yet give you

`model/design.py` reads `FeatureTable`s, and `io.read_table` reads only bundles written
by `io.write_table`. Existing study features are in the pipeline's own per-trial TSV
format. So on the day this lands, **`eegfeat fit` cannot be pointed at the current
study's feature tables.** One of two bridges is needed, and the choice is a real one:

- **Recompute features through `eegfeat`.** Cleanest, and `tests/test_equivalence.py`
  already demonstrates agreement with the pipeline for the measures `eegfeat` implements.
  Costs a cohort-wide feature run, and only covers implemented measures.
- **Write a reader** that builds a `FeatureTable` from the pipeline's TSVs, inferring
  `FeatureMeta` from column names. Cheap and immediate, but it reintroduces exactly the
  name-parsing this design removes, so it should be marked transitional.

This does not change the design. It determines when the design pays off, and it should be
decided before implementation starts rather than discovered at the end.

---

## 12. Open questions

- Which bridge in section 11, and whether it is in scope for this migration.
- Whether `time_generalization.py` and `cnn.py` should follow, and whether the CNN belongs
  in `eegfeat` at all rather than in a package depending on it.
- Whether EEG_fMRI_Pipeline should eventually import `eegfeat` rather than keeping its own
  copy. Worth answering once the equivalence fixtures pass, since they are the evidence
  that the two implementations agree.
