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

`eegfeat` computes features. It should also fit and evaluate models on them, so that
one repository carries a study from preprocessed epochs to a defensible result.

The methodology to do this already exists in EEG_fMRI_Pipeline: roughly 12,000 lines
across fifteen modules, guarded by 123 validity tests that encode years of accumulated
corrections. That methodology is repository-independent and worth keeping. The wiring
around it — BIDS derivative trees, subject discovery, an untyped configuration object
threaded through every function — is not.

This spec describes moving the first and leaving the second.

### Why this is a re-founding, not a copy

`orchestration.py` is 5,668 lines, seven and a half times the largest module in
`eegfeat`. The cause is measurable rather than stylistic: **83 configuration lookups in
that file alone, and 56 functions across the machine-learning modules that take an
untyped `config: Any`.** Every function re-reads and re-validates what it needs from an
opaque object, and the branching that results is most of the bulk.

`eegfeat` runs `mypy --strict` over `src/eegfeat`. `config: Any` does not survive
contact with it. Parsing a typed recipe once at the boundary and passing dataclasses
downward is what collapses the volume, and strict mode enforces that rather than
leaving it to discipline.

A second symptom points the same way. Many of the 123 validity tests reach through the
underscore — `orch._fit_tuned_regression_estimator`, `orch._resolve_permutation_scheme`,
`orch._circular_shift_group`. That is not a stylistic lapse. When a module is 5,668
lines its real units are not reachable from the public surface, so tests trespass to
get at them. Splitting the module makes those same units public and the tests honest.

### Non-goals

EEG_fMRI_Pipeline is not modified. Nothing is deleted, no imports are repointed, no
behaviour changes there. Whether the pipeline should later become a thin adapter over
`eegfeat` is a genuine question and a separate decision; folding it in now would put
existing cohort results at risk for no gain.

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
| `cnn.py` | Adds a torch dependency; least reusable component |
| `time_generalization.py` | Deferred with the CNN; revisit once the core is settled |
| `run_model_comparison_ml`, `run_incremental_validity_ml` | Study-specific analysis modes |
| `plotting.py` | Tied to the pipeline's output conventions; figures are downstream |
| `load_active_matrix` and the loading layer | BIDS trees, subject discovery, events lookup |
| `config.py`, `get_ml_config` | Replaced by typed recipe dataclasses |

---

## 3. Architecture

### 3.1 The seam

The pipeline keeps everything behind `load_active_matrix`. `eegfeat` never learns what
a subject directory is. Its entry point is a `FeatureTable` plus a target and a groups
vector, aligned on trial index.

### 3.2 Layout

```
src/eegfeat/
  iaf.py              fold-local individual alpha frequency (feature side)
  model/
    __init__.py
    design.py         FeatureTable(s) + target + groups -> design matrix
    splits.py         LOSO, within-subject, run-aware, inner CV
    transformers.py   missingness, variance, inf->nan, deconfound, spatial select
    estimators.py     pipelines and parameter grids, regression and classification
    residualize.py    fold-local nuisance regression on targets
    nulls.py          permutation schemes, circular shift, effectiveness checks
    metrics.py        subject-level aggregation and scoring
    fit.py            nested cross-validated execution
    importance.py     SHAP and permutation importance
    uncertainty.py    conformal prediction intervals
  runner/
    fit_recipe.py     parses a model recipe
    fit.py            executes one
    provenance.py     hashes, seeds, versions, git state
```

### 3.3 The boundary invariant

**No module under `model/` imports from `runner/`, reads a TOML file, or writes a
file.** Science takes arrays and returns values; orchestration reads recipes and writes
results.

This is the boundary `runner/compute.py` already holds against the feature modules —
they do not know recipes exist. It makes every module in `model/` unit-testable with
arrays alone, and it is checkable, so `tests/test_packaging.py` gains a test that
asserts it by inspecting imports.

### 3.4 Input contract

Core functions in `model/` take plain `(X, y, groups)` arrays. They are general and
easy to test. One adapter — `design.py` — turns a `FeatureTable` plus a target and
groups into a design matrix.

The payoff is that feature selection and fold harmonization become structured queries
over `FeatureMeta` rather than substring matching on column names. The pipeline
currently selects features by parsing name fragments (`feature_bands`,
`feature_segments`, `feature_scopes`, `feature_stats`); `eegfeat` already holds that
information as typed fields, so a renamed measure raises instead of silently selecting
nothing.

### 3.5 Dependencies

`scikit-learn` becomes a `model` extra and `shap` an `importance` extra, matching how
`connectivity`, `microstates` and `knee` already work. Installing `eegfeat` for feature
extraction alone continues to cost numpy, scipy, pandas and mne.

### 3.6 Fold-local IAF

`analysis/features/cv_hygiene.py` (340 lines) estimates individual alpha frequency from
training trials only and derives band edges from it. It lives on the feature side of the
pipeline but is imported by the machine-learning code, and it depends only on mne, numpy
and scipy.

It belongs in `eegfeat` as `iaf.py`, on the feature side, consumed by `model/design.py`
per fold. It is also the clearest argument for joining the two halves of the repository:
a band definition estimated on training trials only is a feature-computation concern
that exists solely inside a fold.

---

## 4. Module mapping

Every symbol that crosses, and where it lands. `orchestration.py` contributes to six
destinations; everything from it not listed here stays behind.

### `model/splits.py` — from `cv.py`

`create_loso_folds`, `create_inner_cv`, `create_run_aware_cv`,
`create_within_subject_folds`, `create_run_aware_inner_cv`, `get_inner_cv_splits`.

`get_min_channels_required` does not cross; it becomes a recipe field.

### `model/metrics.py` — from `cv.py`, `classification.py`, `orchestration.py`

From `cv.py`: `safe_pearsonr`, `make_pearsonr_scorer`, `create_scoring_dict`,
`aggregate_fold_results`, `compute_subject_level_r`, `compute_subject_level_errors`,
`compute_metrics`.

From `classification.py`: `ClassificationResult`.

From `orchestration.py`: `_subject_mean_metric`, `_count_finite_subject_metric`,
`_subject_metric_values`, `_bootstrap_mean_ci`, `_paired_signflip_p_value`,
`_subject_weighted_r2_scores`, `_within_subject_centered_prediction_metrics`,
`_within_condition_prediction_metrics`, `_within_condition_cells`,
`_center_within_cells`.

### `model/transformers.py` — from `preprocessing.py`

`VarianceThreshold`, `MissingnessThreshold`, `ReplaceInfWithNaN`, `DropAllNaNColumns`,
`SpatialFeatureSelector`, `Deconfounder`, `build_base_preprocessing_steps`,
`transform_feature_names_through_steps`, `validate_subject_missingness`.

### `model/estimators.py` — from `pipelines.py`, `classification.py`

`create_elasticnet_pipeline`, `create_ridge_pipeline`, `create_rf_pipeline`,
`create_svm_pipeline`, `create_logistic_pipeline`, `create_rf_classification_pipeline`,
`create_ensemble_pipeline`, and the six `build_*_param_grid` functions.

Each grid function currently takes `config: Any`; each becomes a typed dataclass
parameter.

### `model/nulls.py` — from `circular_shift.py`, `cv.py`, `orchestration.py`

`circular_shift.py` merges here whole — at 43 lines its concern is nulls, and its
docstring already carries the group-under-composition argument that justifies the
upper-tail p-value.

From `cv.py`: `permutation_changed_fraction`, `is_effective_permutation`,
`run_permutation_test`.

From `orchestration.py`: `_resolve_permutation_scheme`, `_validate_permutation_runs`,
`_validate_permutation_trial_indices`, `_circular_shift_group`,
`_trial_index_ordered_indices`, `_permutation_indices_by_scheme`,
`_permute_labels_by_scheme`, `_generate_effective_permutation`,
`filter_circular_shift_permutation_rows`, `_run_classification_permutations`.

### `model/residualize.py` — from `target_residualization.py`, `orchestration.py`

`FoldNuisanceFit`, `residualize_targets_for_fold`, `fit_nuisance_model_for_fold`,
`_design_matrix`, `_validate_indices`, `_validate_training_nuisance_rank`,
`_StagedResidualPreprocessor`, `_fit_staged_residual_preprocessor`,
`reconstruct_staged_permutation_target_for_fold`.

`configured_target_residualization_columns` does not cross; it becomes a recipe field.

### `model/design.py` — from `cv.py`, `orchestration.py`, and new

New: the `[select]` query over `FeatureMeta`, and assembly of a `FeatureTable` plus
target and groups into a design matrix.

From `cv.py`: `compute_train_group_intersection_mask`,
`apply_fold_feature_harmonization`, `apply_fold_specific_hygiene` (rewritten against
`eegfeat.iaf`).

From `orchestration.py`: `_target_covariate_aliases`,
`_warn_or_raise_if_binary_like_regression_target`,
`_apply_fold_feature_harmonization_foldwise`.

`design.py` owns the feature axis — which columns are in `X`, globally through
`[select]` and per fold through harmonization.

### `model/fit.py` — from `cv.py`, `classification.py`, `orchestration.py`

From `cv.py`: `set_random_seeds`, `determine_inner_n_jobs`, `should_parallelize_folds`,
`execute_folds_parallel`, `fit_with_warning_logging`, `grid_search_with_warning_logging`,
`_raise_for_nonfinite_grid_search_scores`, `nested_loso_predictions_matrix`,
`create_best_params_record`.

From `classification.py`: `decode_binary_outcome`, `nested_loso_classification`.

From `orchestration.py`: `_fit_tuned_regression_estimator`, `_fit_within_subject_fold`,
`_fit_estimator_with_optional_groups`, `_fit_subject_weighted_inner_cv_estimator`,
`_InnerSplitData`, `compute_baseline_predictions`, and the nested cross-validation cores
of `run_regression_ml`, `run_classification_ml`,
`run_within_subject_regression_ml` and `run_within_subject_classification_ml` — stripped
of configuration reading, file writing, plotting and study modes.

`_fit_default_pipeline` crosses as an explicit, named path. It must never be reachable
as a silent fallback from a failed fit; see safeguard 6.

### `model/importance.py` — from `shap_importance.py`, `feature_metadata.py`, `orchestration.py`

`SHAPResult`, `compute_shap_values`, `compute_shap_importance`,
`compute_shap_for_cv_folds`, `aggregate_importance`, and the permutation-importance
computation inside `_run_permutation_importance_stage`.

`feature_metadata.py` largely dissolves: `build_feature_metadata` and
`_channel_to_roi_map` reconstruct by hand what `FeatureMeta` already holds. Only
`aggregate_importance` survives, and it becomes a group-by over metadata fields.

### `model/uncertainty.py` — from `uncertainty.py`

`PredictionIntervalResult`, `compute_prediction_intervals`, `_conformal_split`,
`_conformal_cv_plus`, `_conformalized_quantile_regression`,
`_compute_conformal_quantile`, `_order_stat_quantile`, `_get_cv_splitter`.

### `runner/fit_recipe.py` — new

Nothing maps to it. The pipeline's equivalent is 83 scattered configuration lookups;
this parses the grammar in section 6 once, into typed dataclasses, reporting every
problem at once in the manner `runner/recipe.py` already establishes.

### `runner/provenance.py` — from `orchestration.py`

`_sha256_file`, `_sha256_array`, `_sha256_json`, `build_ml_input_hashes`,
`_git_metadata`, `write_reproducibility_info`, `_json_safe`, `_json_safe_scalar`,
`_normalize_subject_ids`.

Adds one link the pipeline cannot currently make: the upstream feature run's manifest
hash, so a fit records exactly which feature recipe produced its inputs.

### `runner/fit.py` — from `orchestration.py`

`export_subject_selection_report`, `export_baseline_predictions`, and the orchestration
shells of `_run_shap_importance_stage`, `_run_uncertainty_stage` and
`_run_permutation_importance_stage` — the minimum-valid-fold-fraction enforcement and
result writing, with the computation itself delegated to `model/`.

---

## 5. The safeguards contract

The README sells `eegfeat` on eliminating known failure modes. The model subpackage
earns the same section. These seven are what the 123 validity tests encode.

1. **Group-disjoint splits, always.** No subject or run appears in both train and test.
   Inner cross-validation refuses fewer than two training groups. Validation splits stay
   disjoint even when the group splitter falls back.

2. **Fold-local everything.** Feature harmonization, target residualization,
   IAF-derived band edges, imputation and scaling are fit on training trials only.

3. **No target leakage through covariates.** Covariates that alias the target are
   blocked, by name and by explicit column.

4. **Subject-level primary metrics.** Balanced accuracy, precision, recall, F1 and r are
   computed per subject and aggregated with equal subject weight — never pooled across
   trials, and with no fallback to a pooled AUC when the subject-level computation fails.

5. **Honest nulls.** A permutation must actually change labels. The circular-shift set is
   the full group under composition, not a filtered subset. A scheme mismatch raises
   instead of silently downgrading. An incomplete permutation run fails a configured
   completion threshold instead of being averaged over.

6. **Failures surface.** A fold that fails to fit raises; it does not fall back to a
   default fit. Every stage enforces a minimum valid-fold fraction.

7. **Non-finite guards.** Grid search rejects non-finite scores. A single-class training
   fold is an error, not a silent NaN.

### A distinction to preserve in safeguard 6

Safeguard 6 governs fits, not measurements, and the port must not flatten the two into
one policy.

A detector that resolves nothing has *made a measurement*: it must be reported and must
not abort a cohort run. A fold that fails to fit has *failed*: it must raise and must
not be averaged into a result. Both behaviours are correct in their own domain. The
documentation states the distinction explicitly so that neither is later "fixed" into
the other.

---

## 6. The model recipe

A separate file from the feature recipe, not new sections within it. The two runs have
different shapes: a feature run is expensive and parallel over recordings, while a fit
is one cohort-level operation whose parallelism is over folds and permutations. Many
models are fitted against one feature set, so welding them together would force either
duplication or recomputation. `runner/recipe.py` is already the largest module in the
repository at 753 lines and should not absorb a second grammar.

Command: `eegfeat fit model.toml`. Validation stays under the existing `eegfeat check`,
which dispatches on the kind of recipe the file declares; the current `_SECTIONS`
whitelist already rejects a foreign section loudly, so the error is clear either way.

```toml
[inputs]
root = "derivatives/eegfeat"        # a feature run's output tree
targets = "targets.tsv"             # trial index, target, groups, covariates

[select]                            # a structured query over FeatureMeta
bands = ["alpha", "beta"]
measures = ["integrated_band_power", "aperiodic"]
spatial = ["rois"]

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
public API — matching the rest of the `eegfeat` suite — with shared builders in the
existing `tests/synthetic.py`.

Categories overlap, so the counts below do not sum to 123.

| Topic | Count | Destination |
|---|---:|---|
| CV, folds, splits | 42 | `test_splits.py`, `test_fit.py` |
| Loading layer | 26 | Stays, except the leakage constraints |
| Permutations, nulls | 23 | `test_nulls.py` |
| Model comparison, incremental validity | 14 | Stays |
| Time generalization | 11 | Stays |
| Metrics | 11 | `test_metrics.py` |
| Leakage, covariates | 7 | `test_design.py` |
| Uncertainty, conformal | 6 | `test_uncertainty.py` |
| SHAP | 4 | `test_importance.py` |
| CNN | 4 | Stays |

Roughly 85 to 90 tests' worth of constraints cross.

### 8.2 Equivalence fixtures

Unit tests prove the new code is self-consistent. They do not prove it computes what the
old code computed. The feature half already solved this: `scripts/make_fixtures.py` runs
the reference pipeline once against a mounted data drive, writes npz fixtures, and
`tests/test_equivalence.py` asserts `eegfeat` reproduces them. It is never run from the
test suite.

The model half uses the identical method. `make_fixtures.py` gains a mode that runs the
pipeline's machine learning once on a fixed synthetic design matrix and dumps fold
assignments, predictions, metrics and null distributions. Seeds are already set per
fold, so the deterministic paths must match exactly.

This is the step that turns "12,000 lines were refactored" into a checkable claim.

---

## 9. Sequencing

Bottom-up, test-driven. Each module's constraints are written as tests before its code.

| Tier | Modules | Depends on |
|---|---|---|
| 0 | `splits.py`, `metrics.py`, `transformers.py` | numpy, scikit-learn |
| 1 | `nulls.py`, `estimators.py`, `residualize.py` | tier 0 |
| 2 | `iaf.py`, `design.py` | `eegfeat.table`, mne |
| 3 | `fit.py` | tiers 0–2 |
| 4 | `uncertainty.py`, `importance.py` | `fit.py` |
| 5 | `fit_recipe.py`, `runner/fit.py`, `provenance.py`, CLI | tiers 3–4 |

**Gate at every module:** the full suite, `mypy --strict`, `ruff`.
**Gate at the end:** the equivalence fixtures.

The suite runs in eight seconds, so there is no reason to gate on anything narrower.
This differs from EEG_fMRI_Pipeline, where a nine-minute suite makes targeted subsets
the only practical choice.

### Baseline

Measured at `6f22bd4`, before any model module exists:

```
485 passed, 4 skipped in 8.05s
mypy --strict: no issues found in 31 source files
ruff: all checks passed
```

Every gate above is measured against this.

---

## 10. Open questions

None blocking. Two to revisit once the core is settled:

- Whether `time_generalization.py` and `cnn.py` should follow, and whether the CNN
  belongs in `eegfeat` at all or in a separate package that depends on it.
- Whether EEG_fMRI_Pipeline should eventually import `eegfeat` rather than keeping its
  own copy. This becomes worth answering once the equivalence fixtures pass, since they
  are the evidence that the two implementations agree.
