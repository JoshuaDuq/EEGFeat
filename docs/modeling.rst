Predictive Modeling
===================

.. raw:: html

   <p class="hero-lede">
     Carry self-describing EEG feature tables into <strong>group-disjoint</strong>,
     leakage-safe predictive modeling with nested tuning, subject-level evaluation,
     permutation nulls, conformal intervals, and metadata-aware importance.
   </p>

``eegfeat.model`` is an optional scikit-learn-based subpackage. Install it with
``pip install "eegfeat[model]"``. Add ``pip install "eegfeat[importance]"`` for SHAP
importance; held-out permutation importance is included in the model extra.

The modeling boundary
---------------------

Modeling is defined for one row per epoch. The input :class:`~eegfeat.FeatureTable` must carry
``row_ids`` containing ``(recording, epoch, event)`` for every row. The target frame passed to
:func:`eegfeat.model.build_design` must contain matching ``recording``, ``epoch``, and ``event``
columns, the target column, and a grouping column such as ``subject_id``.

Tables from cross-trial measures are intentionally not accepted. ITPC, envelope correlation,
wPLI, and graph summaries have group rows rather than independent epoch rows; broadcasting
those values into an epoch-level design would create pseudo-replication.

Building a cohort
-----------------

For in-memory data, compute the same feature schema for each recording and stack the resulting
per-epoch tables. :func:`eegfeat.stack_rows` preserves input order, values, coverage, flags,
and metadata while rejecting duplicate row identities and schema mismatches. The target frame
must be assembled in the same canonical-key space; it is aligned by ``build_design`` rather
than by row position.

.. code-block:: python

   import pandas as pd
   import eegfeat as ef
   import eegfeat.model as efm

   selection = efm.Selection(
       band=("theta", "alpha"),
       space_kind=("channel",),
   )
   selected_tables = [efm.select(table, selection) for table in tables_by_recording]
   cohort_table = ef.stack_rows(selected_tables)

   # Each frame must contain recording, epoch, event, reaction_time, and subject_id.
   targets = pd.concat(target_frames, ignore_index=True)
   design = efm.build_design(
       cohort_table,
       targets,
       target="reaction_time",
       groups="subject_id",
   )

For runner output, write epoch descriptors with :func:`eegfeat.io.write_table` and pass the
resulting ``*_features.tsv`` paths to :func:`eegfeat.io.read_dataset`:

.. code-block:: python

   from eegfeat.io import read_dataset

   dataset = read_dataset(
       [
           "derivatives/eegfeat/sub-01_task-rest_features.tsv",
           "derivatives/eegfeat/sub-02_task-rest_features.tsv",
       ]
   )
   design = efm.build_design(
       dataset.table,
       dataset.targets,
       target="reaction_time",
       groups="subject_id",
   )

``read_dataset`` reads the descriptor columns named by each JSON sidecar. It constructs the
canonical key columns from the table's stored ``row_ids`` and refuses descriptors that disagree
with those identities. It does not infer targets or grouping variables from filenames; include
them in the descriptor rows when writing the tables.

Selecting features and covariates
---------------------------------

:class:`eegfeat.model.Selection` filters structured metadata fields, not generated feature-name
fragments. Its fields are ``measure``, ``band``, ``space_kind``, ``window``, ``normalization``,
and ``space``. Optional numeric covariates are appended to ``X`` after the feature columns and
are tracked by ``Design.covariate_columns``.

.. code-block:: python

   selection = efm.Selection(
       measure=("power",),
       band=("alpha", "beta"),
       space_kind=("channel",),
       normalization=("log10",),
   )
   design = efm.build_design(
       cohort_table,
       targets,
       target="reaction_time",
       groups="subject_id",
       runs="run",
       covariates=("age", "sex_code"),
       selection=selection,
   )

Missing covariates raise by default. Set ``strict_covariates=False`` only when dropping absent
requested covariates is part of the analysis plan. A target column cannot also be a covariate.

Group-disjoint cross-fitting
----------------------------

The outer split defines the evaluation question. :func:`eegfeat.model.loso_folds` evaluates
generalization to unseen groups. :func:`eegfeat.model.within_subject_folds` evaluates held-out
runs within each subject and requires complete run labels. Inner tuning uses
:class:`eegfeat.model.InnerSplit` and never crosses the selected grouping boundary.

.. code-block:: python

   config = efm.PreprocessingConfig(
       max_feature_missingness=0.2,
       max_subject_missingness=0.5,
       feature_selection_percentile=50.0,
   )
   pipeline = efm.ridge_pipeline(
       config,
       seed=42,
       n_covariates=design.n_covariates,
   )
   folds = efm.loso_folds(design.groups)
   inner = efm.InnerSplit(grouping="subject", n_splits=5)

   predictions = efm.cross_fit_regression(
       folds,
       design.X,
       design.y,
       design.groups,
       pipeline,
       efm.ridge_grid(),
       inner=inner,
       seed=42,
   )

The available regression pipelines are ``elasticnet_pipeline``, ``ridge_pipeline``, and
``random_forest_pipeline``. Classification pipelines are ``svm_pipeline``, ``logistic_pipeline``,
``random_forest_classifier_pipeline``, and ``ensemble_pipeline``; their targets must use 0/1
labels for :func:`eegfeat.model.classification_metrics`.

Preprocessing is fold-local. Pipelines replace infinities, drop all-NaN columns, enforce feature
and subject missingness limits, impute, remove constant columns, and optionally select features,
scale, deconfound, or reduce dimensionality with PCA. ``harmonization="intersection"`` keeps
only features finite for every training group; ``"union_impute"`` keeps the full feature union.
Target residualization is enabled with ``covariates=...`` and ``residualize_on=...`` on the
cross-fitting call, so nuisance models are fitted within each outer training fold.

Evaluation and aggregation
--------------------------

Use :func:`eegfeat.model.fold_results` to recover predictions in fold order and to map them back
to the design groups. :func:`eegfeat.model.regression_metrics` returns overall Pearson ``r``,
``R²``, explained variance, and subject-level correlation summaries. The default subject-level
correlation averages Fisher ``z`` values with equal subject weighting. Use
:func:`eegfeat.model.classification_metrics` for accuracy, balanced accuracy, AUC, average
precision, F1, precision, recall, specificity, and the confusion matrix.

.. code-block:: python

   import numpy as np

   y_true, y_pred, eval_groups, _, _ = efm.fold_results(
       predictions,
       groups=design.groups,
   )
   metrics, per_subject = efm.regression_metrics(
       y_true,
       y_pred,
       groups=np.asarray(eval_groups, dtype=object),
   )
   print(metrics["subject_level_r"])

For uncertainty around a subject-level summary, use :func:`eegfeat.model.bootstrap_mean_ci`
and :func:`eegfeat.model.paired_signflip_p_value` on a pre-specified vector of subject-level
statistics. The returned ``per_subject`` records from ``regression_metrics`` use ``{"subject":
..., "r": ...}`` mappings.

Permutation nulls
-----------------

:func:`eegfeat.model.permutation_test` refits the complete cross-fitting procedure for each
draw. ``NullConfig.scheme`` supports ``"within_subject"``, ``"run_wise"``,
``"within_subject_within_run"``, and ``"circular_shift_within_run"``. Run-aware schemes
require ``runs``; circular shifts additionally require finite within-run trial indices.
Incomplete fits abort the procedure with an error instead of dropping failed draws,
preventing distortion of the null distribution.

.. code-block:: python

   null = efm.permutation_test(
       folds,
       design.X,
       design.y,
       design.groups,
       design.runs,
       pipeline,
       efm.ridge_grid(),
       metrics["subject_level_r"],
       config=efm.NullConfig(
           scheme="within_subject",
           n_permutations=100,
       ),
       inner=inner,
       seed=42,
   )
   print(null.p_value)

Conformal prediction intervals
------------------------------

:func:`eegfeat.model.prediction_intervals` returns lower and upper bounds using ``"split"``,
``"cv_plus"``, or ``"quantile"`` conformal calibration. Providing ``groups`` makes the
model-fitting and calibration splits group-disjoint. Calibration scores are nevertheless
pooled across trials, so participants with more trials contribute more scores. The
implementation does not establish a distribution-free coverage guarantee for a new
participant. The returned object stores ``lower``, ``upper``, ``alpha``, and ``method``;
it does not contain realized test-set coverage.

.. code-block:: python

   intervals = efm.prediction_intervals(
       pipeline,
       design.X,
       design.y,
       design.X[:10],
       alpha=0.10,
       method="cv_plus",
       cv_splits=5,
       seed=42,
       groups=design.groups,
   )
   lower, upper = intervals.lower, intervals.upper

Importance and metadata
-----------------------

:func:`eegfeat.model.permutation_importance_over_folds` computes held-out permutation
importance for the same fold-fitted models used for evaluation. Pass the selected feature names
so scores remain attributable after fold-local filtering. Use
:func:`eegfeat.model.shap_importance_over_folds` for SHAP explanations; PCA and other steps
that mix input columns cannot be mapped back to one original feature.

The aggregation example below uses the feature-only ``design`` from the cohort section. If
covariates are included, fit importance on a matching pipeline and pass
``design.column_names``; do not aggregate covariate scores with EEG ``FeatureMeta`` records.

.. code-block:: python

   importance = efm.permutation_importance_over_folds(
       folds,
       design.X,
       design.y,
       design.groups,
       pipeline,
       efm.ridge_grid(),
       inner=inner,
       feature_names=cohort_table.names,
       seed=42,
   )
   by_band = efm.aggregate_by(importance, cohort_table.meta, field="band")

Importance values are changes in the selected scoring metric on held-out data; they are not
causal effects.
