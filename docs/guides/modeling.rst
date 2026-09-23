Predictive Modeling
===================

.. raw:: html

   <p class="hero-lede">
     Grouped cross-validation on per-epoch feature tables, with tuning inside
     the training folds, subject-level scores, permutation nulls, conformal
     intervals, and feature importance.
   </p>

``eegfeat.model`` is optional. Install it with ``pip install "eegfeat[model]"``.
SHAP needs ``pip install "eegfeat[importance]"``. Held-out permutation
importance is included in ``model``.

Inputs
------

Modeling uses one row per epoch. The :class:`~eegfeat.FeatureTable` must carry
``row_ids`` of ``(recording, epoch, event)`` for every row. The target frame
passed to :func:`eegfeat.model.build_design` must contain the same
``recording``, ``epoch``, and ``event`` columns, the target column, and a
grouping column such as ``subject_id``.

Cross-trial tables are rejected (:ref:`concepts-row-kinds`). ITPC, PPC,
envelope correlation, every ``spectral_connectivity`` method including wPLI, and
their graph summaries have one row per trial group. Copying a
group value onto its epochs repeats one number across rows.

Building a cohort
-----------------

For tables already in memory, compute the same measures per recording and stack
the per-epoch tables. :func:`eegfeat.stack_rows` keeps input order, values,
coverage, flags, and metadata, and it rejects duplicate row identities.
Recordings drop different bad channels, so their columns differ.
``columns="union"`` keeps every column any recording measured. A column a
recording did not measure is NaN with zero coverage. With
``harmonization="intersection"``, each fold then drops columns that some
training subject lacks. The target frame uses the same ``recording``,
``epoch``, and ``event`` keys. :func:`eegfeat.model.build_design` aligns rows
on those keys.

.. code-block:: python

   import pandas as pd
   import eegfeat as ef
   import eegfeat.model as efm

   selection = efm.Selection(
       band=("theta", "alpha"),
       space_kind=("channel",),
   )
   selected_tables = [efm.select(table, selection) for table in tables_by_recording]
   cohort_table = ef.stack_rows(selected_tables, columns="union")

   # Each frame must contain recording, epoch, event, reaction_time, and subject_id.
   targets = pd.concat(target_frames, ignore_index=True)
   design = efm.build_design(
       cohort_table,
       targets,
       target="reaction_time",
       groups="subject_id",
   )

For runner output, pass the ``*_features.tsv`` paths to
:func:`eegfeat.io.read_dataset`.

.. code-block:: python

   import eegfeat.model as efm
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

``read_dataset`` stacks the union of the feature columns. Pass
``columns="identical"`` to require one schema. Descriptor columns are those
named in each JSON sidecar. The canonical key columns are rebuilt from the
stored ``row_ids``, and a descriptor that disagrees with those identities is
rejected. Targets and grouping variables are not read from filenames. Put them
in the descriptor rows when writing the tables.

Selecting features and covariates
---------------------------------

:class:`eegfeat.model.Selection` filters on metadata fields, not on substrings
of the generated column name. The fields are ``measure``, ``band``,
``space_kind``, ``window``, ``normalization``, and ``space``. Numeric
covariates are appended to ``X`` after the feature columns and listed in
``Design.covariate_columns``.

``measure`` matches ``FeatureMeta.measure``, the label stored on the column.
That label is not always the function name or the recipe key.
:func:`eegfeat.integrated_band_power` labels columns ``"band_power"``.
:func:`eegfeat.peak_frequency` labels them ``"peak_freq_adjusted"``.
:func:`eegfeat.aperiodic` emits ``"slope"``, ``"offset"``, and ``"r_squared"``. The labels
in a cohort are ``sorted({m.measure for m in cohort_table.meta})``.

.. code-block:: python

   selection = efm.Selection(
       measure=("band_power",),
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

A missing covariate raises. ``strict_covariates=False`` drops requested
covariates that are absent from the target frame. The target column cannot
also be a covariate.

Group-disjoint cross-fitting
----------------------------

The outer split is the evaluation. :func:`eegfeat.model.loso_folds` holds out
groups. :func:`eegfeat.model.within_subject_folds` holds out runs within each
subject and requires a run label on every row. Inner tuning uses
:class:`eegfeat.model.InnerSplit` and stays inside the outer grouping.

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

Regression pipelines are ``elasticnet_pipeline``, ``ridge_pipeline``, and
``random_forest_pipeline``. Classification pipelines are ``svm_pipeline``,
``logistic_pipeline``, ``random_forest_classifier_pipeline``, and
``ensemble_pipeline``. :func:`eegfeat.model.classification_metrics` requires
labels in ``{0, 1}``.

Preprocessing is fit on the training rows of the fold. The pipeline replaces
infinities, drops all-NaN columns, applies the feature and subject missingness
limits, imputes, removes constant columns, and can select features, scale,
deconfound, or reduce dimension with PCA. ``harmonization="intersection"``
keeps features that have at least one finite value in every training group.
``harmonization="union_impute"`` keeps the full feature union. Target
residualization is ``covariates=...`` and ``residualize_on=...`` on the
cross-fitting call, so the nuisance model is fit inside each outer training
fold. Those least-squares fits scale the training design before solving. The
numerical rank cutoff then does not depend on the units of the covariates.

Evaluation
----------

:func:`eegfeat.model.fold_results` returns predictions in fold order and maps
them back to the design groups. :func:`eegfeat.model.regression_metrics`
returns Pearson ``r``, ``R²``, explained variance, and subject-level
correlation. The default subject-level correlation averages Fisher ``z`` with
equal weight per subject. :func:`eegfeat.model.classification_metrics` returns
accuracy, balanced accuracy, AUC, average precision, F1, precision, recall,
specificity, and the confusion matrix. With ``groups``, each scalar is the
equal-weight mean over the subjects for which it is defined. A subject with one
class is left out of the balanced-accuracy and AUC means. The confusion matrix stays pooled
over trials, so accuracy recomputed from it differs from the reported
``accuracy``.

Pearson correlation and centered ``R²`` use no absolute variance floor, so
whether they are defined does not depend on the units of the target.
Classification probabilities and the predictions used for model selection must
be finite on every trial. A failed prediction is an error.

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

:func:`eegfeat.model.bootstrap_mean_ci` and
:func:`eegfeat.model.paired_signflip_p_value` take a pre-specified vector of
subject-level statistics. ``bootstrap_mean_ci`` is a 95% percentile interval of
the mean. Pass Fisher-:math:`z` values, not correlations.
``paired_signflip_p_value`` is a two-sided test of zero mean on paired
differences and returns :math:`(b + 1)/(B + 1)`. Each ``per_subject`` record from
``regression_metrics`` is ``{"subject": ..., "r": ...}``.

Permutation nulls
-----------------

:func:`eegfeat.model.permutation_test` refits the full cross-fitting procedure
on each draw. It is for regression and refits
:func:`~eegfeat.model.cross_fit_regression`. ``NullConfig.scheme`` accepts ``"within_subject"``,
``"within_subject_within_run"``, and ``"circular_shift_within_run"``.
``"run_wise"`` is an alias of ``"within_subject_within_run"``, the name used
for this shuffle in the upstream pipeline (``runwise``). Both of those schemes
shuffle labels inside each run of each subject. No scheme exchanges whole runs.
Run structure differs by paradigm. Run-aware schemes require ``runs``.
Circular shifts also require finite integer trial indices (``trial_indices``)
that are unique inside each subject and run, and at least
``min_retained_trials`` trials per run (default 8). A draw that fails to fit raises. Failed draws are
not dropped from the null.

``greater_is_better`` chooses the tail. The default is ``True``. Set it to
``False`` when a smaller value is the better score, as with
``mean_squared_error``. Left at the default, a strong effect on an error
metric returns :math:`p \approx 1`, and only a model that scores worse than the
permuted refits returns a small :math:`p`. A model at chance gives a :math:`p`
spread uniformly on :math:`(0, 1]` under either tail. The direction is an argument. It is not inferred from ``metric_fn``.

``permutation_test`` rejects ``residualize_on``. Permuting the raw target and
then refitting the nuisance regression removes the association between the
nuisance and the target. A null that keeps that association needs its own
residual-permutation procedure and its own exchangeability conditions
(Winkler et al., 2014, https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/).

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

Conformal intervals
-------------------

:func:`eegfeat.model.prediction_intervals` returns bounds for ``"split"``,
``"cv_plus"``, or ``"quantile"`` conformal calibration. With ``groups``, the
fitting split and the calibration split are group-disjoint. Calibration scores
are still pooled over trials, so a participant with more trials contributes
more scores. The procedure does not give a distribution-free coverage guarantee
for a new participant. The result stores ``lower``, ``upper``, ``alpha``,
``method``, and ``calibration_unit``. It does not store the coverage realized on a test set.

Split conformal has marginal coverage of at least :math:`1 - \alpha` when
calibration and test trials are exchangeable (Lei et al., 2018). ``"cv_plus"``
and ``"quantile"`` use the CV+ construction of Barber et al. (2021), which puts
:math:`\alpha` in each tail. For :math:`K` folds of :math:`n` trials its
guarantee is

.. math::

   1 - 2\alpha - \min\left\{ \frac{2(1 - 1/K)}{n/K + 1},
   \frac{1 - K/n}{K + 1} \right\},

which is :math:`1 - 2\alpha` in the jackknife+ limit. Coverage near
:math:`1 - \alpha` is typical but not guaranteed. ``"quantile"`` is
conformalized quantile regression (Romano et al., 2019) in CV+ form. A
non-finite calibration score or prediction raises.

The test rows must not be used for fitting or calibration, or no coverage
statement applies to them. The example holds out the first subject.

.. code-block:: python

   held_out = design.groups == design.groups[0]
   intervals = efm.prediction_intervals(
       pipeline,
       design.X[~held_out],
       design.y[~held_out],
       design.X[held_out],
       alpha=0.10,
       method="cv_plus",
       cv_splits=5,
       seed=42,
       groups=design.groups[~held_out],
   )
   lower, upper = intervals.lower, intervals.upper

Importance
----------

:func:`eegfeat.model.permutation_importance_over_folds` computes held-out
permutation importance on the same fold-fitted models used for evaluation.
Pass the feature names that were selected so a score can be matched after
fold-local column drops. :func:`eegfeat.model.shap_importance_over_folds`
computes SHAP values. A step such as PCA that mixes columns cannot be mapped
back to one input feature.

The example below uses the feature-only ``design`` from the cohort section and
a pipeline without covariates. With covariates, fit on a pipeline with the same
``n_covariates`` and pass ``design.column_names``. Covariates have no
``FeatureMeta`` record, so :func:`~eegfeat.model.aggregate_by` refuses them.
Keep only ``design.feature_columns`` before aggregating.

.. code-block:: python

   feature_pipeline = efm.ridge_pipeline(config, seed=42)
   importance = efm.permutation_importance_over_folds(
       folds,
       design.X,
       design.y,
       design.groups,
       feature_pipeline,
       efm.ridge_grid(),
       inner=inner,
       feature_names=cohort_table.names,
       seed=42,
   )
   by_band = efm.aggregate_by(importance, cohort_table.meta, field="band")

The reported value is the mean decrease in the held-out score when that
feature is permuted. With ``scoring=None`` the score is ``estimator.score``:
:math:`R^2` for regressors and accuracy for classifiers.

References
----------

* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  *Distribution-free predictive inference for regression*. Journal of the
  American Statistical Association, 113(523), 1094--1111.
  `doi:10.1080/01621459.2017.1307116
  <https://doi.org/10.1080/01621459.2017.1307116>`__.
* Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021).
  *Predictive inference with the jackknife+*. The Annals of Statistics, 49(1),
  486--507. `doi:10.1214/20-AOS1965 <https://doi.org/10.1214/20-AOS1965>`__.
* Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized quantile
  regression*. Advances in Neural Information Processing Systems, 32.
* Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols,
  T. E. (2014). *Permutation inference for the general linear model*.
  NeuroImage, 92, 381--397. `doi:10.1016/j.neuroimage.2014.01.060
  <https://doi.org/10.1016/j.neuroimage.2014.01.060>`__.
