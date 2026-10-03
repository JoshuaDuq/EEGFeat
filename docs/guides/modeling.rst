Predictive Modeling
===================

.. raw:: html

   <p class="hero-lede">
     Grouped cross-validation on per-epoch feature tables, with tuning inside
     the training folds, subject-level scores, permutation nulls, a univariate
     screen, conformal intervals, and feature importance.
   </p>

``eegfeat.model`` is optional.

- **Install**: ``pip install "eegfeat[model]"``.
- **SHAP**: needs ``pip install "eegfeat[importance]"``.
- **Held-out permutation importance**: included in ``model``.

Inputs
------

Modeling uses one row per epoch.

**Feature table**
   The :class:`~eegfeat.FeatureTable` must carry ``row_ids`` of
   ``(recording, epoch, event)`` for every row.

**Target frame**
   The frame passed to :func:`eegfeat.model.build_design` must contain:

   - the same ``recording``, ``epoch``, and ``event`` columns,
   - the target column,
   - a grouping column such as ``subject_id``.

**Rejected tables**
   Cross-trial tables are rejected (:ref:`concepts-row-kinds`).

   - ITPC, PPC, envelope correlation, every ``spectral_connectivity`` method
     including wPLI, and their graph summaries have one row per trial group.
   - Copying a group value onto its epochs repeats one number across rows.

Building a cohort
-----------------

Stack per-epoch tables into one cohort table and align them with the targets.

Tables in memory
~~~~~~~~~~~~~~~~

Compute the same measures per recording and stack the per-epoch tables.

- **Stacking**: :func:`eegfeat.stack_rows` keeps input order, values, coverage,
  flags, and metadata, and rejects duplicate row identities.
- **Differing columns**: recordings drop different bad channels, so their
  columns differ.
- ``columns="union"``: keeps every column any recording measured. A column a
  recording did not measure is NaN with zero coverage.
- ``harmonization="intersection"``: each fold then drops columns that some
  training subject lacks.
- **Alignment**: the target frame uses the same ``recording``, ``epoch``, and
  ``event`` keys. :func:`eegfeat.model.build_design` aligns rows on those keys.

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

Runner output
~~~~~~~~~~~~~

Pass the ``*_features.tsv`` paths to :func:`eegfeat.io.read_dataset`.

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

- **Columns**: ``read_dataset`` stacks the union of the feature columns. Pass
  ``columns="identical"`` to require one schema; a mismatch names the recording
  and the columns that differ.
- **Per-recording fits**: a measure fitted to each recording, such as its
  microstate templates, names its columns after that fit, so those columns never
  match another recording's. Leave such a measure out of a cohort model with
  ``Selection(exclude=...)``.
- **Descriptors**: descriptor columns are those named in each JSON sidecar. The
  canonical key columns are rebuilt from the stored ``row_ids``, and a
  descriptor that disagrees with those identities is rejected.
- **Targets and groups**: not read from filenames. Put them in the descriptor
  rows when writing the tables.

Matching rows and subsets
~~~~~~~~~~~~~~~~~~~~~~~~~

``build_design`` requires every feature row to have a target row and every
target row a feature row, and says how many are left over on each side.

To model a subset of the epochs, such as the trials that remain after
exclusions, cut the table first.

- :meth:`~eegfeat.FeatureTable.take` cuts the table to the given rows.
- Choose excluded trials before evaluation using the study's quality criteria.
- Keep missingness-based feature selection inside the training folds. Calling
  :meth:`~eegfeat.FeatureTable.drop_missing` on the full cohort uses held-out
  observations to select columns even though it does not use the target.
  The modeling pipeline fits its missingness limit on each training split.

.. code-block:: python

   position = {row_id: i for i, row_id in enumerate(dataset.table.row_ids)}
   keys = zip(kept["recording"], kept["epoch"], kept["event"])
   table = dataset.table.take([position[key] for key in keys])

Selecting features and covariates
---------------------------------

:class:`eegfeat.model.Selection` filters on metadata fields, not on substrings
of the generated column name. The fields are ``measure``, ``band``,
``space_kind``, ``window``, ``normalization``, and ``space``. Numeric
covariates are appended to ``X`` after the feature columns and listed in
``Design.covariate_columns``.

``measure`` matches ``FeatureMeta.measure``, the label stored on the column.
That label is not always the function name or the recipe key.

- :func:`eegfeat.integrated_band_power` labels columns ``"band_power"``.
- :func:`eegfeat.peak_frequency` labels them ``"peak_freq_adjusted"``.
- :func:`eegfeat.aperiodic` emits ``"slope"``, ``"offset"``, and
  ``"r_squared"``.
- The labels in a cohort are ``sorted({m.measure for m in cohort_table.meta})``.

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

``exclude`` takes a second ``Selection``. The columns it matches are dropped
from those the other fields keep, so leaving one measure out does not mean
listing every other one.

.. code-block:: python

   microstates = ("coverage", "duration", "occurrence", "transition")
   selection = efm.Selection(exclude=efm.Selection(measure=microstates))

- **Missing covariate**: raises. ``strict_covariates=False`` drops requested
  covariates that are absent from the target frame.
- **Target as covariate**: the target column cannot also be a covariate.

Group-disjoint cross-fitting
----------------------------

The outer split is the evaluation. Inner tuning stays inside the outer grouping.

- :func:`eegfeat.model.loso_folds` holds out groups.
- :func:`eegfeat.model.within_subject_folds` holds out runs within each subject
  and requires a run label on every row.
- Inner tuning uses :class:`eegfeat.model.InnerSplit`.

.. code-block:: python

   config = efm.PreprocessingConfig(
       max_feature_missingness=0.2,
       max_subject_missingness=0.5,
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
       efm.ridge_grid(design.X),
       inner=inner,
       seed=42,
   )

Pipelines
~~~~~~~~~

- **Regression**: ``elasticnet_pipeline``, ``ridge_pipeline``, and
  ``random_forest_pipeline``.
- **Classification**: ``svm_pipeline``, ``logistic_pipeline``,
  ``random_forest_classifier_pipeline``, and ``ensemble_pipeline``.
  :func:`eegfeat.model.classification_metrics` requires labels in ``{0, 1}``.

Ridge grid
~~~~~~~~~~

:func:`~eegfeat.model.ridge_grid` takes the design because scikit-learn's
``Ridge`` does not divide its penalty by the number of trials.

- On standardized features the eigenvalues of the Gram matrix sum to
  trials × features, so a fixed grid stops shrinking as a cohort grows.
- With 1,200 trials and 12,600 features, a penalty of 100 barely touches any
  direction.
- The grid is scaled by that sum and runs from effectively unpenalized to an
  almost empty model.

Tuning statistic
~~~~~~~~~~~~~~~~

Regression is tuned on the statistic it is reported with.

- **Default**: each inner validation split is scored by its subject-level ``r``
  (:func:`~eegfeat.model.subject_r_scorer`), not by pooled ``R²``.
- **Why**: pooled ``R²`` rewards predicting each subject's mean, which the
  subject-level ``r`` ignores, and with no signal it always prefers the most
  heavily shrunk model.
- **Requirement**: the scorer needs at least 3 held-out trials of each
  validation subject.
- **Override**: pass ``scoring`` to select on something else.
- **Warning**: when every fold chooses the same end of a numeric grid,
  cross-fitting warns that the search may have stopped short, unless that end
  already means no penalty or an empty model.

Preprocessing
~~~~~~~~~~~~~

Preprocessing is fit on the training rows of the fold.

- **Steps**: the pipeline replaces infinities, drops all-NaN columns, applies the
  feature and subject missingness limits, imputes, and removes constant columns.
  It can also select features, scale, deconfound, or reduce dimension with PCA.
- ``harmonization="intersection"``: keeps features that have at least one
  finite value in every training group.
- ``harmonization="union_impute"``: keeps the full feature union.
- **Target residualization**: ``covariates=...`` and ``residualize_on=...`` on
  the cross-fitting call, so the nuisance model is fit inside each outer
  training fold.
- **Scaling**: those least-squares fits scale the training design before
  solving. The numerical rank cutoff then does not depend on the units of the
  covariates.

Within-subject residualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default nuisance model is pooled: one fit over the training subjects, which
removes only the average nuisance effect. Each subject's own departure from it,
such as a steeper response to the stimulus, stays in the residual, and a
feature that follows the stimulus the same way then appears to track the
target within subjects.

``residualize_within="subject"`` fits every subject's own nuisance model
instead.

- It applies the model to the features as well as to the target. The features'
  residuals use no target.
- A subject with training rows is fitted on them alone, including the scale
  used to remove numerical rounding from residuals. Non-finite feature values
  remain missing.
- A held-out subject, as in leave-one-subject-out folds, is fitted on its own
  rows. That defines the subject's residual target without informing any model.
- The estimand becomes the within-subject association beyond the nuisance.
- :func:`~eegfeat.model.residualize_within_subjects` is the same step for a fold
  loop of your own.

.. code-block:: python

   predictions = efm.cross_fit_regression(
       folds,
       design.X,
       design.y,
       design.groups,
       pipeline,
       efm.ridge_grid(design.X),
       inner=inner,
       seed=42,
       covariates=nuisance,  # one row per design row
       residualize_on=("run", "stimulus_temp"),
       residualize_within="subject",
   )

Evaluation
----------

Predictions are mapped back to the design groups, then scored.

- :func:`eegfeat.model.fold_results` returns predictions in fold order and maps
  them back to the design groups.
- :func:`eegfeat.model.regression_metrics` returns Pearson ``r``, ``R²``,
  explained variance, and subject-level correlation. The default subject-level
  correlation averages Fisher ``z`` with equal weight per subject.
- :func:`eegfeat.model.classification_metrics` returns accuracy, balanced
  accuracy, AUC, average precision, F1, precision, recall, specificity, and the
  confusion matrix.

Classification details
~~~~~~~~~~~~~~~~~~~~~~

- With ``groups``, each scalar is the equal-weight mean over the subjects for
  which it is defined.
- A subject with one class is left out of the balanced-accuracy and AUC means.
- The confusion matrix stays pooled over trials, so accuracy recomputed from it
  differs from the reported ``accuracy``.

Defined values
~~~~~~~~~~~~~~

- Subject-level regression summaries require a ``subject_id`` for every trial;
  missing labels raise instead of silently dropping trials during grouping.
- Pearson correlation and centered ``R²`` use no absolute variance floor, so
  whether they are defined does not depend on the units of the target.
- Classification probabilities and the predictions used for model selection must
  be finite on every trial.
- A failed prediction is an error.

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

Dependence between subjects
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Held-out scores from cross-subject folds are not independent. Every fold model
is trained on the other subjects, so two subjects' scores share most of their
training data.

- **Evidence**: in a null simulation of leave-one-subject-out ridge, a
  :math:`t` interval over the per-subject correlations excluded zero in 13–19%
  of cohorts at a nominal 5%.
- **Default**: :func:`~eegfeat.model.subject_level_r` therefore reports no
  interval unless ``AggregationConfig.ci_method`` asks for one.
- **When to ask**: only when no subject is scored by a model trained on another
  subject's data, as with within-subject folds.
- **Testing instead**: test cross-subject scores with
  :func:`~eegfeat.model.permutation_test`, which refits the whole procedure
  under the null.

Independent statistics
~~~~~~~~~~~~~~~~~~~~~~

:func:`eegfeat.model.bootstrap_mean_ci` and
:func:`eegfeat.model.paired_signflip_p_value` take a pre-specified vector of
independent subject-level statistics, which leave-one-subject-out scores are
not.

- ``bootstrap_mean_ci`` is a 95% percentile interval of the mean. Pass
  Fisher-:math:`z` values, not correlations.
- ``paired_signflip_p_value`` is a two-sided test of zero mean on paired
  differences and returns :math:`(b + 1)/(B + 1)`. Finite-sample validity also
  requires each subject's null difference to be symmetric about zero;
  independence and zero mean alone do not establish this.
- Both functions require finite values for every included subject. Exclude
  subjects using pre-specified study criteria before constructing the vector.

Permutation nulls
-----------------

:func:`eegfeat.model.permutation_test` refits the full cross-fitting procedure
on each draw. It is for regression and refits
:func:`~eegfeat.model.cross_fit_regression`.

Schemes
~~~~~~~

``NullConfig.scheme`` accepts ``"within_subject"``,
``"within_subject_within_run"``, and ``"circular_shift_within_run"``.

- ``"run_wise"`` is an alias of ``"within_subject_within_run"``, the name used
  for this shuffle in the upstream pipeline (``runwise``).
- Both of those schemes shuffle labels inside each run of each subject. No
  scheme exchanges whole runs.
- Run structure differs by paradigm. Run-aware schemes require ``runs``.
- Circular shifts also require finite integer trial indices
  (``trial_indices``) that are unique inside each subject and run.
- Circular shifts also require at least ``min_retained_trials`` trials per run
  (default 8).
- A draw that fails to fit raises. Failed draws are not dropped from the null.

These schemes specify rearrangements, not proof of their validity. Under the
null, the joint label or residual distribution must be invariant under the
chosen rearrangements. Run labels alone do not make temporally dependent
trials exchangeable. Circular shifts require invariance under cyclic shifts
of the retained, ordered trial sequence; ordinary stationarity alone does not
establish that condition.

Tail
~~~~

``greater_is_better`` chooses the tail. The default is ``True``.

- Set it to ``False`` when a smaller value is the better score, as with
  ``mean_squared_error``.
- Left at the default, a strong effect on an error metric returns
  :math:`p \approx 1`, and only a model that scores worse than the permuted
  refits returns a small :math:`p`.
- The smallest attainable Monte Carlo :math:`p` is
  :math:`1/(B + 1)` for :math:`B` draws. Ties count as at least as extreme;
  the null :math:`p` distribution can be conservative and discrete.
- The direction is an argument. It is not inferred from ``metric_fn``.

Nuisance covariates
~~~~~~~~~~~~~~~~~~~

With ``residualize_on``, pooled or within subjects, the null is Freedman-Lane
(Freedman & Lane, 1983; Winkler et al., 2014).

- Each fold fits its nuisance model exactly as cross-fitting does, keeps that
  fit's prediction, and permutes only its residuals.
- A draw therefore breaks the feature-target link and leaves the
  nuisance-target link in place.
- Permuting the raw target would break both, and the null would describe a
  different hypothesis.
- The residuals are exchanged within the scheme's blocks, which must stay inside
  each fold.

Batched ridge draws
~~~~~~~~~~~~~~~~~~~

A ridge pipeline fits permutation targets together when all of these hold:

- its preprocessing never sees the target, as in the default
  ``ridge_pipeline``, including any ``ColumnTransformer`` remainder;
- it is tuned on the subject-level ``r`` and scored with it.
- its regressor is unconstrained ``Ridge`` with an intercept and an ``auto``,
  ``cholesky`` or ``svd`` solver; only the scalar penalty is tuned, and every
  candidate penalty is finite and strictly positive.

Each fold and inner split transforms its features once. Its configured
`scikit-learn Ridge solver
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_
then fits batches of permutation targets as multiple outputs for each penalty.
Target residualization and the Freedman-Lane rebuild are linear in the target.
The resulting null matches individual refits up to numerical roundoff. Very
small penalties and nearly dependent predictors amplify that roundoff.
``alpha=0`` raises for batched inference: with dependent predictors, fitting
multiple targets together can change both the null statistic and the p-value.
This restriction does not change ordinary cross-fitting.

- **Full refitting**: other pipelines, custom container subclasses, a
  target-driven step such as ``feature_selection_percentile``, a ``metric_fn``
  or another ``scoring`` refit every draw.

.. code-block:: python

   null = efm.permutation_test(
       folds,
       design.X,
       design.y,
       design.groups,
       design.runs,
       pipeline,
       efm.ridge_grid(design.X),
       metrics["subject_level_r"],
       config=efm.NullConfig(
           scheme="within_subject",
           n_permutations=1000,
       ),
       inner=inner,
       seed=42,
   )
   print(null.p_value)

Univariate screen
-----------------

:func:`eegfeat.model.univariate_screen` asks which single features track the
target within subjects.

- **Statistic**: each subject's correlation is taken over its own trials,
  averaged across subjects in Fisher :math:`z`, and tested with a one-sample
  :math:`t`.
- **Independence**: no model is trained across subjects, so here the subjects are
  independent.
- ``p_fwer``: flipping the sign of a subject's :math:`z` for every feature at
  once keeps the dependence between features. The largest :math:`|t|` over those
  flips gives ``p_fwer``. Its finite-sample family-wise error guarantee assumes
  independent subject vectors with a joint null distribution invariant under
  sign reversal. Independence alone does not establish that symmetry.
- ``q``: the Benjamini-Hochberg adjustment of ``p``.
- ``residualize_on``: removes each subject's own nuisance design from both sides
  first.
- Every trial needs a subject label; missing labels raise.

Statistics use centered sums of squares so identical or nearly identical
subject effects do not produce negative variances through numerical
cancellation. If both the mean effect and its between-subject variance are
zero, ``t``, ``p``, ``q`` and ``p_fwer`` are undefined (NaN). Such features do
not enter the Benjamini-Hochberg adjustment.

.. code-block:: python

   screen = efm.univariate_screen(
       design.X[:, design.feature_columns],
       design.y,
       design.groups,
       feature_names=[m.name for m in cohort_table.meta],
   )
   print(screen.sort_values("p_fwer").head())

Conformal intervals
-------------------

:func:`eegfeat.model.prediction_intervals` returns bounds for ``"split"``,
``"cv_plus"``, or ``"quantile"`` conformal calibration.

- **Groups**: with ``groups``, the fitting split and the calibration split are
  group-disjoint.
- **Pooling**: calibration scores are still pooled over trials, so a participant
  with more trials contributes more scores.
- **No new-participant guarantee**: the procedure does not give a
  distribution-free coverage guarantee for a new participant.
- **Result fields**: ``lower``, ``upper``, ``alpha``, ``method``, and
  ``calibration_unit``.
- **Not stored**: the coverage realized on a test set.

Coverage guarantees
~~~~~~~~~~~~~~~~~~~

- ``"split"``: marginal coverage of at least :math:`1 - \alpha` when
  calibration and test trials are exchangeable (Lei et al., 2018).
- ``"cv_plus"`` and ``"quantile"``: the CV+ construction of Barber et al.
  (2021), which puts :math:`\alpha` in each tail. For :math:`K` folds of
  :math:`n` trials its guarantee is

  .. math::

     1 - 2\alpha - \min\left\{ \frac{2(1 - 1/K)}{n/K + 1},
     \frac{1 - K/n}{K + 1} \right\},

  which is :math:`1 - 2\alpha` in the jackknife+ limit.
- Coverage near :math:`1 - \alpha` is typical but not guaranteed.
- ``"quantile"`` is conformalized quantile regression (Romano et al., 2019) in
  CV+ form.
- A non-finite calibration score or prediction raises.

Example
~~~~~~~

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

Importance is computed on the same fold-fitted models used for evaluation.

- :func:`eegfeat.model.permutation_importance_over_folds` computes held-out
  permutation importance. Pass the feature names that were selected so a score
  can be matched after fold-local column drops.
- :func:`eegfeat.model.shap_importance_over_folds` computes SHAP values.
- A step such as PCA that mixes columns cannot be mapped back to one input
  feature.

The example below uses the feature-only ``design`` from the cohort section and
a pipeline without covariates. With covariates:

- fit on a pipeline with the same ``n_covariates``;
- pass ``design.column_names``;
- keep only ``design.feature_columns`` before aggregating, because covariates
  have no ``FeatureMeta`` record and :func:`~eegfeat.model.aggregate_by`
  refuses them.

.. code-block:: python

   feature_pipeline = efm.ridge_pipeline(config, seed=42)
   importance = efm.permutation_importance_over_folds(
       folds,
       design.X,
       design.y,
       design.groups,
       feature_pipeline,
       efm.ridge_grid(design.X),
       inner=inner,
       feature_names=cohort_table.names,
       seed=42,
   )
   by_band = efm.aggregate_by(importance, cohort_table.meta, field="band")

The reported value is the mean decrease in the held-out score when that
feature is permuted.

- ``scoring=None``: the score is the one the model was selected on, the
  subject-level ``r`` for regressors and accuracy for classifiers.

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
* Freedman, D., & Lane, D. (1983). *A nonstochastic interpretation of reported
  significance levels*. Journal of Business & Economic Statistics, 1(4),
  292--298.
* Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols,
  T. E. (2014). *Permutation inference for the general linear model*.
  NeuroImage, 92, 381--397. `doi:10.1016/j.neuroimage.2014.01.060
  <https://doi.org/10.1016/j.neuroimage.2014.01.060>`__.
