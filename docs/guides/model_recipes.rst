Modeling recipes and result bundles
===================================

``eegfeat model`` runs nested grouped cross-validation from a strict YAML
recipe. Install the modeling dependencies with
``pip install "eegfeat[model]"``. The array-based APIs remain in
``eegfeat.model``; reading recipes and writing bundles belong to the runner.

Start, validate, run
--------------------

.. code-block:: console

   eegfeat model init model.yaml
   # Edit the inputs, target, grid and output in model.yaml.
   eegfeat model check model.yaml
   eegfeat model run model.yaml

``init`` writes the following template and refuses to replace an existing
file. Paths are resolved relative to the recipe, including the output path.

.. literalinclude:: ../../src/eegfeat/runner/model_template.yaml
   :language: yaml

``check`` reads the inputs and validates row alignment, feature selection,
quality criteria, estimator parameter names and every outer/inner split. It
does not fit models or write results. Estimator-specific parameter range
validation still occurs when scikit-learn fits the estimator.

Unknown or duplicate YAML keys, invalid settings, missing samples, and failed
fits raise errors. A failed run leaves no completed output directory. The
output must be a new directory: completed results are never overwritten.

Inputs and sample identity
--------------------------

Every input is a feature TSV with its JSON and coverage sidecars. The reader
checks the stored feature definitions and sidecar identities before stacking.
``columns: union`` retains all measured feature definitions; absent columns
receive NaN and zero coverage. ``columns: identical`` requires the same
feature schema in each input.

Set ``inputs.rows`` explicitly when using trial-group features:

* ``epochs`` uses one sample per ``(recording, epoch, event)`` and delegates
  alignment to :func:`eegfeat.model.build_design`.
* ``groups`` uses one sample per ``(recording, group)`` and delegates to
  :func:`eegfeat.group.build_group_design`. Each row retains its ``n_trials``
  descriptor. Group-level features are never copied onto individual epochs.

Targets and the model grouping variable must be descriptor columns. For
example, ``groups: subject_id`` keeps a participant's rows together in every
split. The grouping column can instead identify another independent study
unit, such as a family or acquisition site, when that matches the evaluation
question. Labels must be present and nonempty on every row.

An optional ``inputs.targets`` TSV supplies additional descriptor columns.
Its identity keys must match all feature samples exactly, without duplicate,
missing or extra samples. Other descriptor names must not overlap existing
ones. Row order may differ because the join uses identity keys.

Regression requires finite numeric targets. Classification currently requires
both classes coded exactly ``0`` and ``1``. Group-row recipes do not accept
covariates. For epoch rows, ``analysis.covariates`` appends numeric covariates
through the existing fold-local preprocessing pipeline.

Feature selection and quality
-----------------------------

``selection`` matches :class:`eegfeat.model.Selection` metadata fields:
``measure``, ``band``, ``space_kind``, ``window``, ``normalization`` and
``space``. Empty lists match all values. ``exclude`` accepts another selection
mapping and removes its matches. Choose these criteria before evaluating
predictive performance.

``quality.min_coverage`` rejects cells with coverage below the specified
fraction. ``quality.rejected_flags`` rejects cells carrying any listed stored
flag; unknown flag names raise. Rejected cells become NaN while their
coverage, flags and a per-cell reason ledger are retained. Rows are preserved.
Preprocessing then learns missingness filters and imputation separately from
each training split. Quality thresholds are fixed criteria rather than
parameters chosen using held-out outcomes.

Estimator and tuning grid
-------------------------

The recipe requires an explicit, nonempty ``model.grid``. Candidates are
scalar YAML values, and parameter names must belong to the selected pipeline.
Useful estimator parameter names are:

.. list-table:: Supported estimators
   :header-rows: 1
   :widths: 25 25 50

   * - Task
     - Estimator
     - Example grid parameters
   * - regression
     - ``ridge``
     - ``regressor__alpha``
   * - regression
     - ``elasticnet``
     - ``regressor__alpha``, ``regressor__l1_ratio``
   * - regression
     - ``random_forest``
     - ``rf__n_estimators``, ``rf__max_depth``
   * - classification
     - ``logistic``
     - ``lr__C``
   * - classification
     - ``svm``
     - ``svm__C``, ``svm__gamma``
   * - classification
     - ``random_forest``
     - ``rf__n_estimators``, ``rf__max_depth``

Choose the grid for the scientific question and sample size. The template's
ridge grid is an example, not a data-adaptive penalty rule. Backend parameter
and fitting errors surface. SVM probability calibration through the
estimator's hidden cross-validation is rejected because it cannot enforce
the configured group boundaries. SVM recipes report predicted-label metrics.
Probability columns and probability-based metrics require an estimator that
provides probabilities under this workflow.

``preprocessing`` configures the existing pipeline's feature and subject
missingness limits, optional supervised feature selection percentile, and
optional PCA component count or explained-variance fraction. Scaling,
imputation, dimension reduction and supervised selection are fitted inside
each training split. No transformer is fitted on the full cohort first.

Nested group splits
-------------------

``validation.outer: loso`` holds out one model group at a time.
``validation.outer: group_kfold`` requires an explicit ``outer_splits`` and
holds out disjoint sets of groups. Every input sample receives exactly one
outer-test prediction. ``validation.inner_splits`` is enforced inside every
outer training set; it is never reduced to fit a smaller cohort. At least
three independent groups are required. Classification additionally requires
both classes in every inner training and validation split.

Hyperparameter tuning reruns the full pipeline inside those inner splits.
The selected pipeline is then refitted on the outer training rows and applied
once to the held-out rows. The grouping variable governs both levels.
``validation.seed`` controls the existing deterministic splitting and
estimator settings.

The recipe defaults to ``neg_mean_squared_error`` for regression and
``balanced_accuracy`` for classification. Supported alternatives are
``neg_mean_absolute_error`` or ``r2`` for regression, and ``accuracy``,
``roc_auc`` or ``average_precision`` for classification. The tuning statistic
is recorded separately from the reported metrics. A recipe that repeatedly
selects a grid boundary warrants a prespecified wider search in a new run.

This structure follows scikit-learn's
`nested cross-validation example
<https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>`_
and uses group-disjoint splitting as described in its
`GroupKFold documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>`_.
Group separation prevents direct sharing of samples from one unit between
training and testing; it does not establish exchangeability or make an
inappropriate grouping variable scientifically valid.

Auditable results
-----------------

The runner stages a complete bundle beside the destination, holds an output
lock, and publishes the directory atomically after validation. Bundle files
include:

* ``predictions.tsv``: original sample identity, model group, observed target,
  held-out prediction, outer fold and available class probabilities.
* ``folds.json``: absolute outer and inner row indices, group labels and each
  outer fold's selected parameters.
* ``metrics.json``: pooled regression errors and group correlation summaries,
  or pooled and equal-group-mean classification metrics with group details.
* ``design_features.tsv``, its coverage TSV and JSON: selected feature
  definitions, quality-masked values, preserved coverage and flags.
* ``design_matrix.npz`` and ``design.json``: the numerical design, target,
  feature/covariate positions and ordered column names.
* ``quality_ledger.tsv`` and ``feature_quality.tsv``: exclusions and support
  summaries for the selected feature definitions.
* ``resolved.yaml``: all resolved settings, absolute paths and recipe identity.
* ``manifest.json``: input content hashes, recipe hash, installed software
  versions, implementation source hash, and SHA-256 for every other bundle
  file.

Input, recipe, software and source identities are captured before preparing
the design or fitting. The runner verifies them again before publication and
rejects changes during execution. Undefined statistics are preserved as
``"NaN"`` in JSON rather than assigned a numerical value; inspect the number
of samples and groups supporting each reported statistic.

The bundle records nested evaluation rather than a single final estimator
trained on all rows. To audit a prediction, use its outer fold, stored row
indices, selected parameters and resolved settings with the existing array
APIs. This separation keeps estimation and evaluation explicit.

Python entry points
-------------------

.. code-block:: python

   from eegfeat.runner.model_recipe import load_model_recipe
   from eegfeat.runner.model_run import check_model, run_model

   recipe = load_model_recipe("model.yaml")
   check = check_model(recipe)
   result = run_model(recipe)
   print(result.output)
