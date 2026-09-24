Predictive Modeling
===================

Available with the ``model`` extra. Inputs are per-epoch arrays or tables.
:func:`eegfeat.model.build_design` rejects cross-trial group-row tables.

The workflow is in :doc:`/guides/modeling`.


Design and preprocessing
-------------------------------------------

.. autoclass:: eegfeat.model.Selection
   :members:

.. autoclass:: eegfeat.model.Design
   :members:

.. autoclass:: eegfeat.model.PreprocessingConfig
   :members:

.. autofunction:: eegfeat.model.select

.. autofunction:: eegfeat.model.build_design

.. autofunction:: eegfeat.model.harmonize_fold

.. autofunction:: eegfeat.model.compute_train_group_intersection_mask

Splits, estimators, and cross-fitting
-------------------------------------------

.. autoclass:: eegfeat.model.Fold
   :members:

.. autoclass:: eegfeat.model.InnerSplit
   :members:

.. autofunction:: eegfeat.model.loso_folds

.. autofunction:: eegfeat.model.within_subject_folds

.. autofunction:: eegfeat.model.ridge_pipeline

.. autofunction:: eegfeat.model.elasticnet_pipeline

.. autofunction:: eegfeat.model.random_forest_pipeline

.. autofunction:: eegfeat.model.logistic_pipeline

.. autofunction:: eegfeat.model.svm_pipeline

.. autofunction:: eegfeat.model.random_forest_classifier_pipeline

.. autofunction:: eegfeat.model.ensemble_pipeline

.. autofunction:: eegfeat.model.elasticnet_grid

.. autofunction:: eegfeat.model.ridge_grid

.. autofunction:: eegfeat.model.random_forest_grid

.. autofunction:: eegfeat.model.svm_grid

.. autofunction:: eegfeat.model.logistic_grid

.. autofunction:: eegfeat.model.random_forest_classifier_grid

.. autofunction:: eegfeat.model.cross_fit_regression

.. autofunction:: eegfeat.model.cross_fit_classification

.. autoclass:: eegfeat.model.FoldPrediction
   :members:

.. autoclass:: eegfeat.model.FoldClassification
   :members:

Metrics, nulls, uncertainty, and importance
-------------------------------------------

.. autofunction:: eegfeat.model.fold_results

.. autoclass:: eegfeat.model.FoldResults
   :members:

.. autofunction:: eegfeat.model.regression_metrics

.. autofunction:: eegfeat.model.subject_level_r

.. autofunction:: eegfeat.model.subject_level_errors

.. autofunction:: eegfeat.model.subject_r_scorer

.. autofunction:: eegfeat.model.within_subject_centered_metrics

.. autofunction:: eegfeat.model.within_condition_metrics

.. autofunction:: eegfeat.model.residualize_targets

.. autofunction:: eegfeat.model.residualize_within_subjects

.. autofunction:: eegfeat.model.classification_metrics

.. autoclass:: eegfeat.model.ClassificationResult
   :members:

.. autoclass:: eegfeat.model.AggregationConfig
   :members:

.. autoclass:: eegfeat.model.SubjectLevelR
   :members:

.. autofunction:: eegfeat.model.bootstrap_mean_ci

.. autofunction:: eegfeat.model.paired_signflip_p_value

.. autofunction:: eegfeat.model.permutation_test

.. autofunction:: eegfeat.model.univariate_screen

.. autoclass:: eegfeat.model.NullConfig
   :members:

.. autoclass:: eegfeat.model.NullResult
   :members:

.. autofunction:: eegfeat.model.prediction_intervals

.. autoclass:: eegfeat.model.PredictionIntervals
   :members:

.. autoclass:: eegfeat.model.Importance
   :members:

.. autofunction:: eegfeat.model.permutation_importance

.. autofunction:: eegfeat.model.permutation_importance_over_folds

.. autofunction:: eegfeat.model.shap_importance

.. autofunction:: eegfeat.model.shap_importance_over_folds

.. autofunction:: eegfeat.model.aggregate_by

