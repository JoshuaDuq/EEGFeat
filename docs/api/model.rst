Predictive Modeling
===================

Available with the ``model`` extra. Inputs are per-epoch arrays or tables.
:func:`eegtable.model.build_design` rejects cross-trial group-row tables.

The workflow is in :doc:`/guides/modeling`.


Design and preprocessing
-------------------------------------------

.. autoclass:: eegtable.model.Selection
   :members:

.. autoclass:: eegtable.model.Design
   :members:

.. autoclass:: eegtable.model.PreprocessingConfig
   :members:

.. autofunction:: eegtable.model.select

.. autofunction:: eegtable.model.build_design

.. autofunction:: eegtable.model.harmonize_fold

.. autofunction:: eegtable.model.compute_train_group_intersection_mask

Splits, estimators, and cross-fitting
-------------------------------------------

.. autoclass:: eegtable.model.Fold
   :members:

.. autoclass:: eegtable.model.InnerSplit
   :members:

.. autofunction:: eegtable.model.loso_folds

.. autofunction:: eegtable.model.within_subject_folds

.. autofunction:: eegtable.model.ridge_pipeline

.. autofunction:: eegtable.model.elasticnet_pipeline

.. autofunction:: eegtable.model.random_forest_pipeline

.. autofunction:: eegtable.model.logistic_pipeline

.. autofunction:: eegtable.model.svm_pipeline

.. autofunction:: eegtable.model.random_forest_classifier_pipeline

.. autofunction:: eegtable.model.ensemble_pipeline

.. autofunction:: eegtable.model.elasticnet_grid

.. autofunction:: eegtable.model.ridge_grid

.. autofunction:: eegtable.model.random_forest_grid

.. autofunction:: eegtable.model.svm_grid

.. autofunction:: eegtable.model.logistic_grid

.. autofunction:: eegtable.model.random_forest_classifier_grid

.. autofunction:: eegtable.model.cross_fit_regression

.. autofunction:: eegtable.model.cross_fit_classification

.. autoclass:: eegtable.model.FoldPrediction
   :members:

.. autoclass:: eegtable.model.FoldClassification
   :members:

Metrics, nulls, uncertainty, and importance
-------------------------------------------

.. autofunction:: eegtable.model.fold_results

.. autoclass:: eegtable.model.FoldResults
   :members:

.. autofunction:: eegtable.model.regression_metrics

.. autofunction:: eegtable.model.subject_level_r

.. autofunction:: eegtable.model.subject_level_errors

.. autofunction:: eegtable.model.subject_r_scorer

.. autofunction:: eegtable.model.within_subject_centered_metrics

.. autofunction:: eegtable.model.within_condition_metrics

.. autofunction:: eegtable.model.residualize_targets

.. autofunction:: eegtable.model.residualize_within_subjects

.. autofunction:: eegtable.model.classification_metrics

.. autoclass:: eegtable.model.ClassificationResult
   :members:

.. autoclass:: eegtable.model.AggregationConfig
   :members:

.. autoclass:: eegtable.model.SubjectLevelR
   :members:

.. autofunction:: eegtable.model.bootstrap_mean_ci

.. autofunction:: eegtable.model.paired_signflip_p_value

.. autofunction:: eegtable.model.permutation_test

.. autofunction:: eegtable.model.univariate_screen

.. autoclass:: eegtable.model.NullConfig
   :members:

.. autoclass:: eegtable.model.NullResult
   :members:

.. autofunction:: eegtable.model.prediction_intervals

.. autoclass:: eegtable.model.PredictionIntervals
   :members:

.. autoclass:: eegtable.model.Importance
   :members:

.. autofunction:: eegtable.model.permutation_importance

.. autofunction:: eegtable.model.permutation_importance_over_folds

.. autofunction:: eegtable.model.shap_importance

.. autofunction:: eegtable.model.shap_importance_over_folds

.. autofunction:: eegtable.model.aggregate_by

Learned signal features
-----------------------

.. autoclass:: eegtable.model.CSPTransformer
   :members:

.. autoclass:: eegtable.model.MicrostateTransformer
   :members:

.. autoclass:: eegtable.model.CovarianceTransformer
   :members:

.. autoclass:: eegtable.model.TangentSpaceTransformer
   :members:

.. autofunction:: eegtable.model.learned_pipeline

.. autofunction:: eegtable.model.cross_fit_signal_classification

.. autofunction:: eegtable.model.cross_fit_signal_regression
