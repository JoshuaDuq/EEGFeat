API Reference
=============

.. raw:: html

   <p class="hero-lede">
     Complete API documentation for all public data containers, spectral descriptors,
     dynamics estimators, connectivity metrics, and predictive modeling tools in
     <code>eegfeat</code>.
   </p>

Core Data Structures
--------------------

.. autoclass:: eegfeat.FeatureTable
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.FeatureMeta
   :members:
   :show-inheritance:

.. autofunction:: eegfeat.concat

.. autofunction:: eegfeat.stack_rows

.. autoclass:: eegfeat.io.FeatureDataset
   :members:

.. autofunction:: eegfeat.io.read_dataset

.. autofunction:: eegfeat.io.read_table

.. autofunction:: eegfeat.io.write_table

.. autoclass:: eegfeat.Spectra
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Window
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Band
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Signal
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.BandSignal
   :members:
   :show-inheritance:

Predictive Modeling
-------------------

The modeling API is available when the ``model`` extra is installed. All model inputs are
per-epoch arrays or tables; cross-trial group-row tables are not accepted by
:func:`eegfeat.model.build_design`.

Design and preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eegfeat.model.Selection
   :members:

.. autoclass:: eegfeat.model.Design
   :members:

.. autoclass:: eegfeat.model.PreprocessingConfig
   :members:

.. autofunction:: eegfeat.model.select

.. autofunction:: eegfeat.model.build_design

.. autofunction:: eegfeat.model.harmonize_fold

Splits, estimators, and cross-fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: eegfeat.model.fold_results

.. autofunction:: eegfeat.model.regression_metrics

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

Spectral Features
-----------------

.. autofunction:: eegfeat.integrated_band_power
.. autofunction:: eegfeat.mean_psd
.. autofunction:: eegfeat.mean_tfr_power

.. autofunction:: eegfeat.band_ratio

.. autofunction:: eegfeat.asymmetry

.. autofunction:: eegfeat.peak_frequency

.. autofunction:: eegfeat.spectral_centroid

.. autofunction:: eegfeat.spectral_bandwidth

.. autofunction:: eegfeat.spectral_edge

.. autofunction:: eegfeat.spectral_entropy

.. autofunction:: eegfeat.aperiodic

.. autofunction:: eegfeat.aperiodic_ratio

Time-Domain Measures
--------------------

.. autofunction:: eegfeat.variance

.. autofunction:: eegfeat.mean_amplitude

.. autofunction:: eegfeat.peak_to_peak

.. autofunction:: eegfeat.area_under_curve

.. autofunction:: eegfeat.peak_amplitude

.. autofunction:: eegfeat.peak_latency

Oscillatory Bursts
------------------

.. autofunction:: eegfeat.burst_count

.. autofunction:: eegfeat.burst_rate

.. autofunction:: eegfeat.burst_duration

.. autofunction:: eegfeat.burst_amplitude

.. autofunction:: eegfeat.fraction_above_threshold

ERDS Dynamics
-------------

.. autofunction:: eegfeat.erds_mean

.. autofunction:: eegfeat.erds_slope

.. autofunction:: eegfeat.erd_magnitude

.. autofunction:: eegfeat.erd_duration

.. autofunction:: eegfeat.ers_magnitude

.. autofunction:: eegfeat.ers_duration

.. autofunction:: eegfeat.erds_peak_latency

.. autofunction:: eegfeat.erds_onset_latency

.. autofunction:: eegfeat.erds_rebound_latency

Phase & Connectivity
--------------------

.. autofunction:: eegfeat.itpc

.. autofunction:: eegfeat.ppc

.. autofunction:: eegfeat.pac

.. autofunction:: eegfeat.envelope_correlation

.. autofunction:: eegfeat.wpli

.. autofunction:: eegfeat.global_efficiency

.. autofunction:: eegfeat.clustering_coefficient

Complexity & Entropy
--------------------

.. autofunction:: eegfeat.sample_entropy

.. autofunction:: eegfeat.multiscale_entropy

Microstates
-----------

.. autofunction:: eegfeat.microstates.segment

.. autofunction:: eegfeat.microstates.microstate_coverage

.. autofunction:: eegfeat.microstates.microstate_duration

.. autofunction:: eegfeat.microstates.microstate_occurrence

.. autofunction:: eegfeat.microstates.microstate_transitions

.. autoclass:: eegfeat.microstates.MicrostateSegmentation
   :members:
   :show-inheritance:
