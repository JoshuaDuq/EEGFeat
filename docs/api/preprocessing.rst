Preprocessing
=============

Available with the ``preprocessing`` extra. The recipe, stage order, and
review files are in :doc:`/guides/preprocessing`.

Workflow
--------

.. autofunction:: eegtable.preprocessing.load_recipe

.. autofunction:: eegtable.preprocessing.load_config

.. autofunction:: eegtable.preprocessing.open_workflow

.. autofunction:: eegtable.preprocessing.list_steps

.. autofunction:: eegtable.preprocessing.run_step

.. autofunction:: eegtable.preprocessing.run_next

.. autofunction:: eegtable.preprocessing.run_until

.. autofunction:: eegtable.preprocessing.read_checkpoint

.. autofunction:: eegtable.preprocessing.reset_from

.. autofunction:: eegtable.preprocessing.preprocess

Settings
--------

.. autoclass:: eegtable.preprocessing.PreprocessingConfig

.. autoclass:: eegtable.preprocessing.ProcessingSettings

Numerical operations
--------------------

.. autofunction:: eegtable.preprocessing.raw.prepare_channels

.. autofunction:: eegtable.preprocessing.events.resolve_events

.. autofunction:: eegtable.preprocessing.raw.crop_raw

.. autofunction:: eegtable.preprocessing.raw.annotate_raw

.. autofunction:: eegtable.preprocessing.quality.detect_annotations

.. autofunction:: eegtable.preprocessing.quality.detect_bad_channels

.. autofunction:: eegtable.preprocessing.quality.detect_bridges

.. autofunction:: eegtable.preprocessing.quality.apply_raw_review

.. autofunction:: eegtable.preprocessing.quality.repair_stimulation

.. autofunction:: eegtable.preprocessing.raw.notch_raw

.. autofunction:: eegtable.preprocessing.raw.filter_raw

.. autofunction:: eegtable.preprocessing.artifacts.reference_artifact_data

.. autofunction:: eegtable.preprocessing.ica.fit_ica

.. autofunction:: eegtable.preprocessing.artifacts.fit_ssp

.. autofunction:: eegtable.preprocessing.artifacts.fit_eog_regression

.. autofunction:: eegtable.preprocessing.artifacts.review_artifact

.. autofunction:: eegtable.preprocessing.epochs.make_epochs

.. autofunction:: eegtable.preprocessing.artifacts.apply_artifact

.. autofunction:: eegtable.preprocessing.rejection.fit_rejection

.. autofunction:: eegtable.preprocessing.rejection.reject_epochs

.. autofunction:: eegtable.preprocessing.rejection.apply_rejection

.. autofunction:: eegtable.preprocessing.rejection.apply_epoch_review

.. autofunction:: eegtable.preprocessing.epochs.interpolate_channels

.. autofunction:: eegtable.preprocessing.epochs.reference_epochs

.. autofunction:: eegtable.preprocessing.sampling.resample_epochs

.. autofunction:: eegtable.preprocessing.sampling.crop_epochs

.. autofunction:: eegtable.preprocessing.epochs.detrend_epochs

.. autofunction:: eegtable.preprocessing.epochs.baseline_epochs

.. autofunction:: eegtable.preprocessing.report.build_report

.. autofunction:: eegtable.preprocessing.report.build_checkpoint_report

.. autofunction:: eegtable.preprocessing.io.write_result
