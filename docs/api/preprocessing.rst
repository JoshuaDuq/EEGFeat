Preprocessing
=============

Available with the ``preprocessing`` extra. The recipe, stage order, and
review files are in :doc:`/guides/preprocessing`.

Workflow
--------

.. autofunction:: eegfeat.preprocessing.load_recipe

.. autofunction:: eegfeat.preprocessing.load_config

.. autofunction:: eegfeat.preprocessing.open_workflow

.. autofunction:: eegfeat.preprocessing.list_steps

.. autofunction:: eegfeat.preprocessing.run_step

.. autofunction:: eegfeat.preprocessing.run_next

.. autofunction:: eegfeat.preprocessing.run_until

.. autofunction:: eegfeat.preprocessing.read_checkpoint

.. autofunction:: eegfeat.preprocessing.reset_from

.. autofunction:: eegfeat.preprocessing.preprocess

Settings
--------

.. autoclass:: eegfeat.preprocessing.PreprocessingConfig

.. autoclass:: eegfeat.preprocessing.ProcessingSettings

Numerical operations
--------------------

.. autofunction:: eegfeat.preprocessing.raw.prepare_channels

.. autofunction:: eegfeat.preprocessing.events.resolve_events

.. autofunction:: eegfeat.preprocessing.raw.crop_raw

.. autofunction:: eegfeat.preprocessing.raw.annotate_raw

.. autofunction:: eegfeat.preprocessing.quality.detect_annotations

.. autofunction:: eegfeat.preprocessing.quality.detect_bad_channels

.. autofunction:: eegfeat.preprocessing.quality.detect_bridges

.. autofunction:: eegfeat.preprocessing.quality.apply_raw_review

.. autofunction:: eegfeat.preprocessing.quality.repair_stimulation

.. autofunction:: eegfeat.preprocessing.raw.notch_raw

.. autofunction:: eegfeat.preprocessing.raw.filter_raw

.. autofunction:: eegfeat.preprocessing.artifacts.reference_artifact_data

.. autofunction:: eegfeat.preprocessing.ica.fit_ica

.. autofunction:: eegfeat.preprocessing.artifacts.fit_ssp

.. autofunction:: eegfeat.preprocessing.artifacts.fit_eog_regression

.. autofunction:: eegfeat.preprocessing.artifacts.review_artifact

.. autofunction:: eegfeat.preprocessing.epochs.make_epochs

.. autofunction:: eegfeat.preprocessing.artifacts.apply_artifact

.. autofunction:: eegfeat.preprocessing.rejection.fit_rejection

.. autofunction:: eegfeat.preprocessing.rejection.reject_epochs

.. autofunction:: eegfeat.preprocessing.rejection.apply_rejection

.. autofunction:: eegfeat.preprocessing.rejection.apply_epoch_review

.. autofunction:: eegfeat.preprocessing.epochs.interpolate_channels

.. autofunction:: eegfeat.preprocessing.epochs.reference_epochs

.. autofunction:: eegfeat.preprocessing.sampling.resample_epochs

.. autofunction:: eegfeat.preprocessing.sampling.crop_epochs

.. autofunction:: eegfeat.preprocessing.epochs.detrend_epochs

.. autofunction:: eegfeat.preprocessing.epochs.baseline_epochs

.. autofunction:: eegfeat.preprocessing.report.build_report

.. autofunction:: eegfeat.preprocessing.report.build_checkpoint_report

.. autofunction:: eegfeat.preprocessing.io.write_result
