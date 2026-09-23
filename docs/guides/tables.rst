Feature Tables and Files
========================

.. raw:: html

   <p class="hero-lede">
     Select columns by metadata, write TSV and JSON, and stack recordings.
   </p>

Field definitions and column names are in :doc:`/concepts`. This page is the
operations on a table.

Querying a table
----------------

.. code-block:: python

   # An empty table when nothing matches.
   alpha_channels = spectral_features.select(
       band=ef.Band("alpha", 8.0, 13.0), space_kind="channel"
   )

   # Fraction of finite input behind each cell. Not an artifact score.
   valid_fractions = spectral_features.coverage

   if "cog_fallback" in spectral_features.flags:
       fell_back = spectral_features.flags["cog_fallback"]

   df = spectral_features.to_dataframe()
   print(df.head())
   # eeg_band-power_alpha_cz_all_raw_p<hash>
   # eeg_peak-freq-adjusted_alpha_cz_all_raw_p<hash>

``select`` matches :class:`~eegfeat.FeatureMeta` fields.

Writing and reading
-------------------

:func:`eegfeat.io.write_table` writes a TSV of values, a ``_coverage.tsv`` of
the same shape, and a JSON sidecar. The filenames follow BIDS TSV and JSON
sidecar conventions. The layout is not a validated BIDS derivative.
:func:`eegfeat.io.read_table` restores metadata, flags, and row identity.
:func:`eegfeat.io.read_dataset` stacks per-epoch tables and returns descriptor
columns separately from features.

.. code-block:: python

   from eegfeat.io import read_dataset, read_table, write_table

   paths = write_table(spectral_features, "sub-01_features.tsv", rows=epochs.metadata)
   restored = read_table("sub-01_features.tsv")
   dataset = read_dataset(
       ["sub-01_features.tsv", "sub-02_features.tsv", "sub-03_features.tsv"]
   )

:doc:`/examples` shows the files the runner writes for a five-subject simulated
cohort.

Building a cohort
-----------------

:func:`eegfeat.stack_rows` concatenates per-epoch tables in input order. It
rejects duplicate row identities and cross-trial group tables.
``columns="union"`` keeps every column any recording measured. A recording that
did not measure a column gets ``NaN`` and zero coverage there. The default,
``columns="identical"``, requires one schema.

.. code-block:: python

   cohort_features = ef.stack_rows(
       [sub_01_features, sub_02_features, sub_03_features], columns="union"
   )

The frame passed to :func:`eegfeat.model.build_design` needs matching
``recording``, ``epoch``, and ``event`` keys, plus the target and grouping
columns. The cross-fitting workflow is in :doc:`/guides/modeling`.
