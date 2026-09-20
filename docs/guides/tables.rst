Feature Tables and Files
========================

.. raw:: html

   <p class="hero-lede">
     Querying a <code>FeatureTable</code> by structured metadata, writing it to
     BIDS-style TSV with its sidecar, reading it back without losing anything, and
     stacking several recordings into one cohort.
   </p>

What a table guarantees, and what its column names mean, is covered in
:doc:`/concepts`. This page is the operations.

Querying a table
----------------

Unlike raw arrays, a :class:`~eegfeat.FeatureTable` is self-describing, quality-aware,
and safe against accidental mislabeling:

.. code-block:: python

   # Query columns using structured metadata
   alpha_rois = spectral_features.select(band=ef.Band("alpha", 8.0, 13.0), space_kind="roi")

   # Inspect finite-data coverage (not an artifact-rejection metric)
   valid_fractions = spectral_features.coverage

   # Inspect boolean quality flags (e.g. where peak prominence fell back to CoG)
   if "cog_fallback" in spectral_features.flags:
       fell_back = spectral_features.flags["cog_fallback"]

   # Convert to pandas DataFrame with canonical column names
   df = spectral_features.to_dataframe()
   print(df.head())
   # Output column names are structured slugs with unique parameter hashes:
   # eeg_band-power_alpha_cz_all_raw_p<hash>
   # eeg_peak-freq-adjusted_alpha_cz_all_raw_p<hash>

Writing and reading
-------------------

Tables serialize to BIDS-style TSV files with accompanying JSON sidecars and
finite-data coverage matrices. This is not a validated BIDS derivative structure.
``read_dataset`` restores descriptive row columns for modeling without treating
them as features:

.. code-block:: python

   from eegfeat.io import read_dataset, read_table, write_table

   # Write values, coverage matrix, and JSON sidecar
   paths = write_table(spectral_features, "sub-01_features.tsv", rows=epochs.metadata)

   # Restore exactly with full metadata, flags, and row labels
   restored = read_table("sub-01_features.tsv")

   # Load several per-epoch tables and their descriptors as one modeling dataset
   dataset = read_dataset(
       ["sub-01_features.tsv", "sub-02_features.tsv", "sub-03_features.tsv"]
   )

:doc:`/examples` shows these files as the runner actually writes them, for a
five-subject simulated cohort.

Building a cohort
-----------------

For in-memory cohort construction, use :func:`eegfeat.stack_rows` on per-epoch tables. It
preserves input order and rejects duplicate row identities and cross-trial group tables.
``columns="union"`` keeps every column any recording measured, which is what a cohort whose
recordings differ in their bad channels needs; the default requires one schema:

.. code-block:: python

   cohort_features = ef.stack_rows(
       [sub_01_features, sub_02_features, sub_03_features], columns="union"
   )

The target frame used by :func:`eegfeat.model.build_design` must carry matching
``recording``, ``epoch``, and ``event`` keys plus the target and grouping columns. See
:doc:`/guides/modeling` for the complete cross-fitting workflow.
