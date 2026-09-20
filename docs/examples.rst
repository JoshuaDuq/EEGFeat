Example Output
==============

.. raw:: html

   <p class="hero-lede">
     Real files, produced by the real pipeline, from simulated recordings. Five
     subjects, two runs each, sixteen trials per run — enough to see exactly what
     comes out before pointing any of it at your own data.
   </p>

Everything below lives in the `examples/ directory
<https://github.com/JoshuaDuq/EEGFeatML/tree/main/examples>`_ of the repository.
Regenerate it all with:

.. code-block:: bash

   python -m pip install -e ".[model]"
   python examples/make_examples.py

What feature extraction writes
------------------------------

`recipe.toml <https://github.com/JoshuaDuq/EEGFeatML/blob/main/examples/recipe.toml>`_
is the input: bands, windows, regions, and one entry per measure. ``eegfeat run``
applies it to every epochs file and writes, for each recording:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - What it contains
   * - ``*_features.tsv``
     - One row per epoch. The first columns are the epoch's identity and the
       metadata you attached to it (``subject``, ``run``, ``trial``,
       ``intensity``, ``rating``, ``painful``); everything after that is a
       feature, named for what it measures.
   * - ``*_features_coverage.tsv``
     - The same shape, saying what fraction of each cell was finite.
   * - ``*_features.json``
     - The sidecar. Every column's measure, band, space, window, unit,
       normalization, full computation parameters and hash, plus flags and the
       provenance of the run.
   * - ``*_crosstrial.tsv``
     - Measures defined across trials rather than within one — here inter-trial
       phase coherence. One row per trial group, kept out of the per-epoch table
       on purpose (:ref:`why <concepts-row-kinds>`).
   * - ``*_crosstrial_coverage.tsv``
     - Finite-data coverage for the cross-trial table.
   * - ``*_crosstrial.json``
     - The cross-trial metadata and provenance sidecar.

Reading ``sub-01_task-pain_run-01_features.json`` alongside its TSV is the fastest
way to understand :ref:`what a column name means <concepts-naming>` in practice.

What modeling writes
--------------------

Leave-one-subject-out ridge regression predicting ``rating`` from the band power
and ERDS columns, scored per subject and tested against 200 within-subject
permutations.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - What it contains
   * - ``example_model_scores.tsv``
     - The cohort result with its confidence interval and permutation *p*, then
       each subject's own correlation.
   * - ``example_model_predictions.tsv``
     - Every held-out prediction, with the fold it came from, so you can plot or
       re-score it yourself.

.. note::

   The numbers are what this simulation deserves, not a benchmark: alpha really is
   suppressed in proportion to stimulus intensity here, and the rating carries a
   lot of noise that no EEG feature could explain.

The workflow that produces them is documented in :doc:`/guides/modeling`.
