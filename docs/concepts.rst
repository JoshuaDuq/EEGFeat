Concepts
========

.. raw:: html

   <p class="hero-lede">
     The five ideas the rest of the documentation assumes: what the containers are
     for, what a <code>FeatureTable</code> guarantees, how columns are named, why
     cross-trial measures live in their own table, and what a missing value means.
   </p>

.. _concepts-mental-model:

From MNE objects to feature tables
----------------------------------

``eegfeat`` accepts standard MNE objects (``Spectrum``, ``EpochsTFR``, or ``Epochs``), wraps
them into strongly validated containers, and passes those containers to feature extractor
functions. Spectra and time-frequency representations are precomputed with MNE;
:class:`~eegfeat.BandSignal` can apply its documented band-pass and Hilbert transform:

.. grid:: 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: 1. Wrap MNE Objects

      Convert MNE outputs into :class:`~eegfeat.Spectra`, :class:`~eegfeat.Signal`,
      or :class:`~eegfeat.BandSignal`.

   .. grid-item-card:: 2. Extract Features

      Call pure extractor functions with explicit parameterization, windows, and ROIs.

   .. grid-item-card:: 3. Query & Export

      Filter columns via :meth:`~eegfeat.FeatureTable.select`, check :attr:`~eegfeat.FeatureTable.coverage`,
      or export via :meth:`~eegfeat.FeatureTable.to_dataframe`.

The library deliberately does not choose your spectral estimator, your filter, or your
trial grouping. Those are the caller's decisions, stated explicitly at the point of
wrapping. The :doc:`/guides/runner` is a caller that makes them once, in a recipe, and
applies them to a whole folder.

.. _concepts-containers:

The containers
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Container
     - What it holds
   * - :class:`~eegfeat.Spectra`
     - A power spectrum or a time-frequency representation, with the estimator
       parameters that produced it. It records which of the two it contains, so
       operations defined for one reject the other rather than conflating units.
   * - :class:`~eegfeat.Signal`
     - Broadband time-domain data, for descriptors of the waveform itself.
   * - :class:`~eegfeat.BandSignal`
     - Band-limited analytic signal — the only container that will filter and
       Hilbert-transform for you, because doing so is part of its definition.
   * - :class:`~eegfeat.Band`
     - A named half-open frequency interval.
   * - :class:`~eegfeat.Window`
     - A named time interval within the epoch.

Bands and windows are named, not positional, everywhere. A column knows it is
``alpha`` in ``baseline``, which is what makes a table selectable after the fact.

.. _concepts-feature-table:

Anatomy of a feature table
--------------------------

Every extractor returns a :class:`~eegfeat.FeatureTable`, which is five aligned
things rather than one array:

``values``
   Shape ``(n_rows, n_features)``. The numbers.

``coverage``
   The same shape. The fraction of numerically finite input that produced each
   value, in ``[0, 1]``. This is **not** an artifact-free-data score — it says how
   much of the input existed, not how clean it was. For Morlet input,
   :attr:`~eegfeat.Spectra.support` separately records the fraction of the
   requested window that had complete wavelet support.

``meta``
   One :class:`~eegfeat.FeatureMeta` per column, describing the measure, band,
   space, window, normalization, unit, source, and the full
   :class:`~eegfeat.ComputationSpec` that produced it. Every field is constant
   across rows by construction.

``flags``
   Per-cell boolean annotations — ``cog_fallback``, ``edge_hit``,
   ``aperiodic_fit_failed``. Facts that vary by row belong here, never in ``meta``.

``row_ids`` / ``row_labels``
   How the rows are identified. See below.

The consequence worth internalizing: a table that has been sliced, concatenated,
written to disk and read back still knows what each of its numbers is. Selection
by :meth:`~eegfeat.FeatureTable.select` matches on structured metadata, not on
string-matching the column name.

.. _concepts-naming:

What a column name means
------------------------

Column names are generated, never hand-written. They are exactly six
underscore-separated fields plus a hash suffix:

.. code-block:: text

   eeg_band-power_alpha_cz_all_raw_p8f3c1d5e9a02
   \_/ \________/ \___/ \/ \_/ \_/ \____________/
    |       |       |   |   |   |         |
    |    measure   band |   |   |    first 12 hex of the SHA-256
  domain            space   |  normalization   of the complete column spec
                          window

Field values have their own underscores and spaces replaced by hyphens, so the
split into six fields is unambiguous. A missing band renders as ``broadband`` and
a missing window as ``all``.

The hash matters. Two columns that differ only in a computation parameter — a
different ``fit_range``, a different burst threshold — produce the same six
readable fields and different hashes, so they can coexist in one table and cannot
be mistaken for each other. It digests the complete column specification, not just
the parameters you passed.

.. _concepts-row-kinds:

Epoch rows and group rows
-------------------------

Two kinds of table exist, and they are never merged.

**Per-epoch tables** have one row per epoch and carry ``row_ids``: a
``(recording, epoch index, event)`` triple for each row. This is what makes
concatenation across recordings safe, and it is the only kind
:func:`eegfeat.model.build_design` accepts.

**Group-row tables** come from measures that are undefined within a single trial —
inter-trial phase coherence, phase locking, envelope correlation, wPLI, and their
graph summaries. A phase coherence value describes a *set* of trials. These tables
carry ``row_labels`` naming their trial groups, and are forbidden from carrying
``row_ids`` at all.

The separation is enforced rather than advised, because broadcasting a group value
back onto its constituent epochs manufactures pseudo-replication: the same number
repeated across rows that a model will treat as independent observations. The
runner writes the two kinds to separate files for the same reason — see
:doc:`/guides/runner`.

.. _concepts-missing:

Missing values
--------------

``NaN`` in ``values`` means the value was withheld because of a data condition, not
that the computation produced a nonsensical number. Non-finite input is treated as
missing throughout rather than propagated: a measure that cannot be estimated from
the data it was given returns ``NaN`` and reports what happened in ``coverage`` and
``flags``, instead of returning a number whose provenance you would have to
reconstruct.

This is why ``coverage`` is worth reading before trusting a cell. A value computed
from 3% of its intended input is a number, and it is a number you probably do not
want in a design matrix; ``max_feature_missingness`` in
:class:`~eegfeat.model.PreprocessingConfig` exists to act on exactly this.
