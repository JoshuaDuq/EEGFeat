Concepts
========

.. raw:: html

   <p class="hero-lede">
     Containers, <code>FeatureTable</code> fields, column names, cross-trial
     tables, and missing values.
   </p>

.. _concepts-mental-model:

From MNE objects to feature tables
----------------------------------

``eegfeat`` accepts MNE ``Spectrum``, ``EpochsTFR``, and ``Epochs`` objects.
Spectra and time-frequency representations are computed in MNE, then wrapped.
:class:`~eegfeat.BandSignal` is the exception.
:meth:`~eegfeat.BandSignal.from_epochs` applies its documented band-pass and
Hilbert transform.

.. grid:: 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: 1. Wrap MNE Objects

      Convert MNE outputs into :class:`~eegfeat.Spectra`, :class:`~eegfeat.Signal`,
      or :class:`~eegfeat.BandSignal`.

   .. grid-item-card:: 2. Extract Features

      Call an extractor with explicit bands, windows, and regions of interest.

   .. grid-item-card:: 3. Query and Export

      Filter columns with :meth:`~eegfeat.FeatureTable.select`, read
      :attr:`~eegfeat.FeatureTable.coverage`, or export with
      :meth:`~eegfeat.FeatureTable.to_dataframe`.

The spectral estimator, the filter, and the trial grouping are arguments of the
wrapper, or keys in a runner recipe. See :doc:`/guides/runner`.

.. _concepts-containers:

The containers
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Container
     - Contents
   * - :class:`~eegfeat.Spectra`
     - A power spectrum or a time-frequency representation, plus the estimator
       parameters that produced it. The container records which of the two it
       holds. An operation defined for one rejects the other.
   * - :class:`~eegfeat.Signal`
     - Broadband time-domain data.
   * - :class:`~eegfeat.BandSignal`
     - Band-limited analytic signal. ``from_epochs`` filters and takes the
       Hilbert transform.
   * - :class:`~eegfeat.Band`
     - A named half-open frequency interval.
   * - :class:`~eegfeat.Window`
     - A named time interval within the epoch.

Bands and windows are named, not positional. A column records that it is
``alpha`` in ``baseline``, and :meth:`~eegfeat.FeatureTable.select` matches
those fields.

.. _concepts-feature-table:

Anatomy of a feature table
--------------------------

Every extractor returns a :class:`~eegfeat.FeatureTable` with five aligned
parts.

``values``
   Shape ``(n_rows, n_features)``.

``coverage``
   Same shape. Fraction of finite input behind each value, in ``[0, 1]``.
   This is the fraction of the input that was present. It is not an artifact
   score. For Morlet input, :attr:`~eegfeat.Spectra.support` is the fraction of
   the requested window with complete wavelet support.

``meta``
   One :class:`~eegfeat.FeatureMeta` per column. Fields are the measure, band,
   space and its kind (``space_kind``), window and its bounds, normalization,
   unit, source, the frequency resolution, the phase and amplitude bands of a
   coupling measure, the node pair of a pairwise measure, and the
   :class:`~eegfeat.ComputationSpec` that produced the column. Each field is
   constant across rows.

``flags``
   Per-cell boolean annotations, for example ``cog_fallback``, ``edge_hit``,
   and ``aperiodic_fit_failed``. A fact that varies by row belongs here.

``row_ids`` / ``row_labels``
   Row identity. See below.

Slicing, concatenation, and a round trip through disk keep this metadata.
:meth:`~eegfeat.FeatureTable.select` matches the metadata. It does not parse
the column name.

.. _concepts-naming:

Column names
------------

Column names are generated. Each name is six underscore-separated fields and a
hash suffix.

.. code-block:: text

   eeg_band-power_alpha_cz_all_raw_p8f3c1d5e9a02
   \_/ \________/ \___/ \/ \_/ \_/ \____________/
    |       |       |   |   |   |         |
    |    measure   band |   |   |    first 12 hex of the SHA-256
  domain            space   |  normalization   of the complete column spec
                          window

Underscores and spaces inside a field are replaced by hyphens, so the split
into six fields is unambiguous. A missing band is written ``broadband``. A
missing window is written ``all``.

The hash is the SHA-256 of the full column specification, not only the
arguments passed to the extractor. Two columns that differ only in a parameter
such as ``fit_range`` or a burst threshold share the six readable fields and
differ in the hash. Both can sit in one table.

.. _concepts-row-kinds:

Epoch rows and group rows
-------------------------

There are two kinds of table. They are not merged.

**Per-epoch tables** have one row per epoch. ``row_ids`` is a
``(recording, epoch index, event)`` triple per row. Concatenation across
recordings uses that triple. :func:`eegfeat.model.build_design` accepts only
per-epoch tables.

**Group-row tables** come from measures that are undefined on one trial.
Those measures are inter-trial phase coherence, pairwise phase consistency,
envelope correlation (per-trial correlations averaged in Fisher :math:`z`),
every ``spectral_connectivity`` method including wPLI, and their graph
summaries. The value describes a
set of trials. The table carries ``row_labels`` for those groups and cannot
carry ``row_ids``.

Copying a group value onto its member epochs repeats one number across rows.
A model then treats those rows as independent observations. The runner writes
the two kinds of table to separate files. See :doc:`/guides/runner`.

.. _concepts-missing:

Missing values
--------------

``NaN`` in ``values`` means the value was withheld because of the input.
Non-finite samples are treated as missing. A measure that cannot be estimated
returns ``NaN`` and records the condition in ``coverage`` and ``flags``.

A finite value can still come from a small fraction of its input. Read
``coverage`` with the value. ``max_feature_missingness`` in
:class:`~eegfeat.model.PreprocessingConfig` is a separate check. It drops a
column when the fraction of ``NaN`` rows exceeds the threshold. It does not
read ``coverage``.
