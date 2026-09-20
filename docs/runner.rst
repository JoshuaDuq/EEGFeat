Command Line Runner
===================

.. raw:: html

   <p class="hero-lede">
     Compute features for a whole folder of preprocessed MNE epochs files from one
     declarative <strong>recipe</strong>, without writing Python. Every table the runner
     writes keeps its column metadata and loads back as a <code>FeatureTable</code>.
   </p>

The library leaves spectral estimation, band filtering and trial grouping to its caller.
The runner is a caller: a recipe states those choices once, and ``eegfeat run`` applies
them to every recording.

Quick Start
-----------

.. code-block:: bash

   eegfeat init recipe.toml     # a commented recipe to start from
   eegfeat check recipe.toml    # validate it, then try it on the first recording
   eegfeat run recipe.toml      # compute features for every recording

``python -m eegfeat`` is the same command. ``check`` writes nothing: it catches what
loading a recipe cannot, such as an ROI naming a channel the data lacks, a window
outside the epochs, or a Morlet wavelet longer than the epoch.

The Recipe
----------

A recipe is a TOML file. Relative paths resolve against its directory, unknown keys
are errors, and every problem found is reported at once.

.. code-block:: toml

   [inputs]
   root = "derivatives/preprocessed/eeg"
   pattern = "**/*_proc-clean_epo.fif"

   [output]
   root = "derivatives/eegfeat"

   [windows]
   baseline = [-5.0, -1.0]
   stimulus = [0.0, 8.0]

   [trials]
   by = "event"

   [[features]]
   measure = "integrated_band_power"
   normalize = "log10"
   ratios = [["theta", "beta"]]

   [[features]]
   measure = "erds_mean"
   bands = ["alpha", "beta"]
   baseline = "baseline"
   spatial = ["global"]

   [[features]]
   measure = "itpc"
   bands = ["theta"]

.. list-table:: Sections
   :header-rows: 1
   :widths: 18 82

   * - Section
     - Keys and defaults
   * - ``[inputs]``
     - ``root`` (required); ``pattern`` = ``"**/*_epo.fif"``; ``picks`` = ``"eeg"`` (a
       channel type or a list of names); ``exclude_bads`` = ``true``. Hidden files are
       skipped, including the ``._`` files macOS leaves on external drives.
   * - ``[output]``
     - ``root`` (required); ``epoch_metadata`` = ``true`` copies each epoch's metadata
       into its row.
   * - ``[bands]``
     - ``name = [low, high]`` in Hz, half-open. Default: delta to gamma, 1–45 Hz.
   * - ``[windows]``
     - ``name = [tmin, tmax]`` in seconds. Default: the whole epoch, named ``all``,
       which is reserved.
   * - ``[rois]``
     - ``name = ["Ch1", "Ch2", ...]``. ``global`` is reserved.
   * - ``[spectra]``
     - ``method`` = ``"welch"``, ``"multitaper"`` or ``"morlet"``; ``fmin``/``fmax``
       default to the span of the bands. See below.
   * - ``[band_signal]``
     - ``pad_sec`` = 0.5, ``pad_cycles`` = 3.0, passed to :meth:`eegfeat.BandSignal.from_epochs`.
   * - ``[trials]``
     - How cross-trial measures group epochs: ``by`` = ``"all"`` (one group),
       ``"event"`` (by event name) or ``"metadata"`` with a ``column``.
   * - ``[microstates]``
     - Parameters of :func:`eegfeat.segment`, fitted once per recording and shared by
       every microstate measure.
   * - ``[[features]]``
     - One entry per measure. See below.

Feature Entries
~~~~~~~~~~~~~~~

``measure`` names a library function. Every other key is either one of that function's
own keyword parameters, checked against its annotations (``normalize``, ``fit_range``,
``threshold``, ``polarity``, ...), or one of the runner's keys:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``bands``
     - Band names. Default: every band. Single-band measures such as
       ``peak_frequency`` run once per band.
   * - ``windows``
     - Window names. Default: every window except the entry's baseline.
   * - ``baseline``
     - A window name. Power measures consume it for normalization; burst and ERDS
       measures calibrate against it.
   * - ``spatial``
     - Any of ``"channels"``, ``"rois"``, ``"global"``. Default ``["channels", "global"]``;
       for pairwise connectivity the levels are node sets, default ``["channels"]``.
   * - ``series``
     - Time-domain and complexity measures: ``"broadband"`` and band names (band
       envelopes). Default ``["broadband"]``.
   * - ``pairs``
     - ``pac`` only: ``[["theta", "gamma"], ...]`` as ``[phase, amplitude]``. PAC
       columns are named by amplitude band, so pairs in one entry need distinct ones.
   * - ``ratios``, ``asymmetry``
     - Power measures only: ``[[numerator, denominator], ...]`` band pairs and
       ``[[left, right], ...]`` channel pairs, through :func:`eegfeat.band_ratio` and
       :func:`eegfeat.asymmetry`.
   * - ``graph``, ``clustering_threshold``
     - ``envelope_correlation`` and ``wpli`` only: ``["global_efficiency",
       "clustering_coefficient"]``; the threshold is required for clustering.

Spectra
~~~~~~~

``welch``
   Hann-tapered segments of ``n_fft`` samples with 50% overlap, computed separately in
   each window. The default segment is 2 s, capped by the shortest window any spectral
   entry uses, so every window shares one frequency grid.
``multitaper``
   Computed in each window, with an optional ``bandwidth``. The grid follows the window
   length, so windows measured together must be equally long.
``morlet``
   One time-frequency decomposition per recording on ``n_freqs`` = 40 frequencies
   (``spacing`` = ``"log"``), with ``n_cycles = clip(f / n_cycles_factor, min_cycles,
   max_cycles)`` (2.0, 3.0, 15.0) and ``decim`` = 4, matching the reference pipeline.
   Each window keeps only the coefficients its wavelets can account for. The lowest
   wavelet must fit inside the epoch.

Outputs
-------

The input tree is mirrored under the output root. For
``sub-01/eeg/sub-01_task-rest_epo.fif``:

.. code-block:: text

   sub-01/eeg/sub-01_task-rest_features.tsv            one row per epoch
   sub-01/eeg/sub-01_task-rest_features_coverage.tsv
   sub-01/eeg/sub-01_task-rest_features.json           column metadata, flags, provenance
   sub-01/eeg/sub-01_task-rest_crosstrial.tsv          one row per trial group
   sub-01/eeg/sub-01_task-rest_crosstrial_coverage.tsv
   sub-01/eeg/sub-01_task-rest_crosstrial.json
   eegfeat_run.json                                    what happened to every recording

Measures estimated within an epoch go to ``_features``. Its rows start with ``epoch``,
``selection`` and ``event``, then the epoch metadata, then the features. Measures
estimated across trials (``itpc``, ``envelope_correlation``, ``wpli`` and their graph
summaries) go to ``_crosstrial``, keyed by ``group``. The two are never merged; see
:func:`eegfeat.itpc` for why. Missing values are ``n/a``.

Load a table back with its metadata:

.. code-block:: python

   import eegfeat as ef
   from eegfeat.io import read_table

   table = read_table("derivatives/eegfeat/sub-01/eeg/sub-01_task-rest_features.tsv")
   alpha = table.select(band=ef.Band("alpha", 8.0, 13.0), space="global")

For predictive modeling across recordings, pass several per-epoch ``*_features.tsv`` paths
to :func:`eegfeat.io.read_dataset`. The loader stacks the tables onto the union of their
feature columns and restores the descriptor columns written by the runner alongside canonical
``recording``, ``epoch``, and ``event`` keys. Include target and grouping variables in the epoch metadata if they are needed
by :func:`eegfeat.model.build_design`; descriptor columns are not feature columns.

.. code-block:: python

   from eegfeat.io import read_dataset

   dataset = read_dataset([
       "derivatives/eegfeat/sub-01/eeg/sub-01_task-rest_features.tsv",
       "derivatives/eegfeat/sub-02/eeg/sub-02_task-rest_features.tsv",
   ])

A run refuses to write over earlier results unless given ``--overwrite``, which also
removes result files the new recipe no longer produces. A recording that fails is
logged with its error and traceback, and the run moves on.

Command Reference
-----------------

.. code-block:: text

   eegfeat run RECIPE [--overwrite] [--n-jobs N] [--progress-json]
   eegfeat check RECIPE [--n-jobs N]
   eegfeat init [PATH]

``--n-jobs`` is passed to MNE's filtering and spectral estimation. Exit status is 0 when
everything succeeded, 1 when some recordings failed (or ``check``'s trial did), and 2 when
nothing could start.

``--progress-json`` prints one JSON object per line: ``start``, ``subject_start``,
``progress`` (``step``, ``current``, ``total``), ``log``, ``subject_done`` (``success``),
``complete`` and ``error``. This is the protocol of the EEG_fMRI_Pipeline terminal UI, with
each recording standing in for a subject.

Things to Know
--------------

- With ``exclude_bads``, each recording keeps its own good channels, so channel-level
  columns differ between recordings. Interpolate bad channels upstream, or use ROI and
  global levels, for one column set across a study.
- Band signals are cached per recording for reuse across entries. On 22 s epochs of
  60 channels at 500 Hz, spectral measures peaked near 1.4 GB and each band-signal band
  added about 1.3 GB.
- Microstate templates are fitted per recording, not across a group.
- Only FIF epochs files (``mne.read_epochs``) are read.
