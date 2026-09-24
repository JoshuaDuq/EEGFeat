Command Line Runner
===================

.. raw:: html

   <p class="hero-lede">
     Compute features for a folder of preprocessed MNE epochs files from one
     TOML recipe. Each written table loads back as a <code>FeatureTable</code>.
   </p>

A recipe sets spectral estimation, band-pass filtering, and trial grouping.
``eegfeat run`` applies that recipe to every recording.

Quick Start
-----------

.. code-block:: bash

   eegfeat init recipe.toml --template task   # a commented recipe to start from
   eegfeat check recipe.toml                  # validate it, time it on the first recording
   eegfeat run recipe.toml --workers 4        # compute features for every recording
   eegfeat status recipe.toml                 # which recordings are done, and what to run next

``python -m eegfeat`` is the same command. ``init`` has three templates: ``basic`` (the
default, a few spectral measures), ``task`` (every measure family, for epochs around an
event, with a baseline and a response window) and ``resting`` (every family that needs no
event). ``check`` writes nothing. It loads the recipe and runs the first recording, which
catches an ROI that names a missing channel, a window outside the epochs, or a Morlet
wavelet longer than the epoch, and it times that recording: see `Checking a Recipe`_.

The Recipe
----------

A recipe is a TOML file. Relative paths resolve against the recipe's directory.
Unknown keys are errors. Every problem found is reported together.

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
     - ``name = ["Ch1", "Ch2", ...]``, or ``name = { match = ["^P[z0-9]+$"] }`` for
       regular expressions matched against each recording's channels, which keep the
       recording's order. An ROI whose patterns match no channel fails that recording, and
       ``check`` names it. ``global`` is reserved.
   * - ``[spectra]``
     - ``method`` = ``"welch"``, ``"multitaper"`` or ``"morlet"``; ``fmin``/``fmax``
       default to the span of the bands. See below.
   * - ``[band_signal]``
     - ``pad_sec`` = 0.5, ``pad_cycles`` = 3.0, passed to :meth:`eegfeat.BandSignal.from_epochs`.
   * - ``[trials]``
     - How cross-trial measures group epochs: ``by`` = ``"all"`` (one group),
       ``"event"`` (by event name) or ``"metadata"`` with a ``column``.
   * - ``[microstates]``
     - Parameters of :func:`~eegfeat.microstates.segment`, fitted once per recording and shared by
       every microstate measure.
   * - ``[defaults]``
     - ``bands``, ``windows``, ``spatial`` and ``series`` for every entry that takes them
       and does not set its own. See below.
   * - ``[[features]]``
     - One entry per measure. See below.

Feature Entries
~~~~~~~~~~~~~~~

``measure`` names a library function. ``measures = [...]`` names several instead, and the
entry's other keys apply to each of them, as if it were written once per measure; a key
one of them does not take is an error for that measure. Every other key is either one of
that function's own keyword parameters, checked against its annotations (``normalize``,
``fit_range``, ``threshold``, ``polarity``, ...), or one of the runner's keys:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``bands``
     - Band names. Default: every band. Single-band measures such as
       ``peak_frequency`` run once per band.
   * - ``windows``
     - Window names, and ``"all"`` for the whole epoch. Default: every window except the
       entry's baseline.
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

Shared Defaults
~~~~~~~~~~~~~~~

A ``[defaults]`` section sets ``bands``, ``windows``, ``spatial`` or ``series`` once. An
entry inherits each key it takes and leaves out, so a default ``bands`` does not reach
``aperiodic``, and a default ``spatial`` keeps only the levels a measure has
(``envelope_correlation`` has no ``global``). An entry with a baseline leaves that window
out of the inherited windows, as leaving ``windows`` out does. Keys an entry sets win.

.. code-block:: toml

   [defaults]
   windows = ["baseline", "response"]
   spatial = ["channels", "global"]

   [[features]]
   measures = ["variance", "line_length", "kurtosis", "hjorth_mobility"]

   [[features]]
   measures = ["erds_mean", "erd_magnitude", "ers_magnitude"]
   bands = ["alpha", "beta"]
   baseline = "baseline"      # computed on "response" only

Spectra
~~~~~~~

``welch``
   Hann-tapered segments of ``n_fft`` samples with 50% overlap, computed separately in
   each window. The default segment is 2 s, capped by the shortest window any spectral
   entry uses, so every window shares one frequency grid.
``multitaper``
   Computed in each window with a frequency-smoothing ``bandwidth`` in Hz. The
   default is 2.0 Hz. MNE's own default is ``8 / window_length`` Hz, which on a
   1 s window smooths by ±4 Hz, wider than delta or theta. A bandwidth below
   ``1.35 / window_length`` is rejected: in a narrower band no Slepian taper keeps
   90% of its power, and MNE would fall back to a leaky one with only a warning.
   The frequency grid follows the window length, so windows measured together
   must be the same length. The call uses MNE ``normalization="full"``, and the
   output is a density in V²/Hz.
``morlet``
   One time-frequency decomposition per recording. Defaults are ``n_freqs`` = 40,
   ``spacing`` = ``"log"``, ``n_cycles = clip(f / n_cycles_factor, min_cycles,
   max_cycles)`` with factors (2.0, 3.0, 15.0), and ``decim`` = 4. Each window
   keeps the coefficients whose wavelet support lies inside it. The lowest
   wavelet must fit inside the epoch. Power is divided by the sampling rate of
   the recording. ``mean_tfr_power`` is then a smoothed density in V²/Hz. MNE's
   raw wavelet power scales with the sampling rate.

Checking a Recipe
-----------------

``check`` computes the first recording and reports how long it took, which measures were
slowest, and a projection for the whole run:

.. code-block:: text

   Time           4 min for this recording
   Slowest        sample_entropy 2 min (40%) · multiscale_entropy 60 s (25%) · ...
   Projected      126 recordings ≈ 8 h 24 min one at a time, ≈ 52 min with --workers 10,
                  if the others are like it

A measure's time includes building the inputs it is the first to need (spectra, band
signals, the microstate fit), so shared costs show up on the first entry that uses them.
The projection assumes each worker gets a full core; on processors with efficiency cores,
or when memory bandwidth is shared, parallel runs take longer than that. ``--workers N``
sets the worker count it assumes (default: one per core, up to the number of recordings).

``check --quick`` computes only the first four epochs and scales the timing up by the epoch
count. It finds recipe errors in seconds, but its projection is rougher, and cross-trial
measures see fewer trials than a run gives them.

Running in Parallel
-------------------

``eegfeat run --workers N`` computes N recordings at once, each in its own process. Results
are the same as one at a time, and the run log lists recordings in input order. Each
worker's BLAS and OpenMP thread pools are sized to its share of the cores
(``OMP_NUM_THREADS`` and the like), unless those variables are already set. Memory grows
with the workers: each holds its recording's spectra and band signals (see
`Things to Know`_).

If a worker process dies (the operating system killing it for memory, a crash in compiled
code), every recording in flight is rerun alone, and only the one that dies again is
recorded as failed; the run goes on.

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
   sub-01/eeg/sub-01_task-rest_failed.json             only while the recording fails: error, traceback
   eegfeat_run.json                                    what happened to every recording

Measures estimated within an epoch go to ``_features``. Its rows start with ``epoch``,
``selection`` and ``event``, then the epoch metadata, then a ``__eegfeat_row_id``
column that ``read_table`` checks against the sidecar, then the features. Measures
estimated across trials (``itpc``, ``ppc``, ``envelope_correlation``, ``wpli`` and their graph
summaries) go to ``_crosstrial``, keyed by ``group``. The two files are separate.
See :ref:`concepts-row-kinds`. Missing values are written ``n/a``.

Load a table back with its metadata:

.. code-block:: python

   import eegfeat as ef
   from eegfeat.io import read_table

   table = read_table("derivatives/eegfeat/sub-01/eeg/sub-01_task-rest_features.tsv")
   alpha = table.select(band=ef.Band("alpha", 8.0, 13.0), space="global")

For modeling across recordings, pass several per-epoch ``*_features.tsv`` paths to
:func:`eegfeat.io.read_dataset`. The loader stacks tables on the union of their
feature columns and restores the descriptor columns written by the runner, plus
``recording``, ``epoch``, and ``event`` from the stored row identity. Put target
and grouping variables in the epoch metadata when :func:`eegfeat.model.build_design`
needs them. Descriptor columns are separate from feature columns.

.. code-block:: python

   from eegfeat.io import read_dataset

   dataset = read_dataset([
       "derivatives/eegfeat/sub-01/eeg/sub-01_task-rest_features.tsv",
       "derivatives/eegfeat/sub-02/eeg/sub-02_task-rest_features.tsv",
   ])

A run refuses to write over earlier results unless given ``--overwrite``, which also
removes result files the new recipe no longer produces. A recording that fails is
logged with its error and traceback, and the run moves on. ``_failed.json`` records the
failure beside the results it did not produce, and a later success removes it. The run log
is rewritten after every recording, with ``finished`` false until the run ends, so a run
that is interrupted still says what it did.

Resuming
--------

``eegfeat status`` reads the results and the run log back, computing nothing, and puts
each recording in one of five states:

``done``
   Every table is complete, was computed by this recipe, and is newer than its input.
``missing``
   No results.
``failed``
   No results, and the last run failed on it. The reason is that run's error, read from
   the recording's ``_failed.json``, so it survives another run into the same output root.
``stale``
   The results were computed by a different recipe, or the input file changed after they
   were written.
``partial``
   Files of a run are gone: a table lacks its coverage file or sidecar, or a table the
   run wrote is missing.

"This recipe" means what the recipe computes. Comments, formatting, and the three
location keys (``inputs.root``, ``inputs.pattern``, ``output.root``) are left out, so
moving the data and repointing ``inputs.root`` keeps results ``done``. Results written
before EEGFeat recorded this fingerprint count as current only while the recipe file is
unchanged byte for byte.

``eegfeat run --resume`` computes only the recordings that are not ``done``. Recordings
that have stale or partial results still need ``--overwrite``, so nothing is replaced
unasked. ``status`` ends with the command to run next, and ``--json`` gives the same
report as one object, ``{recipe, output_root, counts, recordings, next}``, where ``next``
is that command's arguments or ``null``.

Command Reference
-----------------

.. code-block:: text

   eegfeat run RECIPE [--overwrite] [--resume] [--workers N] [--n-jobs N] [--progress-json]
   eegfeat check RECIPE [--quick] [--workers N] [--n-jobs N]
   eegfeat status RECIPE [--json]
   eegfeat init [PATH] [--template basic|task|resting]

``--n-jobs`` is passed to MNE's filtering and spectral estimation. Exit status is 0 when
everything succeeded, 1 when some recordings failed (or ``check``'s trial did, or it found
recordings lacking channels the recipe names), and 2 when
nothing could start. ``status`` exits 0 whatever state the recordings are in.

``--progress-json`` prints one JSON object per line: ``start``, ``subject_start``,
``progress`` (``step``, ``current``, ``total``), ``log``, ``subject_done`` (``success``,
``elapsed``, ``eta``), ``complete`` and ``error``. This is the protocol of the
EEG_fMRI_Pipeline terminal UI, with each recording standing in for a subject. ``eta`` is
seconds left at the pace so far, null after the last recording. With ``--workers``,
recordings start and finish in any order, and each event names its recording. The text
output shows the same estimate after each recording, and names a recording whose result
does not directly follow its start line.

Things to Know
--------------

- With ``exclude_bads``, each recording keeps its own good channels, so channel-level
  columns differ between recordings. Interpolate bad channels upstream, or use ROI and
  global levels, for one column set across a study.
- Band signals are cached per recording for reuse across entries. On 22 s epochs of
  60 channels at 500 Hz, spectral measures peaked near 1.4 GB and each band-signal band
  added about 1.3 GB. With ``--workers``, each worker needs that much.
- Microstate templates are fitted per recording, not across a group.
- Only FIF epochs files (``mne.read_epochs``) are read.
