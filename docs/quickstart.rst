Quick Start
===========

.. raw:: html

   <p class="hero-lede">
     Operational walkthrough mapping MNE data structures to structured <code>FeatureTable</code>
     outputs. Covers spectral integration, support-restricted time-frequency windowing,
     baseline-referenced burst detection, and metadata queries.
   </p>

.. _qs-mental-model:

The Mental Model
----------------

``eegfeat`` computes no time-frequency transforms or filters of its own. You bring standard
MNE objects (``Spectrum``, ``EpochsTFR``, or ``Epochs``), wrap them into strongly validated
containers, and pass them to pure feature extractor functions:

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

----

Workflow 1: Spectral Features from PSD
---------------------------------------

When working with stationary resting-state or whole-epoch recordings, start from an MNE
``Spectrum`` (e.g. computed via Welch or multitaper):

.. code-block:: python

   import mne
   import eegfeat as ef

   # 1. Load your preprocessed epochs
   epochs = mne.read_epochs("sample-epo.fif", preload=True)

   # 2. Compute PSD via MNE
   spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, n_fft=1024)

   # 3. Wrap into an eegfeat Spectra container
   spectra = ef.Spectra.from_spectrum(
       spectrum,
       recording="sample-epo.fif",
       estimator_parameters={
           "method": "welch",
           "fmin": 1.0,
           "fmax": 45.0,
           "n_fft": 1024,
       },
   )

   # 4. Integrate the PSD over exact numerical band boundaries
   bands = [
       ef.Band("theta", 4.0, 8.0),
       ef.Band("alpha", 8.0, 13.0),
       ef.Band("beta", 13.0, 30.0),
   ]
   power_table = ef.integrated_band_power(spectra, bands=bands, include_global=True)

   # 5. Extract aperiodic-whitened peak frequency
   # Divides out fitted 1/f background, applies parabolic interpolation,
   # and gates by prominence with automatic centre-of-gravity fallback
   peak_table = ef.peak_frequency(spectra, band=ef.Band("alpha", 8.0, 13.0))

   # 6. Extract spectral descriptors
   centroid_table = ef.spectral_centroid(spectra, band=ef.Band("alpha", 8.0, 13.0))
   entropy_table = ef.spectral_entropy(spectra, band=ef.Band("alpha", 8.0, 13.0))

   # 7. Join tables cleanly
   spectral_features = ef.concat([power_table, peak_table, centroid_table, entropy_table])

----

Workflow 2: Time-Frequency Features with Support Restriction
-------------------------------------------------------------

For event-related designs, slicing raw Morlet wavelets without accounting for wavelet length
causes low-frequency power to bleed across window boundaries. ``eegfeat`` eliminates this
with support restriction:

.. code-block:: python

   import numpy as np
   import mne
   from mne.time_frequency import tfr_morlet
   import eegfeat as ef

   # 1. Compute Morlet TFR in MNE
   freqs = np.linspace(4.0, 40.0, num=30)
   n_cycles = 6.0
   tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)

   # 2. Define named time windows
   windows = [
       ef.Window("baseline", -0.5, -0.1),
       ef.Window("early_stim", 0.1, 0.4),
       ef.Window("late_stim", 0.4, 0.8),
   ]

   # 3. Build Spectra with exact Morlet support restriction
   # Coefficients contaminated by temporal edge leakage are automatically masked out
   spectra_tfr = ef.Spectra.from_tfr(
       tfr,
       windows=windows,
       recording="sub-01_task-test",
       n_cycles=n_cycles,
   )

   # 4. Extract baseline-normalized power (log ratio relative to reference window)
   norm_power = ef.mean_tfr_power(
       spectra_tfr,
       bands=[ef.Band("alpha", 8.0, 13.0)],
       baseline="baseline",
       normalize="log_ratio",
   )

----

Workflow 3: Bursts and ERDS Dynamics
------------------------------------

To extract instantaneous amplitude dynamics, oscillatory bursts, and ERDS:

.. code-block:: python

   import eegfeat as ef

   # 1. Filter epochs into an analytic BandSignal (FIR filter + Hilbert transform)
   beta_signal = ef.BandSignal.from_epochs(
       epochs,
       band=ef.Band("beta", 13.0, 30.0),
       pad_sec=0.5,
   )

   base_win = ef.Window("baseline", -0.5, -0.1)
   stim_win = ef.Window("stim", 0.1, 0.6)

   # 2. Extract oscillatory burst metrics
   # Threshold is calibrated strictly on baseline to prevent stimulus circularity
   burst_rate = ef.burst_rate(
       [beta_signal],
       windows=[stim_win],
       baseline=base_win,
       threshold=0.75,
       min_duration_ms=100.0,
   )

   # 3. Extract ERDS magnitude and latencies
   erds_table = ef.erds_mean(
       [beta_signal],
       baseline=base_win,
       windows=[stim_win],
       normalize="percent",
   )
   peak_lat = ef.erds_peak_latency([beta_signal], baseline=base_win, windows=[stim_win])

----

Working with FeatureTables
--------------------------

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
   # Output column names are structured slugs:
   # eeg_power_alpha_cz_all_raw
   # eeg_peak-freq-adjusted_alpha_cz_all_raw

----

Exporting & BIDS-style Table I/O
--------------------------------

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

For in-memory cohort construction, use :func:`eegfeat.stack_rows` on compatible per-epoch
tables. It preserves input order and rejects duplicate row identities, schema mismatches, and
cross-trial group tables:

.. code-block:: python

   cohort_features = ef.stack_rows([sub_01_features, sub_02_features, sub_03_features])

The target frame used by :func:`eegfeat.model.build_design` must carry matching
``recording``, ``epoch``, and ``event`` keys plus the target and grouping columns. See
:doc:`modeling` for the complete cross-fitting workflow.
