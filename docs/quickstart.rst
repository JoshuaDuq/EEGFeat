Quick Start
===========

.. raw:: html

   <p class="hero-lede">
     Three worked workflows, from MNE objects to a <code>FeatureTable</code>: spectral
     integration from a PSD, support-restricted time-frequency windowing, and
     baseline-referenced burst and ERDS dynamics.
   </p>

These assume the vocabulary in :doc:`concepts` — containers, bands and windows,
and what a feature table holds. Skim that page first if any of the calls below
look like they are doing something implicit.

----

Workflow 1: Spectral Features from PSD
---------------------------------------

When working with stationary resting-state or whole-epoch recordings, start from an MNE
``Spectrum`` (e.g. computed via Welch or multitaper):

For multitaper, pass ``normalization="full"`` to ``compute_psd`` and include
``"normalization": "full"`` in ``estimator_parameters``. The default MNE
multitaper normalization is not a density in V²/Hz and is rejected.

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
       sfreq=epochs.info["sfreq"],  # the rate before any decim, for V²/Hz scaling
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
       recording="sample-epo.fif",
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

Where to go next
----------------

.. grid:: 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Feature Tables and Files
      :link: guides/tables
      :link-type: doc

      Query the tables you just built, write them to TSV with their sidecars, and
      stack several recordings into a cohort.

   .. grid-item-card:: Command Line Runner
      :link: guides/runner
      :link-type: doc

      Do all of the above for a whole folder from one TOML recipe, without
      writing Python.

   .. grid-item-card:: Predictive Modeling
      :link: guides/modeling
      :link-type: doc

      Carry per-epoch tables into group-disjoint cross-fitting, nulls, and
      conformal intervals.
