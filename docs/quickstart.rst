Quick Start
===========

.. raw:: html

   <p class="hero-lede">
     Three examples, from MNE objects to a <code>FeatureTable</code>. PSD band
     power, Morlet power with wavelet support masking, and burst and ERDS
     measures.
   </p>

Names used below are defined in :doc:`concepts`.

----

Workflow 1. Spectral features from a PSD
----------------------------------------

PSD features start from an MNE ``Spectrum``, for example Welch or multitaper.

For multitaper, pass ``normalization="full"`` to ``compute_psd`` and record
``"normalization": "full"`` in ``estimator_parameters``. Any other multitaper
normalization is rejected, because it is not a density in V²/Hz.

.. code-block:: python

   import mne
   import eegfeat as ef

   epochs = mne.read_epochs("sample-epo.fif", preload=True)

   spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, n_fft=1024)

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

   bands = [
       ef.Band("theta", 4.0, 8.0),
       ef.Band("alpha", 8.0, 13.0),
       ef.Band("beta", 13.0, 30.0),
   ]
   power_table = ef.integrated_band_power(spectra, bands=bands, include_global=True)

   # Divides out a fitted 1/f, interpolates the peak, and falls back to the
   # centre of gravity when prominence is below min_prominence.
   peak_table = ef.peak_frequency(spectra, band=ef.Band("alpha", 8.0, 13.0))

   centroid_table = ef.spectral_centroid(spectra, band=ef.Band("alpha", 8.0, 13.0))
   entropy_table = ef.spectral_entropy(spectra, band=ef.Band("alpha", 8.0, 13.0))

   spectral_features = ef.concat([power_table, peak_table, centroid_table, entropy_table])

----

Workflow 2. Time-frequency power
---------------------------------

Morlet coefficients are kept only where the wavelet support lies inside the
window. A low-frequency coefficient near a window edge is excluded.

.. code-block:: python

   import numpy as np
   import mne
   from mne.time_frequency import tfr_morlet
   import eegfeat as ef

   freqs = np.linspace(4.0, 40.0, num=30)
   n_cycles = 6.0
   tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)

   windows = [
       ef.Window("baseline", -0.5, -0.1),
       ef.Window("early_stim", 0.1, 0.4),
       ef.Window("late_stim", 0.4, 0.8),
   ]

   # sfreq is the rate the TFR was computed at, before decimation.
   spectra_tfr = ef.Spectra.from_tfr(
       tfr,
       windows=windows,
       recording="sub-01_task-test",
       n_cycles=n_cycles,
       sfreq=epochs.info["sfreq"],
   )

   norm_power = ef.mean_tfr_power(
       spectra_tfr,
       bands=[ef.Band("alpha", 8.0, 13.0)],
       baseline="baseline",
       normalize="log_ratio",
   )

----

Workflow 3. Bursts and ERDS
---------------------------

.. code-block:: python

   import eegfeat as ef

   beta_signal = ef.BandSignal.from_epochs(
       epochs,
       band=ef.Band("beta", 13.0, 30.0),
       recording="sample-epo.fif",
       pad_sec=0.5,
   )

   base_win = ef.Window("baseline", -0.5, -0.1)
   stim_win = ef.Window("stim", 0.1, 0.6)

   # threshold is a quantile of the baseline envelope.
   burst_rate = ef.burst_rate(
       [beta_signal],
       windows=[stim_win],
       baseline=base_win,
       threshold=0.75,
       min_duration_ms=100.0,
   )

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

      Query a table, write TSV and JSON, and stack recordings.

   .. grid-item-card:: Command Line Runner
      :link: guides/runner
      :link-type: doc

      Apply one TOML recipe to a folder of epochs files.

   .. grid-item-card:: Predictive Modeling
      :link: guides/modeling
      :link-type: doc

      Grouped cross-fitting, permutation nulls, and conformal intervals.
