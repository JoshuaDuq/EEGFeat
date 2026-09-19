API Reference
=============

.. raw:: html

   <p class="hero-lede">
     Complete API documentation for all public data containers, spectral descriptors,
     dynamics estimators, and connectivity metrics in <code>eegfeat</code>.
   </p>

Core Data Structures
--------------------

.. autoclass:: eegfeat.FeatureTable
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.FeatureMeta
   :members:
   :show-inheritance:

.. autofunction:: eegfeat.concat

.. autofunction:: eegfeat.io.read_table

.. autofunction:: eegfeat.io.write_table

.. autoclass:: eegfeat.Spectra
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Window
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Band
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.Signal
   :members:
   :show-inheritance:

.. autoclass:: eegfeat.BandSignal
   :members:
   :show-inheritance:

Spectral Features
-----------------

.. autofunction:: eegfeat.integrated_band_power
.. autofunction:: eegfeat.mean_psd
.. autofunction:: eegfeat.mean_tfr_power

.. autofunction:: eegfeat.band_ratio

.. autofunction:: eegfeat.asymmetry

.. autofunction:: eegfeat.peak_frequency

.. autofunction:: eegfeat.spectral_centroid

.. autofunction:: eegfeat.spectral_bandwidth

.. autofunction:: eegfeat.spectral_edge

.. autofunction:: eegfeat.spectral_entropy

.. autofunction:: eegfeat.aperiodic

.. autofunction:: eegfeat.aperiodic_ratio

Time-Domain Measures
--------------------

.. autofunction:: eegfeat.variance

.. autofunction:: eegfeat.mean_amplitude

.. autofunction:: eegfeat.peak_to_peak

.. autofunction:: eegfeat.area_under_curve

.. autofunction:: eegfeat.peak_amplitude

.. autofunction:: eegfeat.peak_latency

Oscillatory Bursts
------------------

.. autofunction:: eegfeat.burst_count

.. autofunction:: eegfeat.burst_rate

.. autofunction:: eegfeat.burst_duration

.. autofunction:: eegfeat.burst_amplitude

.. autofunction:: eegfeat.fraction_above_threshold

ERDS Dynamics
-------------

.. autofunction:: eegfeat.erds_mean

.. autofunction:: eegfeat.erds_slope

.. autofunction:: eegfeat.erd_magnitude

.. autofunction:: eegfeat.erd_duration

.. autofunction:: eegfeat.ers_magnitude

.. autofunction:: eegfeat.ers_duration

.. autofunction:: eegfeat.erds_peak_latency

.. autofunction:: eegfeat.erds_onset_latency

.. autofunction:: eegfeat.erds_rebound_latency

Phase & Connectivity
--------------------

.. autofunction:: eegfeat.itpc

.. autofunction:: eegfeat.ppc

.. autofunction:: eegfeat.pac

.. autofunction:: eegfeat.envelope_correlation

.. autofunction:: eegfeat.wpli

.. autofunction:: eegfeat.global_efficiency

.. autofunction:: eegfeat.clustering_coefficient

Complexity & Entropy
--------------------

.. autofunction:: eegfeat.sample_entropy

.. autofunction:: eegfeat.multiscale_entropy

Microstates
-----------

.. autofunction:: eegfeat.microstates.segment

.. autofunction:: eegfeat.microstates.microstate_coverage

.. autofunction:: eegfeat.microstates.microstate_duration

.. autofunction:: eegfeat.microstates.microstate_occurrence

.. autofunction:: eegfeat.microstates.microstate_transitions

.. autoclass:: eegfeat.microstates.MicrostateSegmentation
   :members:
   :show-inheritance:
