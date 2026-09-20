API Reference
=============

.. raw:: html

   <p class="hero-lede">
     Signatures, parameters, and return types for every public class and function.
   </p>

Everything documented here is exported from the top-level ``eegfeat`` namespace,
except the modeling API (``eegfeat.model``) and microstates
(``eegfeat.microstates``). Each feature page faces a page in :doc:`/methods/index`
that gives the definition and the assumptions.

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Containers & I/O
      :link: containers
      :link-type: doc

      ``FeatureTable``, ``Spectra``, ``Signal``, ``BandSignal``, ``Band``,
      ``Window``, and table reading and writing.

   .. grid-item-card:: Spectral Features
      :link: spectral
      :link-type: doc

      Band power, peak frequency, spectral descriptors, aperiodic fits, ratios.

   .. grid-item-card:: Dynamics
      :link: dynamics
      :link-type: doc

      Time-domain descriptors, oscillatory bursts, ERDS magnitudes and latencies.

   .. grid-item-card:: Phase & Connectivity
      :link: connectivity
      :link-type: doc

      ITPC, PPC, PAC, envelope correlation, wPLI, CSP, graph summaries.

   .. grid-item-card:: Complexity & Microstates
      :link: complexity
      :link-type: doc

      Higuchi fractal dimension, sample and multiscale entropy, microstates.

   .. grid-item-card:: Predictive Modeling
      :link: model
      :link-type: doc

      Design construction, cross-fitting, metrics, nulls, uncertainty, importance.

.. toctree::
   :hidden:
   :maxdepth: 2

   containers
   spectral
   dynamics
   connectivity
   complexity
   model
