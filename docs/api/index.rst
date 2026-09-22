API Reference
=============

.. raw:: html

   <p class="hero-lede">
     Signatures, parameters, and return types.
   </p>

These names are exported from ``eegfeat``, except ``eegfeat.model``,
``eegfeat.microstates``, and ``eegfeat.preprocessing``, which live in their submodules. Definitions are in
:doc:`/methods/index`.

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

   .. grid-item-card:: Preprocessing
      :link: preprocessing
      :link-type: doc

      Raw-to-epochs workflow, checkpoints, review, and the numerical operations.

.. toctree::
   :hidden:
   :maxdepth: 2

   containers
   spectral
   dynamics
   connectivity
   complexity
   model
   preprocessing
