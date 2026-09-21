Methods
=======

.. raw:: html

   <p class="hero-lede">
     Definitions, frequency and time grids, and the conditions that return
     NaN or set a flag.
   </p>

Each page matches a page in the :doc:`/api/index`. The method page is the
definition. The API page is the signature.

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Spectral
      :link: spectral
      :link-type: doc

      Band power and normalization, wavelet support, peak frequency,
      spectral descriptors, aperiodic fitting, ratios, and asymmetry.

   .. grid-item-card:: Dynamics
      :link: dynamics
      :link-type: doc

      ERDS, oscillatory bursts, time-domain descriptors, peak amplitude, and
      latency.

   .. grid-item-card:: Phase & Connectivity
      :link: connectivity
      :link-type: doc

      ITPC, phase-amplitude coupling, envelope correlation, wPLI, common
      spatial patterns, and graph summaries.

   .. grid-item-card:: Complexity & Microstates
      :link: complexity
      :link-type: doc

      Higuchi fractal dimension, sample entropy, multiscale entropy, and
      GFP-peak microstate segmentation.

Cross-fitting, permutation nulls, and conformal intervals are in
:doc:`/guides/modeling`.

.. toctree::
   :hidden:
   :maxdepth: 2

   spectral
   dynamics
   connectivity
   complexity
