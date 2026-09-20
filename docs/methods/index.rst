Methods
=======

.. raw:: html

   <p class="hero-lede">
     The mathematical formulation behind every measure: what is computed, on which
     grid, under which assumptions, and what happens when those assumptions fail.
   </p>

Each page here faces a page in the :doc:`/api/index` with the same name. Read the
method for the definition and the rationale; read the API entry for the signature
and the parameters.

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Spectral
      :link: spectral
      :link-type: doc

      Band power and normalization, wavelet support restriction, peak frequency,
      spectral descriptors, aperiodic fitting, ratios and asymmetry.

   .. grid-item-card:: Dynamics
      :link: dynamics
      :link-type: doc

      ERDS, oscillatory bursts, time-domain descriptors, peak amplitude and
      latency.

   .. grid-item-card:: Phase & Connectivity
      :link: connectivity
      :link-type: doc

      ITPC, phase-amplitude coupling, envelope correlation and wPLI, common
      spatial patterns, graph summaries.

   .. grid-item-card:: Complexity & Microstates
      :link: complexity
      :link-type: doc

      Sample entropy, multiscale entropy, and GFP-peak microstate segmentation.

The statistical methods behind predictive modeling — group-disjoint
cross-fitting, permutation nulls, conformal calibration — are documented inline
in the :doc:`/guides/modeling` guide, alongside the code that applies them.

.. toctree::
   :hidden:
   :maxdepth: 2

   spectral
   dynamics
   connectivity
   complexity
