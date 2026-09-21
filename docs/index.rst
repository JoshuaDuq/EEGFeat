eegfeat
=======

.. raw:: html

   <p class="hero-lede">
     <strong>Spectral</strong>, <strong>temporal</strong>, and <strong>connectivity</strong>
     features for <strong>MNE</strong> objects, returned as <code>FeatureTable</code>,
     and grouped model evaluation on per-epoch tables.
   </p>

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Install
      :link: install
      :link-type: doc

      Environment, core dependencies, and optional extras.

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc

      Three worked workflows, MNE objects to feature tables.

   .. grid-item-card:: Concepts
      :link: concepts
      :link-type: doc

      Containers, table anatomy, column names, and what missing means.

   .. grid-item-card:: Example Output
      :link: examples
      :link-type: doc

      Files from a simulated five-subject cohort.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Signatures, parameters, and return types.

   .. grid-item-card:: Validation
      :link: guides/validation
      :link-type: doc

      Public-dataset checks, and the scorecard those tests write.

----

What It Computes
----------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: stage-grid

   .. grid-item-card:: Spectral
      :link: methods/spectral
      :link-type: doc

      Trapezoidal band power, centroid, bandwidth, Shannon entropy and SEF95,
      iterative MAD 1/f fitting, prominence-gated peak detection, and Morlet
      support masking.

      ``integrated_band_power`` · ``peak_frequency`` · ``aperiodic_ratio``

   .. grid-item-card:: Dynamics
      :link: methods/dynamics
      :link-type: doc

      Baseline-calibrated envelope burst rate and duration, signed ERD/ERS
      magnitudes, onset, peak and rebound latencies, and time-domain descriptors
      of the waveform itself.

      ``burst_rate`` · ``erds_mean`` · ``hjorth_complexity``

   .. grid-item-card:: Phase & Connectivity
      :link: methods/connectivity
      :link-type: doc

      Inter-trial phase coherence, pairwise phase consistency,
      phase-amplitude coupling, envelope correlation, wPLI via MNE-Connectivity,
      common spatial patterns, and graph summaries.

      ``itpc`` · ``pac`` · ``wpli``

   .. grid-item-card:: Complexity & Microstates
      :link: methods/complexity
      :link-type: doc

      Native Sample Entropy and Multiscale Entropy, Higuchi fractal dimension,
      and GFP-peak clustered microstate segmentation with duration, occurrence
      and transition metrics.

      ``sample_entropy`` · ``microstates.segment``

   .. grid-item-card:: Predictive Modeling
      :link: guides/modeling
      :link-type: doc

      Per-epoch design matrices, group-disjoint nested validation, preprocessing
      inside the training fold, permutation nulls, conformal intervals, and
      importance with the column metadata attached.

      ``build_design`` · ``cross_fit_regression`` · ``permutation_test``

   .. grid-item-card:: Batch Runner
      :link: guides/runner
      :link-type: doc

      One TOML recipe applied to a folder of preprocessed epochs files. Each
      written table loads back as a ``FeatureTable``.

      ``eegfeat run`` · ``eegfeat check``

Definitions are in :doc:`methods/index`. Signatures are in :doc:`api/index`.

.. raw:: html

   <p class="dev-status">
     <strong>Status.</strong> Pre-release. The public API is stable.
   </p>

.. toctree::
   :hidden:
   :caption: Getting Started

   install
   quickstart
   concepts
   Validation <guides/validation>

.. toctree::
   :hidden:
   :caption: Guides

   guides/tables
   guides/runner
   guides/modeling
   examples

.. toctree::
   :hidden:
   :caption: Methods
   :maxdepth: 2

   methods/index

.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 2

   api/index
