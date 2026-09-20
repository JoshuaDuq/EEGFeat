eegfeat
=======

.. raw:: html

   <p class="hero-lede">
     Labelled, mathematically sound <strong>spectral</strong>, <strong>temporal</strong>,
     and <strong>connectivity</strong> feature extraction plus leakage-safe <strong>predictive
     modeling</strong> for <strong>MNE</strong> objects. Turns raw spectra, time-frequency
     arrays, and analytic band signals into structured, inspectable <code>FeatureTable</code>
     outputs, then carries per-epoch tables into grouped model evaluation.
   </p>

.. grid:: 4
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

      Real files from a simulated five-subject cohort.

----

What It Computes
----------------

.. grid:: 3
   :gutter: 3
   :class-container: stage-grid

   .. grid-item-card:: Spectral
      :link: methods/spectral
      :link-type: doc

      Trapezoidal frequency-weighted band power, centroid, bandwidth, Shannon
      entropy and SEF95, iterative Huber/MAD 1/f fitting with prominence-gated
      peak detection, and exact per-frequency Morlet support masking.

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

      Per-epoch design matrices, group-disjoint nested validation, fold-local
      preprocessing, permutation nulls, conformal intervals, and metadata-aware
      importance.

      ``build_design`` · ``cross_fit_regression`` · ``permutation_test``

   .. grid-item-card:: Batch Runner
      :link: guides/runner
      :link-type: doc

      One declarative TOML recipe applied to a folder of preprocessed epochs
      files, writing metadata-carrying tables that load straight back as
      ``FeatureTable`` objects.

      ``eegfeat run`` · ``eegfeat check``

Every measure above is defined, with its assumptions and failure modes, in
:doc:`methods/index`. Signatures and parameters are in :doc:`api/index`.

.. raw:: html

   <p class="dev-status">
     <strong>Status</strong> — <code>eegfeat</code> is in active pre-release development.
     The public API is stable, strictly typed, and verified across comprehensive unit and property test suites.
   </p>

.. toctree::
   :hidden:
   :caption: Getting Started

   install
   quickstart
   concepts

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
