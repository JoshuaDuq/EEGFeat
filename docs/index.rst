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

.. grid:: 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Install
      :link: install
      :link-type: doc

      Environment, core dependencies, and optional extras.

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc

      From MNE objects to feature tables in a few lines.

   .. grid-item-card:: Methods
      :link: methods
      :link-type: doc

      Mathematical formulations, algorithms, and rationale.

   .. grid-item-card:: Modeling
      :link: modeling
      :link-type: doc

      Design matrices, group-disjoint cross-fitting, uncertainty, and importance.

----

Core Capabilities
-----------------

.. grid:: 3
   :gutter: 3
   :class-container: stage-grid

   .. grid-item-card:: 01 — Spectral Power & Descriptors
      :link: methods
      :link-type: doc

      Trapezoidal frequency-weighted band power, spectral centroid,
      bandwidth, Shannon entropy, and spectral edge frequency (SEF95).

      ``eegfeat.integrated_band_power`` · ``eegfeat.mean_psd`` · ``eegfeat.spectral_centroid``

   .. grid-item-card:: 02 — Wavelet Support Restriction
      :link: methods
      :link-type: doc

      Exact per-frequency temporal support masking for Morlet wavelets,
      preventing edge contamination and pre-stimulus leakage into task windows.

      ``Spectra.from_tfr(..., n_cycles)``

   .. grid-item-card:: 03 — Aperiodic Whitening
      :link: methods
      :link-type: doc

      Iterative Huber/MAD 1/f background fitting, spectral flattening,
      prominence-gated peak detection, and parabolic frequency refinement.

      ``eegfeat.peak_frequency`` · ``aperiodic_ratio``

   .. grid-item-card:: 04 — Bursts & ERDS Dynamics
      :link: methods
      :link-type: doc

      Baseline-calibrated envelope burst rate/duration, signed ERD/ERS
      magnitudes, onset latency, peak latency, and rebound latency.

      ``eegfeat.burst_rate`` · ``eegfeat.erds_mean``

   .. grid-item-card:: 05 — Phase & Connectivity
      :link: methods
      :link-type: doc

      Inter-trial phase coherence (ITPC), phase-amplitude coupling (PAC),
      envelope correlation (AEC), and wPLI via MNE-Connectivity.

      ``eegfeat.itpc`` · ``eegfeat.wpli``

   .. grid-item-card:: 06 — Complexity & Microstates
      :link: methods
      :link-type: doc

      Native Sample Entropy, Multiscale Entropy, and GFP-peak
      clustered microstate segmentation with duration and transition metrics.

      ``eegfeat.sample_entropy`` · ``microstates.segment``

   .. grid-item-card:: 07 — Predictive Modeling
      :link: modeling
      :link-type: doc

      Per-epoch design matrices, group-disjoint nested validation, fold-local preprocessing,
      permutation nulls, conformal intervals, and metadata-aware importance.

      ``eegfeat.model.build_design`` · ``eegfeat.model.cross_fit_regression``

----

Explore the Documentation
-------------------------

.. grid:: 5
   :gutter: 2
   :class-container: docs-nav

   .. grid-item-card:: Install
      :link: install
      :link-type: doc

      Environment, dependencies, and extras.

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc

      End-to-end operational workflows.

   .. grid-item-card:: Methods
      :link: methods
      :link-type: doc

      Mathematical formulations and rationale.

   .. grid-item-card:: Modeling
      :link: modeling
      :link-type: doc

      Leakage-safe predictive workflows and model diagnostics.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Public classes, methods, and functions.

.. raw:: html

   <p class="dev-status">
     <strong>Status</strong> — <code>eegfeat</code> is in active pre-release development.
     The public API is stable, strictly typed, and verified across comprehensive unit and property test suites.
   </p>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   install
   quickstart
   runner

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Scientific Methods

   methods

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Predictive Modeling

   modeling

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   api
