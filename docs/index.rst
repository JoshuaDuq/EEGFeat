:hide-toc:

EEGFeat
=======

.. raw:: html

   <div class="home-overview">
     <div class="home-copy">
       <p class="home-intro">From EEG signals to<br>reproducible features.</p>
       <p class="home-description">
         Extract defined measures from MNE objects. Keep their units and
         provenance, then evaluate models with group-disjoint validation.
       </p>
       <div class="hero-actions">
         <a href="quickstart.html">Quick start <span aria-hidden="true">→</span></a>
         <a href="api/index.html">API reference <span aria-hidden="true">↗</span></a>
       </div>
       <a class="hero-install" href="install.html">Installation &amp; requirements <span aria-hidden="true">→</span></a>
     </div>
     <figure class="hero-visual">
       <svg viewBox="0 0 420 280" role="img" aria-labelledby="workflow-title workflow-description" fill="none">
         <title id="workflow-title">From MNE objects to labelled features</title>
         <desc id="workflow-description">Illustrative EEG traces are transformed into defined spectral measures and a feature table that retains units and provenance. These curves are a schematic, not recorded data.</desc>
         <path class="plot-grid" d="M0 52h420M0 76h420M0 100h420M30 36v64M90 36v64M150 36v64M210 36v64M270 36v64M330 36v64M390 36v64"/>
         <text class="plot-heading" x="0" y="18">01 / MNE OBJECTS</text>
         <text class="plot-label" x="420" y="18" text-anchor="end">Epochs · Spectrum · TFR</text>
         <path class="plot-trace" d="M0 52l8-2 8 4 8-2 8-7 8 10 8-5 8 3 8-9 8 14 8-9 8 1 8-3 8 8 8-6 8 2 8-5 8 8 8-5 8 1 8-2 8 5 8-3 8-7 8 13 8-7 8 2 8 3 8-8 8 4 8 3 8-3 8-2 8 3 8-5 8 9 8-6 8 2 8 3 8-7 8 5 8-3 8 4 8-1 8-3 8 5 8-2 8-1 8-4 8 7 8-4 8 1 4-1"/>
         <path class="plot-trace plot-trace-secondary" d="M0 76c8-12 16 12 24 0s16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0 16-12 24 0 16 12 24 0"/>
         <path class="plot-connector" d="M210 107v17m-4-4 4 4 4-4"/>
         <text class="plot-heading" x="0" y="149">02 / DEFINED MEASURES</text>
         <text class="plot-label" x="420" y="149" text-anchor="end">Bands · windows · assumptions</text>
         <path class="plot-band" d="M84 164h64v48H84z"/>
         <path class="plot-grid" d="M0 188h420M0 212h420"/>
         <path class="plot-trace" d="M0 165C14 178 23 199 40 202S73 208 84 194 101 166 116 172 137 201 156 204 210 206 252 209 349 210 420 211"/>
         <text class="plot-label" x="116" y="160" text-anchor="middle">α</text>
         <path class="plot-connector" d="M210 219v17m-4-4 4 4 4-4"/>
         <text class="plot-heading" x="0" y="267">03 / FEATURE TABLE</text>
         <text class="plot-label" x="420" y="267" text-anchor="end">Values · units · provenance</text>
       </svg>
       <figcaption>Illustrative workflow <span aria-hidden="true">·</span> <a href="concepts.html">Explore the data model →</a></figcaption>
     </figure>
   </div>

Start here
----------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: nav-cards start-directory

   .. grid-item-card:: Set up EEGFeat
      :link: install
      :link-type: doc

      Install from source, prepare your environment, and choose optional extras.

   .. grid-item-card:: Extract your first features
      :link: quickstart
      :link-type: doc

      Work through PSD, time-frequency, and burst examples with MNE objects.

   .. grid-item-card:: Understand your outputs
      :link: concepts
      :link-type: doc

      Inspect containers, feature names, units, provenance, and missing values.

Explore the methods
-------------------

Every feature has a definition. Start with the signal property you want to measure.

.. grid:: 1
   :gutter: 3
   :class-container: method-directory

   .. grid-item-card:: Spectral
      :link: methods/spectral
      :link-type: doc

      Band power, spectral shape, aperiodic fits, peak detection, and Morlet
      support masking.

      ``integrated_band_power`` · ``peak_frequency`` · ``aperiodic_ratio``

   .. grid-item-card:: Dynamics
      :link: methods/dynamics
      :link-type: doc

      Envelope bursts, ERD/ERS magnitudes and latencies, and time-domain
      waveform descriptors.

      ``burst_rate`` · ``erds_mean`` · ``hjorth_complexity``

   .. grid-item-card:: Phase & connectivity
      :link: methods/connectivity
      :link-type: doc

      Phase consistency, phase-amplitude coupling, envelope correlation,
      wPLI, spatial patterns, and graph summaries.

      ``itpc`` · ``pac`` · ``wpli``

   .. grid-item-card:: Complexity & microstates
      :link: methods/complexity
      :link-type: doc

      Sample and multiscale entropy, Higuchi fractal dimension, and
      GFP-peak clustered microstate segmentation.

      ``sample_entropy`` · ``microstates.segment``

Build an analysis workflow
--------------------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Preprocess recordings
      :link: guides/preprocessing
      :link-type: doc

      Raw-to-epochs cleaning with reviewed artifacts and validated checkpoints.

   .. grid-item-card:: Process a cohort
      :link: guides/runner
      :link-type: doc

      Apply one TOML recipe to a folder of preprocessed epochs files.

   .. grid-item-card:: Evaluate a model
      :link: guides/modeling
      :link-type: doc

      Grouped cross-fitting, permutation nulls, and conformal intervals.

.. container:: research-note

   .. rubric:: Inspect the evidence

   Review the :doc:`method definitions <methods/index>` and
   :doc:`public-dataset validation <guides/validation>` before interpreting
   a feature. Explore :doc:`example output <examples>` from a simulated
   five-subject cohort, or learn to :doc:`query and export tables <guides/tables>`.

.. raw:: html

   <p class="dev-status"><strong>Pre-release.</strong> The public API is stable.</p>

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
   guides/preprocessing
   guides/runner
   guides/modeling

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

.. toctree::
   :hidden:
   :caption: Research Resources

   Validation <guides/validation>
   examples
