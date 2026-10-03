:hide-toc:

EEGFeat
=======

.. rst-class:: hero-lede

   Labelled EEG feature extraction and modeling for MNE objects.

Extract spectral, temporal, connectivity, and complexity measures while
retaining their units and provenance. Evaluate models with group-disjoint
validation. Method definitions describe the assumptions and missing-value
behavior of each measure.

.. container:: overview-links

   - :doc:`Quick start <quickstart>`
   - :doc:`API reference <api/index>`
   - :doc:`Validation evidence <guides/validation>`

Start here
----------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: document-directory

   .. grid-item::

      .. rubric:: :doc:`Installation <install>`

      Install from source, prepare your environment, and choose optional extras.

   .. grid-item::

      .. rubric:: :doc:`Quick start <quickstart>`

      Work through PSD, time-frequency, and burst examples with MNE objects.

   .. grid-item::

      .. rubric:: :doc:`Data concepts <concepts>`

      Inspect containers, feature names, units, provenance, and missing values.

Explore the methods
-------------------

Every feature has a definition. Start with the signal property you want to measure.

.. container:: method-directory

   :doc:`Spectral <methods/spectral>`
      Band power, spectral shape, aperiodic fits, peak detection, and Morlet
      support masking.

      ``integrated_band_power`` · ``peak_frequency`` · ``aperiodic_ratio``

   :doc:`Dynamics <methods/dynamics>`
      Envelope bursts, ERD/ERS magnitudes and latencies, and time-domain
      waveform descriptors.

      ``burst_rate`` · ``erds_mean`` · ``hjorth_complexity``

   :doc:`Phase & connectivity <methods/connectivity>`
      Phase consistency, phase-amplitude coupling, envelope correlation,
      wPLI, spatial patterns, and graph summaries.

      ``itpc`` · ``pac`` · ``wpli``

   :doc:`Complexity & microstates <methods/complexity>`
      Sample and multiscale entropy, Higuchi fractal dimension, and
      GFP-peak clustered microstate segmentation.

      ``sample_entropy`` · ``microstates.segment``

Build an analysis workflow
--------------------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: document-directory

   .. grid-item::

      .. rubric:: :doc:`Preprocess recordings <guides/preprocessing>`

      Raw-to-epochs cleaning with reviewed artifacts and validated checkpoints.

   .. grid-item::

      .. rubric:: :doc:`Process a cohort <guides/runner>`

      Apply one TOML recipe to a folder of preprocessed epochs files.

   .. grid-item::

      .. rubric:: :doc:`Evaluate a model <guides/modeling>`

      Grouped cross-fitting, permutation nulls, and conformal intervals.

.. container:: research-note

   .. rubric:: Inspect the evidence

   Review the :doc:`method definitions <methods/index>` and
   :doc:`public-dataset validation <guides/validation>` before interpreting
   a feature. Explore :doc:`example output <examples>` from a simulated
   five-subject cohort, or learn to :doc:`query and export tables <guides/tables>`.

.. toctree::
   :hidden:
   :caption: Getting started

   install
   Quick start <quickstart>
   Data concepts <concepts>

.. toctree::
   :hidden:
   :caption: Guides

   Tables & files <guides/tables>
   guides/preprocessing
   Cohort runner <guides/runner>
   Predictive modeling <guides/modeling>

.. toctree::
   :hidden:
   :caption: Methods
   :maxdepth: 2

   Method definitions <methods/index>

.. toctree::
   :hidden:
   :caption: API reference
   :maxdepth: 2

   API overview <api/index>

.. toctree::
   :hidden:
   :caption: Research resources

   Validation <guides/validation>
   Example outputs <examples>
