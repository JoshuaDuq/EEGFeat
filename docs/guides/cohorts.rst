Reproducible cohorts
====================

EEGFeat keeps the numerical feature definition, its evidence and the recording
identity together. A saved table uses schema 2: its sidecar declares the exact
payload filenames and their SHA-256 checksums. Reading a modified payload raises;
regenerate older bundles rather than loading them through a compatibility path.

Extraction manifests record input content, input-relative recording identity,
resolved settings including defaults, installed scientific package versions and
the Python source identity. Every linked part of a split FIF recording is
checksummed; changing or removing any part invalidates the result.
When EEGFeat preprocessing produced the epochs, its
manifest identity and payload checksums are verified and carried into extraction.
Externally prepared epochs explicitly have no EEGFeat upstream manifest. Moving
the dataset preserves identities when its relative directory structure stays the
same. An implementation or environment change makes old results stale.

Quality policies
----------------

Coverage measures finite numerical input; it does not establish artifact-free
data. Inspect preprocessing retention, rejection decisions and feature-specific
flags as well. Choose a fixed exclusion policy before evaluating a model:

.. code-block:: python

   from eegfeat.io import read_dataset
   from eegfeat.quality import QualityPolicy
   from eegfeat.model import build_design

   dataset = read_dataset(paths)
   policy = QualityPolicy(min_coverage=0.8, rejected_flags=("edge_hit",))
   design = build_design(dataset.table, dataset.targets, target="rating",
                         groups="subject_id", quality=policy)
   # design.meta, coverage, flags and quality_ledger preserve the evidence.

Unknown flags raise rather than silently disabling an exclusion. Rejected cells
become NaN, with a ``quality_rejected`` flag and a ledger of their reasons. The
input table is unchanged. Missingness learned from the cohort, imputation,
scaling, feature selection and supervised transforms still belong inside each
training fold.

Write an HTML report using :func:`eegfeat.report.write_quality_report`, or from
current batch outputs:

.. code-block:: bash

   eegfeat report recipe.toml quality.html --by recording
   eegfeat report recipe.toml quality.html --by subject_id condition --min-coverage 0.8
   eegfeat report recipe.toml groups.html --rows groups --by recording

Reports contain cohort and feature missingness, coverage and flag fractions,
exclusion decisions, complete feature definitions and report provenance. The
descriptor rows passed to the Python reporting functions must already be aligned
to the feature rows. The command-line report also restores observed preprocessing
retention and artifact decisions from extraction manifests; unavailable upstream
evidence stays explicitly missing. Use :func:`eegfeat.report.recording_quality`
to build the same recording summary for the Python reporting interface.

Group samples
-------------

An across-trial coherence or ITPC estimate is one group sample. Its constituent
epochs do not become independent observations of that estimate. The runner saves
``recording`` and ``n_trials`` descriptors for every cross-trial row. Assemble
these outputs separately:

.. code-block:: python

   from eegfeat.group import read_group_dataset, build_group_design

   dataset = read_group_dataset(crosstrial_paths)
   # Supply explicitly matched subject IDs/outcomes in dataset.targets.
   design = build_group_design(dataset, target="outcome", groups="subject_id")

Canonical group identities are ``(recording, group)``. Duplicate identities,
invalid trial counts and incorrectly aligned descriptors raise. Column unions
preserve different measured channel schemas with NaN values and zero coverage
where a recording did not measure a column. Evaluate designs with the existing
group-disjoint classification or regression functions. For generalization to a
new study, group folds by study and keep all observations from that study out of
both fitting and model selection.

Repeated-session reliability
----------------------------

:func:`eegfeat.intraclass_reliability` evaluates one explicitly aggregated estimate
per subject/session, with a complete balanced design and at least three subjects
and two sessions. It reports single-measure absolute agreement ICC(2,1) and
consistency ICC(3,1). Absolute agreement detects systematic session offsets that
consistency can ignore. Missing values, duplicate samples, unbalanced designs and
constant features raise; this API never silently drops subjects or fills values.
The estimates are descriptive and do not supply confidence intervals.

The formulas follow `Pingouin's documented two-way ANOVA implementation
<https://pingouin-stats.org/generated/pingouin.intraclass_corr.html>`_. Quality
reports use `MNE.Report <https://mne.tools/stable/generated/mne.Report.html>`_.

.. autoclass:: eegfeat.quality.QualityPolicy
   :members:

.. autofunction:: eegfeat.quality.apply_quality

.. autofunction:: eegfeat.quality.feature_quality

.. autofunction:: eegfeat.quality.cohort_quality

.. autofunction:: eegfeat.report.write_quality_report

.. autofunction:: eegfeat.report.recording_quality

.. autofunction:: eegfeat.group.read_group_dataset

.. autofunction:: eegfeat.group.build_group_design

.. autofunction:: eegfeat.intraclass_reliability
