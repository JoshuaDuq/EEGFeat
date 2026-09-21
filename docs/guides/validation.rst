Validation on Public Datasets
=============================

.. raw:: html

   <p class="hero-lede">
     The unit tests prove each formula on synthetic signals. This page answers a
     different question: pointed at <strong>real recordings</strong> nobody tuned it on,
     does eegfeat compute the right numbers and find what the literature says is there?
   </p>

.. include:: ../validation/summary.inc

Every number on this page is written by the validation suite itself when it runs; nothing
here is typed in by hand. The tests declare what they establish, record what they observe,
and the page is regenerated from that record, so it cannot drift from the code.

.. _validation-scorecard:

Scorecard
---------

One row per public function, one column per kind of evidence. A check mark with a count
means every check of that kind passed on real data; a dash means no check of that kind
exists for that function.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Kind
     - What a check of this kind establishes
   * - Formula
     - The value equals an independent NumPy, SciPy or closed-form computation on the
       same data, to floating-point precision.
   * - Other estimator
     - The value agrees with a different estimator or library of the same quantity
       (Welch against multitaper, Hilbert against wavelet, ``mne_connectivity``).
   * - Known physiology
     - A textbook effect comes out of the recording, at the level it is known to hold.
   * - Decoding
     - A leakage-safe model recovers a contrast strong enough that a correct pipeline must.
   * - Behaviour
     - A documented refusal, identity, null rate or command-line behaviour holds.

.. include:: ../validation/scorecard.inc

What was confirmed
------------------

Across four public datasets, every closed-form measure matched its reference computation,
every cross-estimator comparison agreed, and the following effects came out of the data:

- **Motor cortex.** Hand movement desynchronizes mu and beta over the hand areas in all
  twenty PhysioNet subjects, deepens and lengthens the desynchronized part of the ERDS trace,
  and suppresses beta bursts. Movement decodes from sensorimotor power within and across
  subjects, and cross-fitted CSP separates it in every subject.
- **Visual entrainment.** The 12 Hz and 15 Hz flicker lines and their second harmonics
  separate the two SSVEP conditions on every trial, phase-lock across trials only when
  driven, and raise occipital coherence at the driven frequency.
- **Evoked potentials.** On ERP CORE the single-trial N1 latency clusters near 180 ms, the
  P3 area is positive on most trials, and wrong button presses produce an error-related
  negativity at FCz about 11 microvolts below correct ones.
- **Sleep depth.** From wake to N3 the slow-wave fraction rises while alpha, beta, the
  spectral edge, centroid, entropy, Hjorth mobility, sample entropy and fractal dimension all
  fall and the aperiodic slope steepens. Deep sleep decodes across subjects at 0.96 balanced
  accuracy, and sleep depth regresses with a subject-level correlation near 0.9 against a
  permutation null at zero.
- **Connectivity and microstates.** Neighbouring electrodes' alpha envelopes correlate far
  more than distant ones until orthogonalized, all eight spectral connectivity methods
  reproduce ``mne_connectivity``, and four resting microstates explain most of the variance
  with durations of tens of milliseconds.

What the data changed
---------------------

Three defaults were changed and one note added because real recordings showed the previous
behaviour would mislead a user who trusted it.

- **ERDS reports decibels.** Single-trial percent change is right-skewed by quiet-baseline
  trials; on the percent scale only about half the motor subjects showed mu desynchronization
  in the trial mean, on the decibel scale all twenty did. Every ``erds_*`` function now
  defaults to ``normalize="db"``; percent remains available.
- **Onset persistence is in cycles.** At a fixed 100 ms the onset latency fired on essentially
  every rest trial, where nothing happened. Six cycles of the band's low edge brings that
  false-onset rate to about 5 percent in every band, and is now the default.
- **Two decibel definitions are named.** ``erds_mean`` averages a per-sample dB trace;
  ``mean_tfr_power`` with a baseline takes dB of the window-mean power, about 2.5 dB higher.
  Both docstrings and the methods page now say so.
- **Peak smoothing has a caveat.** The 1 Hz default reported alpha instead of the 12 Hz
  flicker; the docstring now says to turn smoothing off for narrow entrained lines.

What was not found, and is not asserted
---------------------------------------

- **No contralateral dominance** of motor desynchronization across twenty subjects, on any
  scale or electrode choice, and left against right hand decodes near chance. The dataset is
  known for weak lateralization; the suite reports what is there.
- **Raw phase-amplitude coupling** did not resolve slow-oscillation to spindle coupling
  against a time-shifted surrogate on two sleep derivations, and sits well above zero even
  with the coupling destroyed. Only the formula and that surrogate floor are asserted; read
  :func:`~eegfeat.pac` against a null of your own.
- **The post-movement beta rebound** was not visible at the group level in the 4.5 to 5.8 s
  window of the motor task.
- **Single-trial ERDS onsets** fired on movement trials at the same rate as on rest trials at
  every persistence, so the onset is a weak single-trial detector even where the group effect
  is unmistakable.
- **Prediction intervals** covered 88 to 91 percent of exchangeable epochs at 90 percent
  nominal, but 72 to 99 percent when calibrated on two sleep subjects and applied to a third.
  Group-disjoint fitting does not establish coverage for new subjects.
- **The frontal sleep derivation** (Fpz-Cz) does not separate wake from N3, because eye and
  movement activity in wake put as much power below 4 Hz as slow waves do. The sleep checks
  read Pz-Oz.

Datasets
--------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Dataset
     - What it is
     - What must come out
   * - PhysioNet EEG Motor Movement/Imagery
     - Twenty subjects, 64 channels at 160 Hz, three runs of cued left- and right-fist movement
       with rest between trials
     - Mu and beta desynchronization over the hand areas, fewer bursts, movement decodable and
       separable by CSP, volume conduction in envelope correlation, resting microstates
   * - MNE SSVEP example
     - One subject, 32 channels, ten 20-second trials each of 12 Hz and 15 Hz flicker
     - Spectral lines and harmonics at the flicker frequency, phase locking and coherence only
       when driven
   * - ERP CORE, Flankers task
     - One subject, 30 channels at 1024 Hz, 400 arrow stimuli with a left or right press
     - The visual N1, a parietal P3, and an error-related negativity at FCz
   * - Sleep-EDF (PhysioNet)
     - Three subjects, two EEG derivations at 100 Hz, expert-scored 30-second epochs
     - Slow-wave power with depth, every spectral and complexity descriptor falling from wake
       to N3, the aperiodic slope steepening, N3 and sleep depth recoverable across subjects

Running it
----------

The recordings are downloaded on first use, about 350 MB in total, into MNE's data
directory (``MNE_DATA``, default ``~/mne_data``). The suite is skipped unless asked for:

.. code-block:: bash

   python -m pip install -e ".[dev,model,connectivity,microstates]"
   EEGFEAT_DATASETS=1 python -m pytest tests/validation -ra

A run rewrites ``docs/validation/results.json`` and the fragments this page includes, so
the scorecard and the tables below reflect the last run on the machine that built the docs.
The ``validation`` GitHub workflow runs the suite weekly and on demand, once on the oldest
supported MNE (1.8) and once on the newest, with the data cached between runs.

.. _validation-results:

Every check
-----------

Each row is one test: the claim it makes, the criterion it applies, what it observed on the
last run, and whether it passed. Rows are grouped by dataset.

.. include:: ../validation/results.inc
