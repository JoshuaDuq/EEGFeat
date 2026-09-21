Validation on Public Datasets
=============================

.. raw:: html

   <p class="hero-lede">
     The unit tests prove each formula against synthetic signals. The validation suite
     asks a different question: pointed at <strong>real recordings</strong> nobody tuned it
     on, does eegfeat find what the literature says is there?
   </p>

The suite lives in ``tests/validation`` and runs against four datasets that ship with
MNE-Python. Each has a textbook effect, and each check states the effect in advance and
tests for it at the level where it is known to hold.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Dataset
     - What it is
     - What must come out
   * - PhysioNet EEG Motor Movement/Imagery
     - Twenty subjects, 64 channels, three runs of cued left- and right-fist movement
     - Mu and beta desynchronization over the hand areas during movement in every
       subject, with fewer bursts; movement decodable from sensorimotor power and
       separable by cross-fitted CSP; neighbouring electrodes' envelopes correlate far
       more than distant ones until orthogonalized; four resting microstates lasting
       tens of milliseconds
   * - MNE SSVEP example
     - One subject, 32 channels, ten 20-second trials each of 12 Hz and 15 Hz flicker
     - A spectral line at the flicker frequency and its second harmonic over occipital
       cortex, separating the two conditions on every trial; phase locked across trials
       and coherent across occipital electrodes only when driven
   * - ERP CORE, Flankers task
     - One subject, 30 channels at 1024 Hz, 400 arrow stimuli with a left or right button
       press
     - The visual N1 near 180 ms over lateral occipital cortex, a positive P3 over the
       parietal midline, and an error-related negativity at FCz after wrong presses
   * - Sleep-EDF (PhysioNet)
     - Three subjects, two EEG derivations, expert-scored 30-second epochs
     - Power shifts to slow waves with sleep depth; edge frequency, centroid, entropy,
       Hjorth mobility, sample entropy and fractal dimension all fall from wake to N3
       while the aperiodic slope steepens; N3 decodable across subjects

Running it
----------

The recordings are downloaded on first use, about 350 MB in total, into MNE's data
directory (``MNE_DATA``, default ``~/mne_data``). The suite is skipped unless asked for:

.. code-block:: bash

   python -m pip install -e ".[dev,model]"
   EEGFEAT_DATASETS=1 python -m pytest tests/validation -ra

The ``validation`` GitHub workflow runs the same command weekly and on demand, with the
data cached between runs, once on the oldest supported MNE (1.8) and once on the newest.

Three kinds of check
--------------------

**Agreement with direct computation.** On the real spectra and time series, the library's
number must be the textbook number: integrated band power is the trapezoid integral of
MNE's Welch PSD over the band, mean PSD is that integral over the band width, the
uncorrected peak frequency is the argmax bin, the aperiodic fit without peak rejection is
ordinary least squares in log-log space, the ERDS mean, slope and peak latency are what the
documented trace gives and the two magnitudes and durations balance to that mean, and
variance, RMS, peak-to-peak, mean, quantiles,
skewness, kurtosis, line length, Hjorth mobility, ITPC, PAC, raw envelope correlation and
asymmetry are their documented formulas. These hold to floating-point precision. every spectral connectivity
method must reproduce a direct ``mne_connectivity`` call, a multitaper PSD must integrate
edge to edge on a grid that misses the band edges, multiscale entropy at scale one must equal
sample entropy, CSP filters and patterns must be mutually inverse, and the microstate
measures must obey their identities: coverage sums to one, equals duration times
occurrence, and each state's transition probabilities sum to one. The Hilbert-envelope
ERDS is also compared with MNE's multitaper time-frequency ERDS, a different estimator of
the same quantity, and the two rank channels and trials alike on every subject.

**Known physiology.** Effects are tested where they are established. Motor
desynchronization is a group-level claim, so the per-subject mean over a five-electrode
cluster around each hand area is tested with a one-sided t-test across twenty subjects.
The SSVEP and sleep effects are large enough to hold on every trial, or as areas under the
curve near zero or one within each subject.

**Leakage-safe decoding.** The modeling layer is checked on contrasts strong enough that a
correct pipeline must find them: movement against rest within subject with run-disjoint
folds and across subjects with leave-one-subject-out folds, and wake against deep sleep
across subjects. Regression is checked on sleep depth scored 0 to 3, with the
subject-level correlation, its interval, and a permutation null that refits the whole
pipeline under within-subject shuffled targets. Left against right hand is deliberately not a decoding check: with 45
movement trials per subject it is a weak contrast in this dataset, and the group-level
search for its signature came up empty (below).

The runner is exercised on the same data: two subjects' epochs are written to FIF, a recipe
computes ERDS over the hand ROI, and the values read back from the TSV bundle must equal
the API's to nine digits. The ``eegfeat`` command itself is run as a subprocess: ``check``
validates without writing and reports a channel the data lacks, ``run`` writes a
per-epoch bundle and a cross-trial bundle per recording with a parseable JSON progress
stream, and a second run is refused until ``--overwrite`` is passed.

What the data taught us
-----------------------

Running real recordings through the library surfaced eight things worth knowing.

*Prediction intervals hold across epochs, not across subjects.* Split and CV+ conformal
intervals at 90 percent nominal covered 88 to 91 percent of exchangeable held-out epochs.
Calibrated on two subjects and applied to a third, the same intervals covered anywhere
from 72 to 99 percent. The docstring says group-disjoint fitting does not establish
coverage for new subjects; this is what that looks like.

*The wavelet scaling is right.* Dividing MNE's Morlet power by the sampling rate gives a
density whose band mean agrees with a Welch PSD of the same window to within 2 percent at
the median and does not move when the recording is resampled from 160 to 80 Hz. Without
that division the two rates would disagree by a factor of two.

*The ERDS onset latency fires on rest trials too.* With a fixed 100 ms persistence an
onset was found on essentially every rest epoch, where nothing happened, in theta, mu and
beta alike. Measuring the false-onset rate against persistence showed that about six
cycles of the band's low edge brings it to roughly 5 percent in every band, so **the
persistence is now expressed in cycles by default** (``min_duration_cycles=6.0``) and the
validation asserts the resulting null rate. The same measurement showed the onset firing
on movement trials at the same rate as on rest trials, so it remains a weak single-trial
detector, and the docstring says so. The post-movement beta rebound was also not visible
at the group level in the 4.5 to 5.8 s window of this task, so nothing about it is
asserted.

*Single-trial percent change is right-skewed.* A trial whose baseline happens to be quiet
reports a change of several hundred percent, and a handful of those pull the mean over
forty-five trials above zero even when most trials show desynchronization. On the percent
scale only about half the subjects showed mu desynchronization in the mean; on the decibel
scale, a symmetric log ratio, every one of the twenty did, and the percent mean sat above
its median in most subjects. **Decibels are now the default for every ERDS measure**, and
a test records the skew that motivated the change. Percent remains available.

*Contralateral dominance did not show up.* Desynchronization was expected to be stronger
over the hemisphere opposite the moving hand. Over twenty subjects, on either scale, with
single electrodes or clusters, the contralateral-minus-ipsilateral difference averaged
about zero. The dataset is known for weak left-versus-right separability; the library
reports what is there, and the suite does not assert what is not.

*The default peak smoothing is for endogenous rhythms.* :func:`~eegfeat.peak_frequency`
smooths over 1 Hz before searching, which suits a broad alpha peak. A steady-state response
is a line a few bins wide; at 1 Hz smoothing it blurs into the shoulder of the alpha peak
sitting just below it, and the search reports alpha. With ``smoothing_hz=0.0`` and a band
that excludes alpha the interpolated peak lands on the stimulation frequency. Narrow
entrained responses need the smoothing turned off.

*Raw phase-amplitude coupling did not resolve slow-oscillation to spindle coupling.* The
mean vector length between 0.5 to 1.5 Hz phase and 12 to 15 Hz amplitude, read on
30-second epochs of two derivations, did not exceed a time-shifted surrogate consistently
in N2 or N3. The effect is real in the literature but small at the sensor level, and the
raw statistic sits well above zero even when the coupling is destroyed. The suite checks
the formula and records the surrogate floor; it does not assert the physiology. Read
:func:`~eegfeat.pac` against a null of your own construction, as its docstring says.

*The frontal derivation does not separate wake from deep sleep.* At Fpz-Cz the wake
epochs of both subjects carry as much power below 4 Hz as slow-wave sleep does, from eye
and movement activity, and their spectral edge sits at 4 Hz in wake as in N3. At Pz-Oz every
descriptor separates the stages cleanly. The sleep checks therefore name the derivation
they read. This is the data, not the code, but it is the kind of thing a feature set should
be validated against before it is trusted.

Observed values
---------------

.. _validation-results:

Recorded from a full run on 2026-09-21, MNE-Python 1.13. The thresholds in the tests sit
well inside these values so that ordinary variation across MNE versions does not fail them.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Check
     - Observed
   * - Band power, mean PSD, uncorrected peak, variance and Hjorth mobility against direct
       NumPy/SciPy computation
     - Agree to better than 1e-12 relative
   * - Hilbert ERDS against MNE multitaper ERDS, 20 subjects, correlation across channels
       and across trial-channel cells
     - 0.79 to 0.99 by channel; 0.79 to 0.98 by cell
   * - Runner TSV bundle against API values, 2 subjects
     - Equal to 1e-9 relative
   * - Mu desynchronization over the hand areas, dB, per-subject mean
     - Mean -3.5 dB, range -5.1 to -2.4, negative in 20 of 20, p = 3e-15
   * - Beta desynchronization, likewise
     - Mean -3.6 dB, range -4.4 to -2.9, negative in 20 of 20, p = 3e-18
   * - Movement against rest, within subject, run-disjoint folds, 1740 epochs
     - Balanced accuracy 0.79 (per subject 0.60 to 0.95)
   * - Movement against rest, leave-one-subject-out
     - Balanced accuracy 0.68
   * - SSVEP: occipital 12 Hz over 15 Hz log power ratio
     - 12 Hz trials at least 0.87, 15 Hz trials at most 0.02
   * - SSVEP: second harmonic, 24 Hz over 30 Hz
     - 12 Hz trials at least 0.49, 15 Hz trials at most 0.37
   * - SSVEP: peak frequency, 11 to 16 Hz band, no smoothing
     - Medians 12.00 and 15.00 Hz; 17 of 20 trials within 0.3 Hz
   * - SSVEP: peak frequency with the defaults, 10 to 17 Hz
     - 3 of 20 trials within 0.3 Hz (reports alpha)
   * - Sleep, Pz-Oz, wake against N3 as area under the curve, two subjects
     - Relative delta 0.91 to 0.95; relative alpha 0.01 to 0.02; relative beta 0.00;
       edge frequency 0.04 to 0.06; centroid 0.08; entropy 0.13 to 0.14; Hjorth mobility
       0.00 to 0.04
   * - Deep sleep against wake, leave-one-subject-out, three subjects, 893 epochs
     - Balanced accuracy 0.96 (per subject 0.91 to 0.98)
   * - Aperiodic slope, 2 to 30 Hz, median by stage, two subjects
     - Pz-Oz wake -1.4 and -0.5, N2 -2.8 and -2.6, N3 -3.2 and -3.2; wake against N3
       AUC at most 0.05; correlation with plain least squares 0.99
   * - Beta bursts over the hand areas, movement against rest, 20 subjects
     - Burst rate 0.21 against 0.46 per second, fraction above threshold 0.18 against
       0.27; lower during movement in 20 of 20 (mu likewise)
   * - SSVEP phase locking across trials, 12 Hz band, occipital
     - ITPC 0.49 driven against 0.31 undriven (chance level 0.32 for ten trials); PPC
       0.19 against 0.02
   * - SSVEP occipital coherence at 12 Hz and at 15 Hz
     - 0.92 in the driven condition against 0.83 to 0.85 in the other
   * - Alpha envelope correlation, resting, neighbouring against distant pairs
     - Raw 0.73 against 0.30; orthogonalized 0.17 against 0.16
   * - Resting microstates, four templates
     - Global explained variance 0.62; median durations 54 to 83 ms
   * - Sample entropy and Higuchi dimension, Pz-Oz, wake against N3
     - Medians 1.17 against 0.49 and 1.67 against 1.25; AUC at most 0.06
   * - Cross-fitted CSP, movement against rest, best component AUC, 20 subjects
     - Mu 0.67 to 0.96 (mean 0.84); beta 0.66 to 0.99 (mean 0.86)
   * - Morlet density against Welch PSD, band means, subject 1
     - Median ratio 0.98 (mu) and 1.02 (beta); log correlation 0.98; resampling to 80 Hz
       changes the density by 0.1 percent at the median
   * - ERD magnitude and duration in dB, movement against rest, 20 subjects
     - Larger and longer during movement in 20 of 20 (beta) and 19 of 20 (mu); ERS
       magnitude smaller in 20 of 20
   * - Onset latency false-positive rate on rest trials, beta
     - Fires on nearly every rest trial at a fixed 100 ms; about 6 percent at the
       six-cycle default (0.46 s at 13 Hz); under 1 percent at 1500 ms
   * - Sleep depth 0 to 3, leave-one-subject-out ridge, three subjects, 2511 epochs
     - Subject-level r 0.88 (interval 0.87 to 0.90, per subject 0.88 to 0.89), R² 0.67;
       50-permutation null within ±0.06, p = 0.02
   * - Held-out permutation importance, sleep depth
     - Pz-Oz carries 0.79 of the total; relative beta and alpha rank first and second
   * - Conformal coverage at 90 percent nominal, exchangeable epochs
     - Split 0.88 to 0.89, CV+ 0.91, quantile 1.00; across subjects 0.72 to 0.99
   * - Multitaper against Welch band power, subject 1
     - Median ratio 0.99; log correlation 0.93; integral equals the edge-interpolated
       trapezoid to 1e-9
   * - Spectral connectivity, eight methods against ``mne_connectivity``
     - All equal to 1e-9; wPLI rises from 0.28 to 0.43 when trials fall from 42 to 10,
       debiased wPLI² moves from 0.06 to 0.09
   * - ERP CORE: single-trial N1 latency, lateral occipital
     - Median 181 ms, interquartile range 165 to 196 ms on a 150 ms window
   * - ERP CORE: P3 area 300 to 500 ms, Pz and CPz
     - Positive on 82 percent of trials; pre-stimulus area centred on zero
   * - ERP CORE: error-related negativity at FCz, 0 to 100 ms after the press
     - Mean +3.9 µV on 346 correct trials against -7.1 µV on 54 errors
