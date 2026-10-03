Dynamics Methods
================

Amplitude change within an epoch, bursts of the analytic envelope, and
time-domain summaries of the waveform. Inputs are :class:`~eegtable.Signal` and
:class:`~eegtable.BandSignal`.

Signatures are in :doc:`/api/dynamics`.

Event-Related Desynchronization and Synchronization (ERDS)
----------------------------------------------------------

ERDS is the change in band-limited power relative to a baseline window
:math:`B`. It is reported per sample as a trace, then summarized inside the
analysis window.

.. math::

   \begin{aligned}
   B_\epsilon &= \max(B, \epsilon), \qquad P_\epsilon(t) = \max(P(t), \epsilon) \\[6pt]
   \text{ERDS}_{\%}(t) &= \frac{P(t) - B_\epsilon}{B_\epsilon} \cdot 100 \\[6pt]
   \text{ERDS}_{\text{dB}}(t) &= 10 \log_{10}\left(\frac{P_\epsilon(t)}{B_\epsilon}\right)
   \end{aligned}

Terms
~~~~~

- :math:`P(t)` is instantaneous power :math:`|z(t)|^2` from the Hilbert
  envelope.
- :math:`B` is mean baseline power.
- :math:`\epsilon` is :math:`10^{-12}` of the largest finite power of that
  epoch and channel. The floor is relative, so it does not depend on the
  recording units.
- The percent numerator is unfloored, so zero power is an exact
  :math:`-100\%` change.

Baseline checks
~~~~~~~~~~~~~~~

A baseline is rejected when any of these holds:

- it is non-finite;
- it is non-positive;
- it is no greater than :math:`10^{-6}` of that channel's mean power over the
  epoch.

The :math:`10^{-6}` guard is relative to the channel, so it does not depend on
the recording units.

**Missing values.** Rejected cells are NaN and carry ``baseline_degenerate``.

**Flags.** A cell whose peak power exceeds :math:`10^4` times its baseline is
reported and flagged ``baseline_extreme_ratio``.

Trace
~~~~~

.. code-block:: python

   power = np.where(np.isfinite(signal.power), signal.power, np.nan)
   baseline_power = np.nanmean(power[..., baseline_mask], axis=-1)
   baseline_power = np.where(degenerate, np.nan, baseline_power)
   floor = 1e-12 * np.nanmax(power, axis=-1)
   baseline_power = np.maximum(baseline_power, floor)
   percent = (power - baseline_power[..., None]) / baseline_power[..., None] * 100.0
   decibels = 10.0 * np.log10(np.maximum(power, floor[..., None]) / baseline_power[..., None])

Summaries
~~~~~~~~~

Summaries use the finite samples :math:`\{t_k\}` inside the analysis window.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Summary
     - Definition
   * - ``mean``
     - Mean of the trace, in percent or in decibels.
   * - ``slope``
     - Ordinary least-squares slope of the trace against time. Needs at least
       three finite samples.
   * - ``erd_magnitude``
     - Mean of :math:`|\text{ERDS}|` over samples with :math:`\text{ERDS} < 0`.
       The value is 0 when no sample is negative.
   * - ``erd_duration``
     - Number of negative samples divided by the sampling rate, in seconds.
   * - ``ers_magnitude``
     - Mean of :math:`\text{ERDS}` over samples with :math:`\text{ERDS} > 0`.
       The value is 0 when no sample is positive.
   * - ``ers_duration``
     - Number of positive samples divided by the sampling rate, in seconds.
   * - ``peak_latency``
     - Time of the maximum of :math:`|\text{ERDS}|`.
   * - ``onset_latency``
     - Start of the first run of samples on which
       :math:`|P(t) - B| > \sigma_B`. See below.
   * - ``rebound_latency``
     - Time of the maximum of :math:`\text{ERDS}` strictly after
       ``peak_latency``.

Onset rule
~~~~~~~~~~

- The run must last at least ``min_duration_cycles`` cycles of the band's low
  edge (default 6), or ``min_duration_ms`` when that is given.
  The required duration is rounded up to whole samples.
- :math:`\sigma_B` is the population standard deviation (divisor :math:`n`) of
  baseline power.
- A non-finite sample breaks the run.
- The test uses raw power, before percent or decibel conversion.

.. warning::

   Under no task effect these summaries are not centered on zero, and the
   duration summaries are not half the window. Instantaneous band power is
   close to exponential, so the trace sits below its baseline mean a fraction
   :math:`1 - e^{-1}` of the time.

   .. list-table::
      :header-rows: 1
      :widths: 25 25 25 25

      * - Measure
        - No effect, dB (default)
        - No effect, percent
        - Value if the trace were symmetric about zero
      * - ``erd_duration``
        - :math:`\approx 63\%` of the window
        - :math:`\approx 63\%` of the window
        - half the window
      * - ``ers_duration``
        - :math:`\approx 37\%` of the window
        - :math:`\approx 37\%` of the window
        - half the window
      * - ``erd_magnitude``
        - :math:`\approx 5.5` dB
        - :math:`\approx 55\text{–}60\%`
        - 0
      * - ``ers_magnitude``
        - :math:`\approx 2.6` dB
        - :math:`\approx 90\text{–}120\%`
        - 0
      * - ``erds_onset_latency``
        - fires on some trials, at a rate that depends on bandwidth
        - fires on some trials, at a rate that depends on bandwidth
        - NaN when nothing happens

   Why the onset rule is in cycles:

   - A single sample exceeds a one-standard-deviation criterion about 14% of
     the time by chance.
   - On rest epochs of a public motor dataset, a fixed 100 ms run still
     produced an onset on essentially every trial in theta, mu, and beta.
   - A narrow-band envelope moves slowly, so neighbouring samples are
     dependent.
   - About six cycles of the band's low edge brought that false-onset rate to
     roughly 5% in each of those bands, which is why the default length is in
     cycles.
   - On that dataset the onset rate on movement trials matched the rate on
     rest trials.

   How to compare:

   - Compare a condition with a null from shuffled labels or from
     pre-stimulus windows.
   - Durations scale with window length, and magnitudes depend on the
     distribution of power.
   - Two windows of different length need that comparison before they are
     compared with each other.

Scale
~~~~~

Every ``erds_*`` function reports decibels by default. Percent is available
with ``normalize="percent"``.

**Why decibels.**

- Percent change on a single trial is right-skewed.
- A quiet baseline turns an ordinary fluctuation into several hundred
  percent, and those trials dominate a mean.
- On twenty subjects of a public motor dataset, the trial-mean percent ERDS
  showed mu desynchronization in about half of the subjects. The decibel mean
  showed it in all twenty.

**Order of averaging.** Two decibel quantities in the library average in a
different order.

- The ``erds_*`` functions average the per-sample dB trace.
- :func:`~eegtable.mean_tfr_power` with a baseline and ``normalize="db"`` takes
  the decibel of the window-mean power.
- For near-exponential instantaneous power the per-sample mean sits about
  2.5 dB below the decibel of the mean.
- On real data the two correlate only moderately across trials.
- Report which one was used.

**Reference.** The ERD/ERS terms and the baseline-referenced interpretation
follow Pfurtscheller and Lopes da Silva (1999). The relative baseline guard,
the onset rule, and the rebound rule are defined above.

Oscillatory Bursts
------------------

A burst is a contiguous run of the band-limited amplitude envelope :math:`E(t)`
above a threshold :math:`\theta`.

Detection
~~~~~~~~~

1. **Threshold.** :math:`\theta` is an envelope quantile (default
   ``threshold=0.75``) or an array supplied by the caller.

   - The quantile is computed separately for each epoch and channel.
   - It is calibrated on the baseline when one is given, and on the analysis
     windows otherwise.
   - A sample is above threshold when :math:`E(t) > \theta`.
   - A non-finite sample is not above threshold, so it ends a run.

2. **Runs.** Contiguous runs are found by differencing. A run that touches the
   window edge is kept.
3. **Minimum duration.** Runs shorter than ``min_duration_ms`` (default
   100 ms, rounded up to whole samples, minimum one) are dropped.

.. code-block:: python

   threshold = np.nanquantile(envelope[..., calibration_mask], q, axis=-1)
   above = envelope > threshold[..., None]
   min_samples = max(1, int(np.ceil(min_duration_ms * sfreq / 1000.0)))
   runs = contiguous_true_runs(above)
   retained = [run for run in runs if len(run) >= min_samples]
   count = len(retained)
   rate = count / (n_times / sfreq)
   fraction_above = np.count_nonzero(above) / np.count_nonzero(np.isfinite(envelope))

The sample conversion allows one floating-point ULP of roundoff at exact
boundaries, so a 140 ms minimum at 100 Hz requires 14 samples, not 15.

An externally supplied threshold array is broadcast to
``(n_epochs, n_channels)`` and used in place of the quantile.

Summaries
~~~~~~~~~

Five summaries are taken from the retained runs.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Summary
     - Definition
   * - ``count``
     - Number of retained bursts.
   * - ``rate``
     - ``count`` divided by the window length in seconds.
   * - ``duration_mean``
     - Mean duration of retained bursts, in seconds.
   * - ``amp_mean``
     - Mean of the peak envelope inside each retained burst.
   * - ``fraction_above``
     - Fraction of finite samples above threshold, computed before the
       duration filter. A missing sample is omitted from this fraction.

**Missing values.** With no retained burst:

- ``count`` and ``rate`` are 0.
- ``duration_mean`` and ``amp_mean`` are NaN.
- ``fraction_above`` is 0 when no sample exceeds :math:`\theta`.

**Reference.** Whitten et al. (2011) and Hughes et al. (2012) detect
oscillations by fitting a 1/f background, applying a chi-square threshold to
power, and requiring a minimum number of cycles (BOSC). The detector here uses
the envelope quantile or the supplied threshold, and ``min_duration_ms``.

Time-Domain Measures
--------------------

These measures summarize one window: ``variance``, ``mean_amplitude``,
``peak_to_peak``, ``area_under_curve``, ``amplitude_quantile``,
``root_mean_square``, ``skewness``, ``kurtosis``, ``line_length``, and
``zero_crossing_rate``. ``hjorth_mobility`` and ``hjorth_complexity`` are
also included.

**Input.**

- On a :class:`~eegtable.Signal` the input is the waveform.
- On a :class:`~eegtable.BandSignal` the input is the envelope.

**Missing values and coverage.**

- Non-finite samples are omitted and counted in ``coverage``.
- ``coverage`` is the window mean of the signal's per-sample coverage, which by
  default is the finite fraction.
- A large finite artifact remains in the summary unless it was rejected before
  extraction.

Simple reductions
~~~~~~~~~~~~~~~~~

The other reductions are the corresponding NumPy reductions with non-finite
samples omitted: ``nanvar``, ``nanmean``, ``nanmax - nanmin``,
``sqrt(nanmean(x**2))``, and ``nanquantile``. Hjorth activity is ``variance``.
Subha, Joseph, Acharya, and Lim (2010) review this family as EEG features.

Area under the curve
~~~~~~~~~~~~~~~~~~~~

``area_under_curve`` applies the trapezoid rule on each contiguous run of
finite samples and sums the runs. A gap contributes nothing. It is not bridged
by a straight line.

Skewness and kurtosis
~~~~~~~~~~~~~~~~~~~~~

- **Skewness** is SciPy's biased Fisher–Pearson :math:`g_1 = m_3 / m_2^{3/2}`.
- **Kurtosis** is Fisher excess :math:`g_2 = m_4 / m_2^2 - 3`.
- NaNs are omitted, and :math:`m_r` uses divisor :math:`n`.

**Missing values.**

- Skewness needs at least three finite samples.
- Kurtosis needs at least four.

**Reference.** Joanes and Gill (1998). See
`scipy.stats.skew <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html>`__
and
`scipy.stats.kurtosis <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html>`__.

Line length
~~~~~~~~~~~

``line_length`` is the mean absolute first difference times the sampling rate,
in the lineage of Esteller et al. (2001). It is a mean, not a cumulative sum.

.. code-block:: python

   clean = np.where(np.isfinite(signal), signal, np.nan)
   finite_differences = np.abs(np.diff(clean))
   finite_differences = np.where(np.isfinite(finite_differences),
                                 finite_differences, np.nan)
   line_length = np.nanmean(finite_differences, axis=-1) * sfreq

Zero-crossing rate
~~~~~~~~~~~~~~~~~~

``zero_crossing_rate`` counts sign changes between successive nonzero finite
samples within each contiguous finite segment and divides by the window length
in seconds (Rice, 1944, 1945). Zeros keep the previous sign. Missing samples
reset it, so a sign difference across a gap is not counted as a crossing.

Hjorth parameters
~~~~~~~~~~~~~~~~~

``hjorth_mobility`` and ``hjorth_complexity`` follow Hjorth (1970).

- Mobility uses the sample difference scaled by the sampling rate, so the
  derivative is per second.
- Mobility is divided by :math:`2\pi` and reported in hertz.
- Mobility therefore does not depend on the sampling rate, apart from the
  finite-difference gain :math:`\sin(\pi f/f_s)/(\pi f/f_s)` near Nyquist.
- Complexity is a ratio of mobilities, so unscaled differences suffice.

.. code-block:: python

   derivative = np.diff(signal) * sfreq
   mobility = np.sqrt(np.nanvar(derivative) / np.nanvar(signal)) / (2.0 * np.pi)
   first = np.diff(signal)
   second = np.diff(signal, n=2)
   complexity = np.sqrt(np.nanvar(second) * np.nanvar(signal)) / np.nanvar(first)

Peak Amplitude and Latency
--------------------------

``peak_amplitude`` is the signed extremum in the window, and ``peak_latency``
is its time.

**Polarity.**

- ``polarity="positive"`` searches the signal.
- ``"negative"`` searches its negation.
- ``"absolute"`` searches its magnitude.
- The returned amplitude is the original signed sample in every case.

**Naming.** The window name is not read as a component label. There is no
inference of N2, P300, or any other named component.

**Prominence.** When ``prominence`` is set:

- Candidates are the local maxima returned by
  `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`__
  at that prominence, and the most prominent is kept.
- An edge sample has no interior prominence, so it is never a candidate.
- Non-finite samples split the search into contiguous finite stretches; gaps
  cannot establish a peak's baseline.
- When no local maximum reaches the prominence, amplitude and latency are NaN.
  A monotonic trend has no qualifying interior peak.
- A tall interior sample with finite neighbours may qualify, however narrow.

**Reference.** Duncan et al. (2009) review windowed peak measures for ERP
components. Report the window, the polarity, and the prominence.

References
----------

* Pfurtscheller, G., & Lopes da Silva, F. H. (1999). *Event-related EEG/MEG
  synchronization and desynchronization: Basic principles*. Clinical
  Neurophysiology, 110(11), 1842--1857.
  `doi:10.1016/S1388-2457(99)00141-8
  <https://doi.org/10.1016/S1388-2457(99)00141-8>`__.
* Whitten, T. A., Hughes, A. M., Dickson, C. T., & Caplan, J. B. (2011). *A
  better oscillation detection method robustly extracts EEG rhythms across
  brain state changes: The human alpha rhythm as a test case*. NeuroImage, 54,
  860--874. `doi:10.1016/j.neuroimage.2010.08.064
  <https://doi.org/10.1016/j.neuroimage.2010.08.064>`__.
* Hughes, A. M., Whitten, T. A., Caplan, J. B., & Dickson, C. T. (2012). *BOSC:
  A better oscillation detection method, extracts both sustained and transient
  rhythms from rat hippocampal recordings*. Hippocampus, 22, 1417--1428.
  `doi:10.1002/hipo.20979 <https://doi.org/10.1002/hipo.20979>`__.
* Subha, D. P., Joseph, P. K., Acharya, U. R., & Lim, C. M. (2010). *EEG signal
  analysis: A survey*. Journal of Medical Systems, 34, 195--212.
  `doi:10.1007/s10916-008-9231-z
  <https://doi.org/10.1007/s10916-008-9231-z>`__.
* Joanes, D. N., & Gill, C. A. (1998). *Comparing measures of sample skewness
  and kurtosis*. The Statistician, 47, 183--189.
  `doi:10.1111/1467-9884.00122 <https://doi.org/10.1111/1467-9884.00122>`__.
* Esteller, R., Echauz, J., Tcheng, T., Litt, B., & Pless, B. (2001). *Line
  length: An efficient feature for seizure onset detection*. Proceedings of
  the 23rd Annual International Conference of the IEEE Engineering in Medicine
  and Biology Society, 1707--1710.
  `doi:10.1109/IEMBS.2001.1020545
  <https://doi.org/10.1109/IEMBS.2001.1020545>`__.
* Rice, S. O. (1944). *Mathematical analysis of random noise*. Bell System
  Technical Journal, 23(3), 282--332.
  `doi:10.1002/j.1538-7305.1944.tb00874.x
  <https://doi.org/10.1002/j.1538-7305.1944.tb00874.x>`__.
* Rice, S. O. (1945). *Mathematical analysis of random noise* (conclusion).
  Bell System Technical Journal, 24(1), 46--156.
  `doi:10.1002/j.1538-7305.1945.tb00453.x
  <https://doi.org/10.1002/j.1538-7305.1945.tb00453.x>`__.
* Hjorth, B. (1970). *EEG analysis based on time domain properties*.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306--310.
  `doi:10.1016/0013-4694(70)90143-4
  <https://doi.org/10.1016/0013-4694(70)90143-4>`__.
* Duncan, C. C., Barry, R. J., Connolly, J. F., Fischer, C., Michie, P. T.,
  Näätänen, R., Polich, J., Reinvang, I., & Van Petten, C. (2009).
  *Event-related potentials in clinical research: Guidelines for eliciting,
  recording, and quantifying mismatch negativity, P300, and N400*. Clinical
  Neurophysiology, 120(11), 1883--1908.
  `doi:10.1016/j.clinph.2009.07.045
  <https://doi.org/10.1016/j.clinph.2009.07.045>`__.
