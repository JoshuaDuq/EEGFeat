Dynamics Methods
================

Measures of how amplitude evolves within an epoch: power change relative to a
baseline window, discrete bursts of the analytic envelope, and time-domain
descriptors of the waveform itself. These are estimated from
:class:`~eegfeat.Signal` and :class:`~eegfeat.BandSignal` containers rather than
from a spectrum.

Signatures for these functions are in :doc:`/api/dynamics`.

Event-Related Desynchronization and Synchronization (ERDS)
----------------------------------------------------------

ERDS measures time-varying power changes in band-limited signals relative to a baseline reference period :math:`B`:

.. math::

   \begin{aligned}
   B_\epsilon &= \max(B, \epsilon), \qquad P_\epsilon(t) = \max(P(t), \epsilon) \\[6pt]
   \text{ERDS}_{\%}(t) &= \frac{P(t) - B_\epsilon}{B_\epsilon} \cdot 100 \\[6pt]
   \text{ERDS}_{\text{dB}}(t) &= 10 \log_{10}\left(\frac{P_\epsilon(t)}{B_\epsilon}\right)
   \end{aligned}

where :math:`P(t)` is the instantaneous power derived from the Hilbert envelope
:math:`|z(t)|^2`, :math:`B` is the mean baseline power, and
:math:`\epsilon = 10^{-20}`. The percent numerator is deliberately *not*
floored, so zero power is an exact :math:`-100\%` change. A baseline is
rejected when it is non-finite, non-positive, or no more than :math:`10^{-6}` of
that channel's mean power over the whole epoch. Such cells evaluate to NaN and
carry the ``baseline_degenerate`` flag; the relative guard avoids imposing a
unit-dependent absolute power floor.

The exact trace construction is:

.. code-block:: python

   power = np.where(np.isfinite(signal.power), signal.power, np.nan)
   baseline_power = np.nanmean(power[..., baseline_mask], axis=-1)
   baseline_power = np.maximum(baseline_power, 1e-20)
   percent = (power - baseline_power[..., None]) / baseline_power[..., None] * 100.0
   decibels = 10.0 * np.log10(np.maximum(power, 1e-20) / baseline_power[..., None])

Summary measures are evaluated over discrete finite sample points :math:`\{t_k\}_{k=1}^K` within defined analysis windows:

- **mean**: Mean percentage or decibel excursion: :math:`\bar{E} = \frac{1}{K} \sum_{k=1}^K \text{ERDS}(t_k)`.
- **slope**: Ordinary least squares linear slope of :math:`\text{ERDS}(t_k)` against :math:`t_k`, requiring at least three finite samples.
- **erd_magnitude**: Mean magnitude of negative excursions: :math:`\frac{1}{|K_-|} \sum_{t_k \in K_-} |\text{ERDS}(t_k)|`, where :math:`K_- = \{t_k : \text{ERDS}(t_k) < 0\}` (returns :math:`0.0` if :math:`K_- = \emptyset`).
- **erd_duration**: Cumulative desynchronization duration: :math:`|K_-| / f_s` in seconds.
- **ers_magnitude**: Mean magnitude of positive excursions: :math:`\frac{1}{|K_+|} \sum_{t_k \in K_+} \text{ERDS}(t_k)`, where :math:`K_+ = \{t_k : \text{ERDS}(t_k) > 0\}` (returns :math:`0.0` if :math:`K_+ = \emptyset`).
- **ers_duration**: Cumulative synchronization duration: :math:`|K_+| / f_s` in seconds.
- **peak_latency**: Latency of the maximum absolute excursion: :math:`t^* = \arg\max_{t_k} |\text{ERDS}(t_k)|`.
- **onset_latency**: Earliest latency where the absolute raw-power departure from
  baseline exceeds one baseline standard deviation:
  :math:`\min \{t_k : |P(t_k)-B| > \sigma_B\}`. The criterion is evaluated before
  percent or decibel reporting, so normalization choice cannot move the onset.
- **rebound_latency**: Latency of the maximal excursion occurring strictly after the peak latency: :math:`\arg\max_{t_k > t^*} \text{ERDS}(t_k)`.

The ERD/ERS terminology and baseline-referenced power interpretation follow
the synthesis by `Gert Pfurtscheller and F. H. Lopes da Silva (1999)
<https://doi.org/10.1016/S1388-2457(99)00141-8>`__, which distinguishes
frequency-specific changes in ongoing activity from phase-locked ERPs. The
``erds_*`` functions then apply explicit window summaries to that trace; the
relative baseline-degeneracy guard and onset/rebound definitions are
implementation-specific additions and should be reported with the feature
values.

Oscillatory Bursts
------------------

Oscillatory bursts are identified as contiguous suprathreshold excursions of the band-limited amplitude envelope:

1. Envelope thresholding: :math:`E(t) > \theta`, where :math:`\theta` is either an intra-trial envelope quantile or an externally provided threshold array.
2. Interval bounding: Contiguous runs above threshold are detected via differencing, closing intervals that touch window boundaries.
3. Duration filtering: Intervals with durations shorter than ``min_duration_ms`` are discarded.

From the surviving burst intervals, five measures are extracted per window:

- **count**: Total number of detected bursts surviving duration filtering.
- **rate**: Burst frequency in bursts per second (:math:`\text{count} / T_{\text{window}}`).
- **duration_mean**: Mean duration of surviving bursts in seconds.
- **amp_mean**: Mean peak envelope amplitude across surviving bursts.
- **fraction_above**: Overall fraction of samples above threshold prior to duration filtering.

In implementation terms, the detector is:

.. code-block:: python

   # Scalar threshold: q is calibrated on the baseline, or on analysis windows
   # when no baseline is supplied. An ndarray threshold is used directly.
   threshold = np.nanquantile(envelope[..., calibration_mask], q, axis=-1)
   # threshold = provided_threshold  # broadcast to (n_epochs, n_channels)
   above = envelope > threshold[..., None]
   min_samples = max(1, round(min_duration_ms * sfreq / 1000.0))
   runs = contiguous_true_runs(above)
   retained = [run for run in runs if len(run) >= min_samples]
   count = len(retained)
   rate = count / (n_times / sfreq)
   fraction_above = np.count_nonzero(above) / np.count_nonzero(np.isfinite(envelope))

Runs touching a window edge are retained; a missing envelope sample is excluded
from ``fraction_above`` rather than counted as a below-threshold sample.

Thresholding an analytic amplitude and requiring a minimum duration are the
core ideas of the BOSC family of oscillation detectors. The relevant sources
are `Tara A. Whitten, Adam M. Hughes, Clayton T. Dickson, and Jeremy B. Caplan
(2011) <https://doi.org/10.1016/j.neuroimage.2010.08.064>`__ and `Adam M.
Hughes, Tara A. Whitten, Jeremy B. Caplan, and Clayton T. Dickson (2012)
<https://doi.org/10.1002/hipo.20979>`__. This package intentionally implements a
transparent envelope-quantile or user-threshold detector, not BOSC: it does not
fit a background 1/f spectrum, use a chi-square power threshold, or impose a
cycle-count criterion. Those differences are part of the estimator definition.



Time-Domain Measures
--------------------

``variance``, ``mean_amplitude``, ``peak_to_peak``, ``area_under_curve``,
``amplitude_quantile``, ``root_mean_square``, ``skewness``, ``kurtosis``,
``line_length`` and ``zero_crossing_rate`` summarize a series within a window.
``hjorth_mobility`` and ``hjorth_complexity`` provide the standard Hjorth
parameters. They accept a raw :class:`~eegfeat.Signal` or a
:class:`~eegfeat.BandSignal`, reading the signal itself in the first case and
the envelope in the second.

Non-finite samples are excluded and reported through ``coverage``. Coverage is
a finite-data measure, not an artifact detector: large finite artifacts remain
numerically valid unless rejected before feature extraction. This ensures
window summaries reflect only valid, finite electrophysiological data without
silent zero-filling or whole-window invalidation.

``area_under_curve`` integrates by the trapezoid rule over each contiguous run of
finite samples and sums them. A gap is skipped rather than interpolated across,
so missing data contributes nothing instead of contributing a straight line.

The remaining reductions are direct sample statistics (with non-finite samples
omitted):

.. code-block:: python

   variance = np.nanvar(signal)
   mean_amplitude = np.nanmean(signal)
   peak_to_peak = np.nanmax(signal) - np.nanmin(signal)
   root_mean_square = np.sqrt(np.nanmean(signal ** 2))
   amplitude_quantile = np.nanquantile(signal, q)
   zero_crossing_rate = count_sign_changes(signal) / (n_times / sfreq)

Here ``count_sign_changes`` compares successive nonzero finite samples; zeros and
gaps preserve the previous sign and do not themselves create crossings.

For ``area_under_curve``, the implementation sums
``scipy.integrate.trapezoid(signal[finite_run], times[finite_run])`` separately
for each contiguous finite run. For peaks, it searches ``signal``, ``-signal``,
or ``abs(signal)`` according to ``polarity`` and returns the original signed
sample at the selected index; if ``prominence`` is supplied, local maxima are
selected with `scipy.signal.find_peaks
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`__.

These waveform summaries are conventional descriptive statistics rather than
single named EEG algorithms. Their use as EEG feature families is reviewed by
`D. Puthankattil Subha, Paul K. Joseph, Rajendra Acharya U., and Choo Min Lim
(2010) <https://doi.org/10.1007/s10916-008-9231-z>`__. The distributional
estimators for skewness and kurtosis follow the sample-statistic conventions
compared by `D. N. Joanes and C. A. Gill (1998)
<https://doi.org/10.1111/1467-9884.00122>`__. ``line_length`` is the mean
absolute first difference multiplied by sampling frequency, not a cumulative
sum, although its lineage is the absolute first difference used in EEG
detection by `R. Esteller, J. Echauz,
T. Tcheng, B. Litt, and B. Pless (2001)
<https://doi.org/10.1109/IEMBS.2001.1020545>`__, and ``zero_crossing_rate`` is
the discrete crossing-rate analogue of the level-crossing analysis of `S. O.
Rice (1944) <https://doi.org/10.1002/j.1538-7305.1944.tb00874.x>`__.

The exact line-length reduction is:

.. code-block:: python

   clean = np.where(np.isfinite(signal), signal, np.nan)
   finite_differences = np.abs(np.diff(clean))
   finite_differences = np.where(np.isfinite(finite_differences),
                                 finite_differences, np.nan)
   line_length = np.nanmean(finite_differences, axis=-1) * sfreq

The distributional defaults are SciPy's biased Fisher--Pearson skewness and
Fisher excess kurtosis with NaNs omitted: ``scipy.stats.skew(...,
nan_policy="omit")`` and ``scipy.stats.kurtosis(..., nan_policy="omit")``.
The implementation withholds skewness with fewer than three finite samples and
kurtosis with fewer than four.
See the authoritative SciPy definitions for `skew
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html>`__
and `kurtosis
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html>`__.
For finite samples with mean :math:`\bar{x}`, these defaults are
:math:`g_1 = m_3 / m_2^{3/2}` and Fisher excess
:math:`g_2 = m_4 / m_2^2 - 3`, where :math:`m_r` is the biased central moment
with divisor :math:`n`.

``hjorth_mobility`` and ``hjorth_complexity`` are the named exception: they are
the time-domain parameters introduced by `Bo Hjorth (1970)
<https://doi.org/10.1016/0013-4694(70)90143-4>`__. The implementation's
sample-difference form makes mobility dependent on the sampling frequency and
therefore reports it in hertz.

Their numerical kernels are:

.. code-block:: python

   derivative = np.diff(signal) * sfreq
   mobility = np.sqrt(np.nanvar(derivative) / np.nanvar(signal)) / (2.0 * np.pi)
   first = np.diff(signal)
   second = np.diff(signal, n=2)
   complexity = np.sqrt(np.nanvar(second) * np.nanvar(signal)) / np.nanvar(first)

Peak Amplitude and Latency
--------------------------

``peak_amplitude`` returns the signed value of the extremum in a window and
``peak_latency`` its time. Which extremum is found is set by ``polarity``:
``"positive"`` searches the signal, ``"negative"`` its negation, ``"absolute"``
its magnitude. The returned amplitude is always signed.

**Polarity is an explicit parameter (``"positive"``, ``"negative"``, or ``"absolute"``).**
Peak extraction is decoupled from window nomenclature or paradigm-specific ERP component
labels (such as ``N2`` or ``P300``), ensuring unambiguous measurement semantics across
arbitrary experimental designs.

``prominence``, when given, confines the search to local maxima meeting that
prominence and takes the most prominent. This matters where the extremum of a
window sits at its edge on a monotonic trend, which is not a peak at all. It does
not reject narrow spikes: an isolated tall sample is highly prominent by
definition.

Peak amplitude and peak latency are standard windowed ERP scores, not labels for
any particular component. The measurement convention and its limitations are
summarized by `Connie C. Duncan, Robert J. Barry, John F. Connolly, Catherine
Fischer, Patricia T. Michie, Risto Näätänen, John Polich, Ivar Reinvang, and
Cyma Van Petten (2009) <https://doi.org/10.1016/j.clinph.2009.07.045>`__. In
particular, the window, polarity, and prominence criterion must be reported;
the function does not infer a paradigm-specific N2, P300, or other component.

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
