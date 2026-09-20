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
   \text{ERDS}_{\%}(t) &= \frac{P(t) - B}{B} \cdot 100 \\[6pt]
   \text{ERDS}_{\text{dB}}(t) &= 10 \log_{10}\left(\frac{P(t)}{B}\right)
   \end{aligned}

where :math:`P(t)` is the instantaneous power derived from the Hilbert envelope
:math:`|z(t)|^2`, and :math:`B` is the mean baseline power. A baseline is
rejected when it is non-finite, non-positive, or no more than :math:`10^{-6}` of
that channel's mean power over the whole epoch. Such cells evaluate to NaN and
carry the ``baseline_degenerate`` flag; the relative guard avoids imposing a
unit-dependent absolute power floor.

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

