Methods
=======

This document details the mathematical formulation and implementation of each spectral feature in ``eegfeat``.

Band Power
----------

Band power is computed as a trapezoidally weighted mean across the in-band frequency bins:

.. math::

   \bar{P} = \frac{\sum_{i} P(f_i) w_i}{\sum_{i} w_i}

where :math:`w_i` are the trapezoidal integration weights along the frequency axis:

.. math::

   w_0 = \frac{f_1 - f_0}{2}, \quad w_i = \frac{f_{i+1} - f_{i-1}}{2}, \quad w_{N-1} = \frac{f_{N-1} - f_{N-2}}{2}

This weights each spectral estimate by the width of the frequency interval it represents, estimating the band integral divided by the band width :math:`\int_{f_{\min}}^{f_{\max}} P(f) df / (f_{\max} - f_{\min})`. Weighting by interval width is essential on logarithmic or non-uniform frequency grids, where unweighted bin averaging disproportionately biases power toward densely sampled lower frequencies.

Normalization
-------------

Power normalization is applied per channel before spatial aggregation:

- **log10**: :math:`\log_{10}(\max(P, \epsilon))`
- **log_ratio**: :math:`\log_{10}(\max(P, \epsilon) / \max(B, \epsilon))`
- **db**: :math:`10 \log_{10}(\max(P, \epsilon) / \max(B, \epsilon))`

where :math:`B` is the baseline power in a designated reference window, and :math:`\epsilon = 10^{-20}` is a symmetric floor applied equally to the numerator and denominator to prevent infinite ratios while avoiding numerator bias.

Applying normalization per channel before spatial aggregation ensures that region-of-interest (ROI) values reflect the mean of log-ratios rather than the log-ratio of channel means.

Peak Frequency
--------------

A bare argmax is a poor peak estimator on real spectra, so ``peak_frequency``
applies three corrections by default, each of which can be switched off.

**Aperiodic adjustment.** The fitted 1/f component is divided out first, so the
search runs on :math:`P(f) / P_{\text{ap}}(f)` where a pure power law is flat at
one. Without this step a steep spectrum reports the low edge of the band
whatever the oscillation is doing, because the largest raw value in the band is
simply the leftmost one. The fit spans ``fit_range``, defaulting to
:math:`(\min(2, f_{\min}), \max(40, f_{\max}))`, which deliberately reaches
outside the band: a 1/f slope estimated from a five-hertz window is not a 1/f
slope. This sets the measure name to ``peak_freq_adjusted``.

**Smoothing.** The spectrum is averaged over a window of ``smoothing_hz`` before
the search, so a single noisy bin cannot win. The average ignores non-finite bins
and renormalizes by how many contributed, rather than closing the gap and
averaging across it.

**Prominence guard.** If the maximum stands less than ``min_prominence`` above
the band median, in :math:`\log_{10}` units, the centre of gravity
:math:`\sum f P(f) / \sum P(f)` is reported instead and the ``cog_fallback``
flag is set. A centre of gravity degrades gracefully when no oscillation is
present; an argmax does not.

The surviving maximum is then refined by parabolic interpolation through the
discrete maximum and its two neighbours:

.. math::

   \delta = \frac{1}{2} \frac{P(f_{k-1}) - P(f_{k+1})}{P(f_{k-1}) - 2 P(f_k) + P(f_{k+1})}

.. math::

   f_{\text{peak}} = f_k + \delta \cdot \frac{f_{k+1} - f_{k-1}}{2}

where :math:`k` is the discrete argmax index. Interpolation requires at least
three frequency bins in the band, which is also the definition domain of an
interior maximum; narrower bands raise. Set ``interpolate=False`` to report the
bin frequency itself, which is what the reference pipeline does.

If the discrete maximum falls on the first or last bin of the band, the
``edge_hit`` flag is set, indicating that the true peak may lie outside the
evaluated band. Every column additionally reports ``freq_resolution_hz``, the
median in-band bin spacing, so the precision the grid could support is visible
alongside the estimate.

Spectral Centroid and Bandwidth
-------------------------------

The spectral centroid represents the spectral center of mass within a band:

.. math::

   f_c = \frac{\sum_i f_i P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}

Spectral bandwidth is the mass-weighted standard deviation around the centroid:

.. math::

   \text{BW} = \sqrt{\frac{\sum_i (f_i - f_c)^2 P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}}

Both measures utilize central-difference frequency weights :math:`\Delta f_i` (:func:`numpy.gradient`), consistent with standard spectral descriptor conventions.

Spectral Edge Frequency
-----------------------

Spectral edge frequency (SEF) is the frequency below which a specified fraction (:math:`\alpha`, default 0.95) of the band power is concentrated:

.. math::

   \frac{\sum_{i=0}^{k} P(f_i) \Delta f_i}{\sum_{i} P(f_i) \Delta f_i} \ge \alpha

The value is computed directly via search without interpolation, returning the frequency grid coordinate that first meets or exceeds the cumulative threshold.

Spectral Entropy
----------------

Spectral entropy measures the uniformity of the spectral distribution within a band, normalized to :math:`[0, 1]`:

.. math::

   p_i = \frac{P(f_i) \Delta f_i}{\sum_j P(f_j) \Delta f_j}

.. math::

   H = -\frac{\sum_i p_i \ln(p_i)}{\ln(N)}

where :math:`N` is the number of frequency bins in the band. A value of 1 indicates uniform power across the band, while 0 indicates concentration in a single bin. Because the normalization depends on :math:`\ln(N)`, entropy values across bands with different bin counts are not directly comparable.

Aperiodic Fit
-------------

The aperiodic (1/f) background is modeled in log-log space:

.. math::

   \log_{10} P(f) = \text{offset} + \text{slope} \cdot \log_{10} f

Fitting uses iterative peak rejection: an initial least-squares line is fit over ``fit_range``, residuals :math:`r(f) = \log_{10} P(f) - (\text{offset} + \text{slope} \log_{10} f)` are computed, and points with positive residuals exceeding :math:`z \cdot \text{MAD}(r)` (default :math:`z = 2.5`) are rejected before refitting, repeated up to ``max_iterations`` times. Only positive residuals are excluded because oscillatory peaks project above the aperiodic component and would otherwise artificially flatten the estimated slope.

Event-Related Desynchronization and Synchronization (ERDS)
----------------------------------------------------------

ERDS measures time-varying power changes in band-limited signals relative to a baseline reference period :math:`B`:

.. math::

   \text{ERDS}_{\%}(t) = \frac{P(t) - B}{B} \cdot 100

.. math::

   \text{ERDS}_{\text{dB}}(t) = 10 \log_{10}\left(\frac{P(t)}{B}\right)

where :math:`P(t)` is the instantaneous power derived from the Hilbert envelope. Summary measures are computed within defined analysis windows:

- **mean**: Mean percentage or decibel excursion across the window.
- **slope**: Least-squares linear rate of change over time.
- **erd_magnitude**: Mean magnitude of negative excursions (:math:`P < B`).
- **erd_duration**: Total duration in seconds spent in desynchronization.
- **ers_magnitude**: Mean magnitude of positive excursions (:math:`P > B`).
- **ers_duration**: Total duration in seconds spent in synchronization.
- **peak_latency**: Time of the maximum absolute excursion within the window.
- **onset_latency**: Earliest time where :math:`|\text{ERDS}(t)|` exceeds the baseline coefficient of variation (:math:`\sigma_B / \mu_B \cdot 100`).
- **rebound_latency**: Time of the maximum value occurring strictly after the peak latency.

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

``variance``, ``peak_to_peak``, ``mean_amplitude`` and ``area_under_curve``
summarize a series within a window. They accept a raw :class:`~eegfeat.Signal` or
a :class:`~eegfeat.BandSignal`, reading the signal itself in the first case and
the envelope in the second.

Non-finite samples are excluded and reported through ``coverage``, rather than
poisoning the window as they do in the reference implementation, which returns
NaN if any sample is non-finite.

``area_under_curve`` integrates by the trapezoid rule over each contiguous run of
finite samples and sums them. A gap is skipped rather than interpolated across,
so missing data contributes nothing instead of contributing a straight line.

Peak Amplitude and Latency
--------------------------

``peak_amplitude`` returns the signed value of the extremum in a window and
``peak_latency`` its time. Which extremum is found is set by ``polarity``:
``"positive"`` searches the signal, ``"negative"`` its negation, ``"absolute"``
its magnitude. The returned amplitude is always signed.

**Polarity is an explicit argument, never inferred from the window's name.** The
reference pipeline reads the first letter of the label, so a window called
``noxious`` silently searches for a negative peak and one called ``post`` for a
positive one. Component-name parsing of the ``N2`` and ``P300`` form is likewise
not implemented: those conventions belong to a paradigm, not to a measurement.

``prominence``, when given, confines the search to local maxima meeting that
prominence and takes the most prominent. This matters where the extremum of a
window sits at its edge on a monotonic trend, which is not a peak at all. It does
not reject narrow spikes: an isolated tall sample is highly prominent by
definition.

Sample and Multiscale Entropy
-----------------------------

Sample entropy is the negative log probability that two template vectors matching
over ``order`` samples still match over ``order + 1``:

.. math::

   H(x, m, r) = -\log \frac{C(m + 1, r)}{C(m, r)}

Two templates match when their Chebyshev distance is strictly below
:math:`r \cdot \sigma_x`. Pairs are counted once. The result is NaN when no
length-:math:`m` pair matches and infinite when some do but no
length-:math:`m+1` pair does; neither case is reported as zero, because zero
entropy means perfect regularity rather than absent evidence.

``multiscale_entropy`` applies the same measure after coarse-graining by
non-overlapping block averages, one column per scale. The tolerance is recomputed
from each coarse-grained series, following the reference, so it tracks the
variance surviving the averaging. Note that coarse-graining removes non-finite
samples before blocking, which closes gaps rather than preserving sample
positions.

Cost grows with the square of the window length, so entropy on long windows is
markedly slower than the spectral measures.

What Is Deliberately Absent
---------------------------

The reference pipeline computes an ``snr`` and a ``muscle`` ratio. Neither is
implemented here, because both are named for an interpretation rather than for
what they compute: ``snr`` is a band-power density ratio that asserts 1-30 Hz is
signal and 40-80 Hz is noise, which is false for any study of gamma; ``muscle``
is a high-frequency power fraction that asserts the high frequencies are muscle.
Both are available through :func:`~eegfeat.band_power` and
:func:`~eegfeat.band_ratio` with bands the caller chooses, which puts the
assumption in the call where it can be seen and argued with.


Inter-Trial Phase Coherence
---------------------------

.. math::

   \mathrm{ITPC} = \frac{1}{T} \sum_t
   \left| \frac{1}{N} \sum_n e^{i \phi_n(t)} \right|

One means the phase is identical on every trial at that latency, zero that it is
uniformly distributed. **Trials are averaged first and time second.** Reversing
the order measures something else entirely: a phase that sweeps over time but is
identical across trials gives one under the correct order and nearly zero under
the reverse.

Under the null of uniform phase the expected value is about
:math:`1/\sqrt{N}`, not zero, so coherence from a small number of trials is
biased upward and values from different trial counts are not comparable.

**ITPC has one row per trial group, not one per epoch.** It is estimated across
trials, so a per-epoch row would be the same number repeated, and a model fitted
on it would treat one estimate as many independent observations. The reference
implementation broadcasts, and its own documentation calls that
pseudo-replication; the table returned here carries ``row_labels`` and
:func:`~eegfeat.concat` refuses to join it to per-epoch features. Pass ``trials``
to estimate within groups, for example one row per condition.

Phase-Amplitude Coupling
------------------------

Mean vector length, the amplitude-weighted resultant of the slow band's phase:

.. math::

   \mathrm{MVL} = \frac{\left| \sum_t A(t) e^{i \phi(t)} \right|}{\sum_t A(t)}

Normalizing by the summed amplitude makes the value independent of overall power
and so comparable across channels and trials; without it the result scales with
amplitude.

Computed within each trial, so it carries no cross-trial leakage and has one row
per epoch. **No surrogate correction is applied.** A raw coupling value is biased
upward by amplitude and phase autocorrelation and should be read against a null
you construct, not absolutely. The reference offers trial-shuffle and
circular-shift surrogates; trial shuffling mixes information across trials, so if
you build a null here, prefer a within-trial circular shift.
