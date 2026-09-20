Methods
=======

This document details the mathematical formulation and implementation of each spectral feature in ``eegfeat``.

Spectral Power
--------------

``integrated_band_power`` integrates a PSD over exact numerical band boundaries:

.. math::

   P_B = \int_{f_{\min}}^{f_{\max}} S(f)\,df

Piecewise-linear quadrature includes interpolated contributions at both boundaries,
so changing from a linear to a logarithmic grid does not silently change the
represented interval. For EEG PSD in V²/Hz the result is V².

``mean_psd`` divides that integral by band width and retains V²/Hz units.
``mean_tfr_power`` is the corresponding frequency-weighted mean for Morlet
time-frequency power. :class:`~eegfeat.Spectra` records which representation it
contains, and these operations reject the wrong one instead of conflating units.

Normalization
-------------

Power normalization is applied per channel before spatial aggregation:

- **log10**: :math:`\log_{10}(\max(P, \epsilon))`
- **log_ratio**: :math:`\log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
- **db**: :math:`10 \log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
- **percent**: :math:`\frac{P - \max(B, \epsilon)}{\max(B, \epsilon)} \cdot 100`

where :math:`B` is the baseline power in a designated reference window, and :math:`\epsilon = 10^{-20}` is a symmetric floor applied equally to the numerator and denominator to prevent infinite ratios while avoiding numerator bias (in percent normalization, the numerator is unfloored so that a true zero power remains an exact :math:`-100\%` decrease).

Applying normalization per channel before spatial aggregation ensures that region-of-interest (ROI) values reflect the mean of log-ratios rather than the log-ratio of channel means.

Wavelet Support Restriction
---------------------------

When computing spectral power or features across time-frequency representations (TFRs), Morlet wavelet kernels possess frequency-dependent temporal duration. For a Morlet wavelet at center frequency :math:`f` parameterized with :math:`n_{\text{cycles}}` cycles, the temporal half-support is:

.. math::

   \tau(f) = \frac{5 n_{\text{cycles}}}{2 \pi f}

A time-frequency coefficient at time coordinate :math:`t` draws upon underlying signal data from :math:`[t - \tau(f), t + \tau(f)]`. Consequently, a coefficient is mathematically attributable to an analysis window :math:`[t_{\min}, t_{\max}]` if and only if its full temporal support is contained entirely within the window bounds:

.. math::

   t_{\min} + \tau(f) \le t \le t_{\max} - \tau(f)

Bins failing this condition are masked out prior to window averaging. Frequencies where :math:`\tau(f) > (t_{\max} - t_{\min}) / 2` retain no attributable coefficients across the window and are returned as NaN, preventing edge and baseline leakage.

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

**Smoothing.** The spectrum is averaged over frequencies within the requested
``smoothing_hz`` interval around each output frequency. Distances are measured in
hertz, so bandwidth stays fixed on linear and logarithmic grids. Non-finite bins
are excluded without closing the gap.

**Prominence guard.** If the maximum stands less than ``min_prominence`` above
the band median, in :math:`\log_{10}` units, the centre of gravity
:math:`\sum f P(f) / \sum P(f)` is reported instead and the ``cog_fallback``
flag is set. A centre of gravity degrades gracefully when no oscillation is
present; an argmax does not.

The surviving maximum is then refined by parabolic interpolation through the
discrete maximum and its two neighbours:

.. math::

   \begin{aligned}
   \delta &= \frac{1}{2} \frac{P(f_{k-1}) - P(f_{k+1})}{P(f_{k-1}) - 2 P(f_k) + P(f_{k+1})} \\[6pt]
   f_{\text{peak}} &= f_k + \delta \cdot \frac{f_{k+1} - f_{k-1}}{2}
   \end{aligned}

where :math:`k` is the discrete argmax index. Interpolation requires at least
three frequency bins in the band, which is also the definition domain of an
interior maximum; narrower bands raise. Set ``interpolate=False`` to report the
discrete bin frequency itself without interpolation.

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

   \begin{aligned}
   p_i &= \frac{P(f_i)}{\sum_j P(f_j)} \\[6pt]
   H &= -\frac{\sum_i p_i \ln(p_i)}{\ln(N)}
   \end{aligned}

where :math:`N` is the number of frequency bins in the band. A value of 1 indicates uniform power across the band, while 0 indicates concentration in a single bin. This discrete definition requires an approximately uniform frequency grid; non-uniform grids are rejected because bin probabilities and density-weighted spectral mass are different quantities. Recompute or interpolate a non-uniform PSD onto a uniform-Hz grid before calculating entropy. Because the normalization depends on :math:`\ln(N)`, entropy values across bands with different bin counts are not directly comparable.

Aperiodic Fit
-------------

The aperiodic (1/f) background is modeled in log-log space:

.. math::

   \log_{10} P(f) = \text{offset} + \text{slope} \cdot \log_{10} f

Fitting uses iterative peak rejection: an initial least-squares line is fit over ``fit_range``, residuals :math:`r(f) = \log_{10} P(f) - (\text{offset} + \text{slope} \log_{10} f)` are computed, and points with positive residuals exceeding :math:`z \cdot \text{MAD}(r)` (default :math:`z = 2.5`) are rejected before refitting, repeated up to ``max_iterations`` times. Only positive residuals are excluded because oscillatory peaks project above the aperiodic component and would otherwise artificially flatten the estimated slope.

If the fit cannot be estimated, ``aperiodic_ratio`` returns NaNs for that cell
and marks ``aperiodic_fit_failed``. It never labels an unchanged raw spectrum as
aperiodic-adjusted.

Event-Related Desynchronization and Synchronization (ERDS)
----------------------------------------------------------

ERDS measures time-varying power changes in band-limited signals relative to a baseline reference period :math:`B`:

.. math::

   \begin{aligned}
   \text{ERDS}_{\%}(t) &= \frac{P(t) - B}{B} \cdot 100 \\[6pt]
   \text{ERDS}_{\text{dB}}(t) &= 10 \log_{10}\left(\frac{P(t)}{B}\right)
   \end{aligned}

where :math:`P(t)` is the instantaneous power derived from the Hilbert envelope :math:`|z(t)|^2`, and :math:`B` is the mean baseline power. To prevent extreme instability on low-amplitude channels, the baseline is guarded such that :math:`B \ge 10^{-12}\,\text{V}^2`; channels failing this guard evaluate to NaN.

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

``variance``, ``peak_to_peak``, ``mean_amplitude`` and ``area_under_curve``
summarize a series within a window. They accept a raw :class:`~eegfeat.Signal` or
a :class:`~eegfeat.BandSignal`, reading the signal itself in the first case and
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

Sample and Multiscale Entropy
-----------------------------

Sample entropy is the negative log probability that two template vectors matching
over ``order`` (:math:`m`) samples still match over ``order + 1`` (:math:`m + 1`):

.. math::

   H(x, m, r) = -\log \frac{C(m + 1, r)}{C(m, r)}

Two templates match when their Chebyshev distance is strictly below
:math:`r \cdot \sigma_x`. Pairs are counted once. The result is NaN when no
length-:math:`m` pair matches and infinite when some do but no
length-:math:`m+1` pair does; neither case is reported as zero, because zero
entropy means perfect regularity rather than absent evidence.

``multiscale_entropy`` applies the same measure after coarse-graining by
non-overlapping block averages, one column per scale. ``tolerance_mode="original_sd"``
(the default) holds :math:`r\sigma_x` from the original signal fixed across scales,
matching classical MSE. ``"scale_sd"`` recomputes it after coarse-graining and is
recorded as a distinct estimator definition. Coarse blocks containing non-finite
samples remain non-finite, and embedding templates crossing a gap are excluded;
missing samples are never removed in a way that creates new temporal neighbours.

Cost grows with the square of the window length, so entropy on long windows is
markedly slower than the spectral measures.

Band Ratio and Asymmetry
------------------------

Band power ratios and hemispheric asymmetry indices operate on computed band power tables. To maintain mathematical consistency across linear and logarithmic scales, the transformation adapts to the input normalization:

**Band Ratio:** For linear power, the ratio between numerator band :math:`A` and denominator band :math:`B` is:

.. math::

   R_{A/B} = \frac{P_A}{P_B}

When the input is logarithmic (``"log10"``, ``"log_ratio"``, or ``"db"``), division is replaced by subtraction:

.. math::

   R_{A/B} = \log_{10} P_A - \log_{10} P_B = \log_{10}\left(\frac{P_A}{P_B}\right)

**Hemispheric Asymmetry:** For a left-right homologous channel pair :math:`(L, R)`, raw power asymmetry is defined as the normalized difference:

.. math::

   A_{L, R} = \frac{P_R - P_L}{P_R + P_L}

For logarithmic input, the difference of logs corresponds directly to the logarithm of the power ratio:

.. math::

   A_{L, R} = \log_{10} P_R - \log_{10} P_L = \log_{10}\left(\frac{P_R}{P_L}\right)

**Units:** the output unit names the scale the result is on, because the same subtraction means different things on different scales. A ``"db"`` input carries the factor of ten through the subtraction, so the result is a difference in dB rather than a bare log ratio:

.. list-table::
   :header-rows: 1

   * - Input normalization
     - Band ratio
     - Asymmetry
   * - ``"raw"``, ``"percent"``
     - ``ratio``
     - ``a.u.``
   * - ``"log10"``, ``"log_ratio"``
     - ``log10 ratio``
     - ``log10 ratio``
   * - ``"db"``
     - ``dB``
     - ``dB``


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
biased upward and values from different trial counts are not comparable. ITPC
requires at least two valid trials by default and flags cells that do not meet
``min_valid_trials``.

Pairwise phase consistency (:func:`~eegfeat.ppc`) is a distinct estimator of squared
population phase locking without this finite-sample mean bias:

.. math::

   \mathrm{PPC} = \frac{1}{T} \sum_t \left( \frac{2}{N(N - 1)} \sum_{j < k} \cos(\phi_j(t) - \phi_k(t)) \right)

By evaluating the cosine of pairwise relative phase differences across all distinct trial
pairs :math:`j < k`, PPC has an expected value of zero under uniform phase, making values
comparable across conditions or subjects with different numbers of trials.

**ITPC and PPC have one row per trial group, not one per epoch.** Because phase coherence
and consistency are estimated across trials, assigning a per-epoch row would replicate the
identical summary estimate across single trials, introducing severe pseudo-replication in
downstream statistical or predictive models. Feature tables returned by
:func:`~eegfeat.itpc` and :func:`~eegfeat.ppc` carry explicit ``row_labels`` and
:func:`~eegfeat.concat` refuses to join trial-group rows to per-epoch rows without an
explicit broadcasting strategy. Pass ``trials`` to estimate coherence within distinct groups
(for example, per experimental condition).

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
upward by amplitude and phase autocorrelation and should be evaluated against an
empirical null distribution (preferring within-trial circular time shifts over
trial-shuffling to preserve single-trial spectral structure).


Connectivity
------------

``envelope_correlation`` is the Pearson correlation of band envelopes between
every pair of nodes; ``wpli`` is the weighted phase lag index, which discounts
zero-lag coupling and so is less vulnerable to volume conduction.

**wPLI delegates to** ``mne_connectivity.spectral_connectivity_epochs``, the
canonical implementation for cross-spectral phase lag calculation. It is an
optional dependency: ``pip install eegfeat[connectivity]``.

Nodes are channels, or ROIs when ``groups`` is given, in which case the
channel-level matrix is averaged within each ROI block and a node's own block
excludes the diagonal. Both measures are estimated across trials, so results have
one row per trial group; see :func:`~eegfeat.itpc`.

Graph Measures
--------------

``global_efficiency`` and ``clustering_coefficient`` take a pairwise table and
reduce it to one summary value per band and window, analogous to how :func:`~eegfeat.band_ratio`
summarizes a power table. Both graph metrics are computed directly without
third-party network graph dependencies.

Global efficiency converts nonzero edge weights :math:`w_{ij}` into path distances
:math:`L_{ij} = 1/|w_{ij}|`, treats zero and non-finite weights as absent edges,
computes all-pairs shortest paths via Floyd-Warshall, and averages inverse distance
across all node pairs. A disconnected pair contributes exactly zero:

.. math::

   E_{\text{global}} = \frac{2}{N (N - 1)} \sum_{i < j} \frac{1}{d_{ij}}

where :math:`N` is the number of network nodes.

The clustering coefficient **binarizes** the connectivity matrix at user-specified ``threshold`` :math:`\theta` (:math:`A_{ij} = 1` if :math:`|w_{ij}| > \theta`, else :math:`0`), and computes the average local clustering coefficient over nodes with degree :math:`k_i \ge 2`:

.. math::

   \begin{aligned}
   C_i &= \frac{(A^3)_{ii}}{k_i (k_i - 1)} = \frac{2 T_i}{k_i (k_i - 1)} \\[6pt]
   C &= \frac{1}{|\{i : k_i \ge 2\}|} \sum_{i : k_i \ge 2} C_i
   \end{aligned}

where :math:`(A^3)_{ii}` is the diagonal entry of the cubed adjacency matrix (representing twice the number of triangles :math:`T_i` containing node :math:`i`), and :math:`k_i = \sum_j A_{ij}` is the node degree. Nodes with :math:`k_i < 2` are excluded from the average rather than counted as zero, so networks too sparse to contain a triangle evaluate to NaN rather than an artificially suppressed value.


Microstates
-----------

Templates are clustered from the topographies at local maxima of the global field power (GFP):

.. math::

   \text{GFP}(t) = \sqrt{\frac{1}{N_{\text{channels}}} \sum_{c=1}^{N_{\text{channels}}} \left( V_c(t) - \bar{V}(t) \right)^2 }

where :math:`\bar{V}(t)` is the instantaneous average-reference potential. GFP peak maps are sign-normalized such that the channel with the largest absolute amplitude is positive, and partitioned into :math:`K` prototype classes using :math:`k`-means clustering.

Continuous EEG samples are subsequently assigned to the prototype template exhibiting the highest absolute spatial correlation:

.. math::

   s(t) = \arg\max_k \frac{|V(t)^T \mu_k|}{\|V(t)\| \|\mu_k\|}

Segments shorter than ``min_duration_ms`` are absorbed into neighbouring states: the longer one, or split between them on a tie.

**Template fitting pools across trials; the measures do not.** ``fit_on`` names which trials may contribute topographies, as a cross-validation fold requires. The default uses every trial, which is right for description and leaks for prediction. Assignment and every measure derived from it are per epoch, so the returned feature tables have one row per epoch.

Cluster indices are arbitrary, so unmatched templates are labelled ``state1``
onward. A-D labels require one-to-one topographic matching to an identified
reference set and are not assigned automatically. The segmentation exposes its
templates and global explained variance. From the resulting sequence :math:`s(t)`,
four temporal statistics are derived:

- **coverage**: Fractional occupancy time: :math:`\frac{1}{T} \sum_t \mathbb{I}[s(t) = k]` (compositional, sums to 1 across states).
- **duration**: Mean continuous dwell time per visit in milliseconds (evaluates to NaN if state :math:`k` was never entered).
- **occurrence**: Number of distinct visits per second (evaluates to :math:`0.0` if never entered).
- **transitions**: Directional transition probability between successive segments:

.. math::

   T_{i \to j} = \frac{N_{i \to j}}{\sum_{m \ne i} N_{i \to m}} \quad (i \ne j)

Self-transitions (:math:`i = j`) are omitted from segment-based transition matrices.

Segmentation requires scikit-learn: ``pip install eegfeat[microstates]``.
