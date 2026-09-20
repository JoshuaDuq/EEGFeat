Spectral Methods
================

How power is integrated over a band, how it is normalized, and how the shape of
the spectrum is summarized. Every measure here is estimated from a
:class:`~eegfeat.Spectra` container, so it applies equally to Welch, multitaper,
and Morlet input unless a section says otherwise.

Signatures for these functions are in :doc:`/api/spectral`.

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

For baseline-normalized spectral power, coverage is the minimum of the analysis
and baseline coverage per channel, and flags from either window propagate to the
result. The baseline name and time bounds are included in the computation metadata.

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


