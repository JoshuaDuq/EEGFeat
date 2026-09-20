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

The three functions are summaries of an already estimated spectral
representation; they do not introduce a new PSD estimator. When the input was
produced by Welch averaging, its statistical provenance is the short-segment
modified-periodogram estimator of `Peter D. Welch (1967)
<https://doi.org/10.1109/TAU.1967.1161901>`__. When it was produced by
multitaper estimation, the relevant method is `David J. Thomson (1982)
<https://doi.org/10.1109/PROC.1982.12433>`__. For ``mean_tfr_power``, the
underlying Morlet construction follows `Jean Morlet, G. Arens, E. Fourgeau,
and D. Giard (1982) <https://doi.org/10.1190/1.1441328>`__. The integration
and averaging performed here are deterministic numerical summaries of those
estimates.

The implementation first constructs exact piecewise-linear integration weights
on the requested band and then reduces finite bins with those weights. In
pseudocode, the core reductions are:

.. code-block:: python

   weights = band_integration_weights(freqs, fmin, fmax)
   selected = weights > 0.0
   finite = np.isfinite(power[..., selected])
   selected_power = power[..., selected]
   selected_weights = weights[selected]
   weighted_sum = np.sum(np.where(finite, selected_power * selected_weights, 0.0), axis=-1)
   weight_sum = np.sum(np.where(finite, selected_weights, 0.0), axis=-1)
   mean = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
   integral = np.where(finite.all(axis=-1), weighted_sum, np.nan)

Thus ``mean_psd`` and ``mean_tfr_power`` return a weighted mean, whereas
``integrated_band_power`` returns the weighted integral and invalidates a cell
if any bin required by the integration is non-finite.

Normalization
-------------

Power normalization is applied per channel before spatial aggregation:

- **log10**: :math:`\log_{10}(\max(P, \epsilon))`
- **log_ratio**: :math:`\log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
- **db**: :math:`10 \log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
- **percent**: :math:`\frac{P - \max(B, \epsilon)}{\max(B, \epsilon)} \cdot 100`

where :math:`B` is the baseline power in a designated reference window, and :math:`\epsilon = 10^{-20}` is a symmetric floor applied equally to the numerator and denominator to prevent infinite ratios while avoiding numerator bias (in percent normalization, the numerator is unfloored so that a true zero power remains an exact :math:`-100\%` decrease).

Applying normalization per channel before spatial aggregation ensures that region-of-interest (ROI) values reflect the mean of log-ratios rather than the log-ratio of channel means.

The corresponding implementation is:

.. code-block:: python

   epsilon = 1e-20
   floored_power = np.maximum(power, epsilon)
   floored_baseline = np.maximum(baseline, epsilon)
   log_ratio = np.log10(floored_power / floored_baseline)
   normalized_db = 10.0 * log_ratio
   normalized_percent = (power - floored_baseline) / floored_baseline * 100.0

The percent branch intentionally retains the unfloored numerator, so zero
power is reported as an exact :math:`-100\%` change.

The baseline-referenced percentage and logarithmic forms are relative-power
estimators rather than a separate spectral decomposition. Their interpretation
as event-related synchronization or desynchronization follows the baseline
logic formalized by `Gert Pfurtscheller and F. H. Lopes da Silva (1999)
<https://doi.org/10.1016/S1388-2457(99)00141-8>`__; the sign convention and
units remain the explicit choices of this package.

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

The separation of aperiodic background and oscillatory peaks is grounded in the
spectral parameterization of `Thomas Donoghue, Matar Haller, Erik J. Peterson,
Paroma Varma, Priyadarshini Sebastian, Richard Gao, Torben Noto, Antonio H.
Lara, Joni D. Wallis, Robert T. Knight, Avgusta Shestyuk, and Bradley Voytek
(2020) <https://doi.org/10.1038/s41593-020-00744-x>`__. This implementation
uses that scientific distinction but deliberately exposes a compact robust
aperiodic fit rather than claiming to reproduce the complete FOOOF model.

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

At the implementation level, the final interpolation is equivalent to:

.. code-block:: python

   denominator = left - 2.0 * centre + right
   delta = 0.5 * (left - right) / denominator
   delta = np.clip(np.nan_to_num(delta, nan=0.0), -0.5, 0.5)
   peak_frequency = frequency[index] + delta * neighbour_spacing

The code uses ``delta = 0`` at a zero denominator and for edge maxima; a
prominence failure instead returns the power-weighted centre of gravity.

Spectral Centroid and Bandwidth
-------------------------------

The spectral centroid represents the spectral center of mass within a band:

.. math::

   f_c = \frac{\sum_i f_i P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}

Spectral bandwidth is the mass-weighted standard deviation around the centroid:

.. math::

   \text{BW} = \sqrt{\frac{\sum_i (f_i - f_c)^2 P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}}

Both measures utilize central-difference frequency weights :math:`\Delta f_i` (:func:`numpy.gradient`), consistent with standard spectral descriptor conventions.

The centroid and bandwidth definitions are the first two spectral moments
described by `Geoffroy Peeters (2004)
<https://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf>`__.
The frequency weighting used here is adapted to a PSD density, so the
calculation integrates spectral mass rather than treating unequal frequency
bins as equally probable observations.

The calculation is implemented as:

.. code-block:: python

   weights = np.gradient(freqs)
   mass = power * weights
   total = np.sum(mass, axis=-1)
   centroid = np.sum(mass * freqs, axis=-1) / total
   bandwidth = np.sqrt(np.sum(mass * (freqs - centroid[..., None]) ** 2,
                              axis=-1) / total)

Spectral Edge Frequency
-----------------------

Spectral edge frequency (SEF) is the frequency below which a specified fraction (:math:`\alpha`, default 0.95) of the band power is concentrated:

.. math::

   \frac{\sum_{i=0}^{k} P(f_i) \Delta f_i}{\sum_{i} P(f_i) \Delta f_i} \ge \alpha

The value is computed directly via search without interpolation, returning the frequency grid coordinate that first meets or exceeds the cumulative threshold.

SEF is a cumulative-power quantile, not a peak-frequency estimator. Its use as
an EEG summary is exemplified by `Hazel H. Szeto (1990)
<https://doi.org/10.1203/00006450-199003000-00018>`__, who defined the spectral
edge as the frequency below which a specified percentage of electrocortical
power resides. The package keeps the quantile configurable rather than
assuming the 90-percent convention used in that study.

The exact reduction is:

.. code-block:: python

   mass = power * weights
   cumulative = np.cumsum(mass, axis=-1) / np.sum(mass, axis=-1, keepdims=True)
   index = np.argmax(cumulative >= percentile, axis=-1)
   edge_frequency = freqs[index]

If no bin reaches the requested fraction because of floating-point rounding,
the implementation returns the final frequency bin.

Spectral Entropy
----------------

Spectral entropy measures the uniformity of the spectral distribution within a band, normalized to :math:`[0, 1]`:

.. math::

   \begin{aligned}
   p_i &= \frac{P(f_i)}{\sum_j P(f_j)} \\[6pt]
   H &= -\frac{\sum_i p_i \ln(p_i)}{\ln(N)}
   \end{aligned}

where :math:`N` is the number of frequency bins in the band. A value of 1 indicates uniform power across the band, while 0 indicates concentration in a single bin. This discrete definition requires an approximately uniform frequency grid; non-uniform grids are rejected because bin probabilities and density-weighted spectral mass are different quantities. Recompute or interpolate a non-uniform PSD onto a uniform-Hz grid before calculating entropy. Because the normalization depends on :math:`\ln(N)`, entropy values across bands with different bin counts are not directly comparable.

This is the normalized entropy of the power-spectrum proportions introduced for
EEG irregularity by `T. Inouye, K. Shinosaki, H. Sakamoto, S. Toi, S. Ukai,
A. Iyama, Y. Katsuda, and M. Hirano (1991)
<https://doi.org/10.1016/0013-4694(91)90138-T>`__. The implementation retains
their information-theoretic interpretation while making the frequency-grid
assumption explicit.

The implementation uses uniform-bin mass and omits the undefined ``0 log 0``
term:

.. code-block:: python

   mass = np.where(np.isfinite(power), power, 0.0)
   n_bins = mass.shape[-1]
   probability = mass / np.sum(mass, axis=-1, keepdims=True)
   entropy_terms = np.where(probability > 0.0,
                            probability * np.log(probability), 0.0)
   entropy = -np.sum(entropy_terms, axis=-1) / np.log(n_bins)

Aperiodic Fit
-------------

The aperiodic (1/f) background is modeled in log-log space:

.. math::

   \log_{10} P(f) = \text{offset} + \text{slope} \cdot \log_{10} f

Fitting uses iterative peak rejection: an initial least-squares line is fit over ``fit_range``, residuals :math:`r(f) = \log_{10} P(f) - (\text{offset} + \text{slope} \log_{10} f)` are computed, and points with positive residuals exceeding :math:`z \cdot \text{MAD}(r)` (default :math:`z = 2.5`) are rejected before refitting, repeated up to ``max_iterations`` times. Only positive residuals are excluded because oscillatory peaks project above the aperiodic component and would otherwise artificially flatten the estimated slope.

If the fit cannot be estimated, ``aperiodic_ratio`` returns NaNs for that cell
and marks ``aperiodic_fit_failed``. It never labels an unchanged raw spectrum as
aperiodic-adjusted.

For each cell, the fitted curve and adjustment are evaluated as:

.. code-block:: python

   log_frequency = np.log10(freqs)
   fitted_log_power = offset + slope * log_frequency
   aperiodic_power = 10.0 ** fitted_log_power
   adjusted_power = power / aperiodic_power

Only positive finite power values enter the fit; non-positive frequencies are
not transformed, and failed fits return NaN with ``aperiodic_fit_failed``.

The robust rejection loop is equivalent to:

.. code-block:: python

   keep = np.isfinite(power) & (power > 0.0)
   for _ in range(max_iterations):
       slope, offset = np.polyfit(log_frequency[keep],
                                  np.log10(power[keep]), 1)
       residual = np.log10(power) - (offset + slope * log_frequency)
       mad = median_abs_deviation(residual[keep], scale="normal")
       tightened = keep & (residual <= peak_rejection_z * mad)
       if (not np.isfinite(mad) or mad <= np.finfo(float).eps
               or tightened.sum() < 5 or np.array_equal(tightened, keep)):
           break
       keep = tightened

This is intentionally a compact project-specific robust fit, not a claim of
bit-for-bit equivalence to FOOOF or another spectral-parameterization package.

The fitted line is therefore a named scientific model, while the positive-
residual rejection rule and its stopping criteria are implementation choices.
They should not be cited as an exact reimplementation of any software package
unless those settings are reproduced independently.

Band Ratio and Asymmetry
------------------------

Band power ratios and hemispheric asymmetry indices operate on computed band power tables. To maintain mathematical consistency across linear and logarithmic scales, the transformation adapts to the input normalization:

**Band Ratio:** For linear power, the ratio between numerator band :math:`A` and denominator band :math:`B` is:

.. math::

   R_{A/B} = \frac{P_A}{P_B}

When the input is logarithmic (``"log10"``, ``"log_ratio"``, or ``"db"``), division is replaced by subtraction of the already-normalized values:

.. math::

   R_{A/B} = P_A^{(\mathrm{log})} - P_B^{(\mathrm{log})}

For ``"log10"`` and ``"log_ratio"`` this equals
:math:`\log_{10}(P_A/P_B)`; for ``"db"`` it equals
:math:`10\log_{10}(P_A/P_B)` in decibels.

**Hemispheric Asymmetry:** For a left-right homologous channel pair :math:`(L, R)`, raw power asymmetry is defined as the normalized difference:

.. math::

   A_{L, R} = \frac{P_R - P_L}{P_R + P_L}

For logarithmic input, the same normalized difference is used:

.. math::

   A_{L, R} = P_R^{(\mathrm{log})} - P_L^{(\mathrm{log})}

Thus ``"log10"`` and ``"log_ratio"`` produce
:math:`\log_{10}(P_R/P_L)`, whereas ``"db"`` produces the corresponding
dB difference.

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

The band-ratio operation is a deterministic arithmetic transform of two
already-computed band-power columns; no unique historical estimator is claimed
for it. The left-right asymmetry convention is the power-asymmetry framework
used by `Richard J. Davidson, John P. Chapman, Linda J. Chapman, and John B.
Henriques (1990) <https://doi.org/10.1111/j.1469-8986.1990.tb01970.x>`__.
Because log-ratio asymmetry is a different scale, it is reported as such rather
than being silently called the same raw-power index.

The corresponding elementwise operations are:

.. code-block:: python

   if normalization in {"log10", "log_ratio", "db"}:
       band_ratio = numerator - denominator
       asymmetry = right - left
   else:
       band_ratio = np.where(denominator != 0.0,
                             numerator / denominator, np.nan)
       total = right + left
       asymmetry = np.where(total != 0.0,
                            (right - left) / total, np.nan)

References
----------

* Welch, P. D. (1967). *The use of fast Fourier transform for the estimation of
  power spectra: A method based on time averaging over short, modified
  periodograms*. IEEE Transactions on Audio and Electroacoustics, 15(2),
  70--73. `doi:10.1109/TAU.1967.1161901
  <https://doi.org/10.1109/TAU.1967.1161901>`__.
* Thomson, D. J. (1982). *Spectrum estimation and harmonic analysis*.
  Proceedings of the IEEE, 70(9), 1055--1096.
  `doi:10.1109/PROC.1982.12433 <https://doi.org/10.1109/PROC.1982.12433>`__.
* Morlet, J., Arens, G., Fourgeau, E., & Giard, D. (1982). *Wave propagation
  and sampling theory; Part I, Complex signal and scattering in multilayered
  media*. Geophysics, 47(2). `doi:10.1190/1.1441328
  <https://doi.org/10.1190/1.1441328>`__.
* Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R.,
  Noto, T., Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek,
  B. (2020). *Parameterizing neural power spectra into periodic and aperiodic
  components*. Nature Neuroscience, 23, 1655--1665.
  `doi:10.1038/s41593-020-00744-x
  <https://doi.org/10.1038/s41593-020-00744-x>`__.
* Peeters, G. (2004). *A large set of audio features for sound description
  (similarity and classification) in the CUIDADO project*. IRCAM technical
  report. `Report PDF
  <https://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf>`__.
* Szeto, H. H. (1990). *Spectral edge frequency as a simple quantitative
  measure of the maturation of electrocortical activity*. Pediatric Research,
  27, 289--292. `doi:10.1203/00006450-199003000-00018
  <https://doi.org/10.1203/00006450-199003000-00018>`__.
* Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S., Iyama, A.,
  Katsuda, Y., & Hirano, M. (1991). *Quantification of EEG irregularity by use
  of the entropy of the power spectrum*. Electroencephalography and Clinical
  Neurophysiology, 79(3), 204--210.
  `doi:10.1016/0013-4694(91)90138-T
  <https://doi.org/10.1016/0013-4694(91)90138-T>`__.
* Pfurtscheller, G., & Lopes da Silva, F. H. (1999). *Event-related EEG/MEG
  synchronization and desynchronization: Basic principles*. Clinical
  Neurophysiology, 110(11), 1842--1857.
  `doi:10.1016/S1388-2457(99)00141-8
  <https://doi.org/10.1016/S1388-2457(99)00141-8>`__.
* Davidson, R. J., Chapman, J. P., Chapman, L. J., & Henriques, J. B. (1990).
  *Asymmetrical brain electrical activity discriminates between
  psychometrically-matched verbal and spatial cognitive tasks*. Psychophysiology,
  27, 528--543. `doi:10.1111/j.1469-8986.1990.tb01970.x
  <https://doi.org/10.1111/j.1469-8986.1990.tb01970.x>`__.
