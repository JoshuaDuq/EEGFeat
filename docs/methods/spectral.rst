Spectral Methods
================

Power integrated over a band, normalization of that power, and summaries of
spectral shape. Each measure reads a :class:`~eegfeat.Spectra` container and
applies to Welch, multitaper, and Morlet input unless a section says otherwise.

Signatures are in :doc:`/api/spectral`.

Spectral Power
--------------

``integrated_band_power`` integrates a PSD over the band limits.

.. math::

   P_B = \int_{f_{\min}}^{f_{\max}} S(f)\,df

Quadrature is piecewise linear and includes interpolated contributions at both
boundaries, so a linear grid and a logarithmic grid represent the same interval.
For an EEG PSD in V²/Hz the integral is in V².

``mean_psd`` divides that integral by the bandwidth and stays in V²/Hz.
``mean_tfr_power`` is the same frequency-weighted mean of Morlet power, also in
V²/Hz. MNE scales its Morlet wavelets to energy 2 (norm :math:`\sqrt{2}`, unit
norm for the real part), so the power it returns for a stationary signal is the one-sided density at the wavelet
frequency, smoothed over the wavelet bandwidth, times the sampling rate.
:meth:`~eegfeat.Spectra.from_tfr` divides that factor out and therefore needs
the sampling rate the TFR was computed at. A decimated TFR reports only the
decimated rate. Without the division, resampling a recording from 250 Hz to
500 Hz doubles the raw power, and an aperiodic offset fitted to it shifts by
:math:`\log_{10}` of the rate ratio.

After that division each value is a density in V²/Hz, smoothed over the
wavelet bandwidth. ``mean_tfr_power`` averages within the band rather than
integrating, so the result is comparable to ``mean_psd``. Integrating it would
give wavelet-smoothed band power in V².
:class:`~eegfeat.Spectra` records which representation it holds. Each function
rejects the other representation.

These functions summarize an estimate MNE has already computed. Welch input is
the short-segment modified periodogram of Welch (1967). Multitaper input is
Thomson (1982). Morlet input follows Morlet, Arens, Fourgeau, and Giard (1982).

Integration weights are the piecewise-linear weights on the closed interval
:math:`[f_{\min}, f_{\max}]`. The grid points on either side of each bound carry
interpolated weight. ``mean_psd`` and ``mean_tfr_power`` average over
the finite bins. ``integrated_band_power`` returns NaN if any weighted bin is
non-finite.

Normalization
-------------

Normalization is applied per channel, then channels are aggregated. An ROI
value is the mean of the per-channel normalized values.

- **log10**: :math:`\log_{10}(\max(P, \epsilon))`
- **log_ratio**: :math:`\log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
- **db**: :math:`10 \log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
- **percent**: :math:`\frac{P - \max(B, \epsilon)}{\max(B, \epsilon)} \cdot 100`

:math:`B` is the same measure computed in the named baseline window, and
:math:`\epsilon = 10^{-20}`. Logarithmic forms floor the numerator and the
denominator. Percent floors the denominator only, so zero power is an exact
:math:`-100\%` change.

**percent** is ERD/ERS% in the sense of Pfurtscheller and Lopes da Silva
(1999), where a negative value is desynchronization. ``log_ratio`` and ``db``
are logarithmic forms of the same baseline ratio.

For baseline-normalized power, coverage is the minimum of the analysis-window
and baseline coverage on that channel. Flags from either window are copied to
the result. The baseline name and its time bounds are stored in the column
metadata.

Wavelet Support Restriction
---------------------------

A Morlet wavelet at frequency :math:`f` with :math:`n_{\text{cycles}}` cycles
has temporal half-support

.. math::

   \tau(f) = \frac{5 n_{\text{cycles}}}{2 \pi f}

The coefficient at time :math:`t` uses samples in
:math:`[t - \tau(f), t + \tau(f)]`. It belongs to an analysis window
:math:`[t_{\min}, t_{\max}]` when that support lies inside the window.

.. math::

   t_{\min} + \tau(f) \le t \le t_{\max} - \tau(f)

Coefficients outside this interval are excluded before the window mean. If
:math:`\tau(f) > (t_{\max} - t_{\min}) / 2`, the frequency has no usable
coefficient in the window and the result is NaN. If no frequency in the band
retains a coefficient, construction raises ``ValueError``.

Peak Frequency
--------------

``peak_frequency`` applies three corrections to the in-band argmax. Each can
be turned off.

**Aperiodic adjustment.** The search runs on
:math:`P(f) / P_{\text{ap}}(f)`, where a pure power law is flat at one. On a
steep spectrum the largest raw bin in the band is the low edge, whatever the
oscillation does. The fit uses ``fit_range``, default
:math:`(\min(2, f_{\min}), \max(40, f_{\max}))`, which extends outside the
analysis band. A slope estimated on a window a few hertz wide is not stable.
With this adjustment the measure name is ``peak_freq_adjusted``.

The split between aperiodic background and oscillatory peaks follows Donoghue
et al. (2020). The fit itself is the robust log-log line in
`Aperiodic Fit`_. It is a straight line. FOOOF also estimates a knee and uses
a different peak model.

**Smoothing.** Each frequency is replaced by the piecewise-linear mean over a
centred window of total width ``smoothing_hz`` (:math:`\pm` ``smoothing_hz``/2),
truncated at the band edges. Bands of three bins or fewer are not smoothed.
Distances are in hertz on both linear and logarithmic grids.
Non-finite bins are left out of the mean. The gap is not filled.

**Prominence guard.** If the maximum is less than ``min_prominence`` above the
band median, in :math:`\log_{10}` units, the reported frequency is the centre
of gravity :math:`\sum f P(f) / \sum P(f)` and ``cog_fallback`` is set.
:math:`P` here is the smoothed (and, if enabled, aperiodic-adjusted) power,
not weighted by bin width.

The retained maximum is refined by parabolic interpolation through the peak bin
and its two neighbours.

.. math::

   \begin{aligned}
   \delta &= \frac{1}{2} \frac{P(f_{k-1}) - P(f_{k+1})}{P(f_{k-1}) - 2 P(f_k) + P(f_{k+1})} \\[6pt]
   f_{\text{peak}} &= f_k + \delta \cdot \frac{f_{k+1} - f_{k-1}}{2}
   \end{aligned}

:math:`k` is the discrete argmax. Interpolation needs at least three bins,
which is also the requirement for an interior maximum. Narrower bands raise.
``interpolate=False`` returns the bin frequency. The vertex is exact on a
uniform grid. On a non-uniform grid the half-spacing step is an approximation.

A zero denominator, or a non-finite :math:`\delta`, is treated as
:math:`\delta = 0`. :math:`\delta` is clipped to :math:`[-0.5, 0.5]`. A maximum
on the first or last bin of the band is not interpolated, and ``edge_hit`` is
set unless the centre-of-gravity fallback applies. Every column also stores ``freq_resolution_hz``, the median in-band bin
spacing.

Spectral Centroid and Bandwidth
-------------------------------

The spectral centroid is the centre of mass of power in the band.

.. math::

   f_c = \frac{\sum_i f_i P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}

Bandwidth is the mass-weighted standard deviation about that centroid.

.. math::

   \text{BW} = \sqrt{\frac{\sum_i (f_i - f_c)^2 P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}}

:math:`\Delta f_i` is from :func:`numpy.gradient`: a central difference in the
interior and one-sided at the band edges.
Weighting by :math:`\Delta f_i` integrates a density. Unequal bins are not
treated as equally probable samples. The two quantities are the spectral centroid
and spread (the first moment and the square root of the second central moment)
in Peeters (2004), adapted here to a PSD.

Spectral Edge Frequency
-----------------------

Spectral edge frequency is the lowest grid frequency at which the cumulative
band power reaches a fraction :math:`\alpha`. The default is 0.95.

.. math::

   \frac{\sum_{i=0}^{k} P(f_i) \Delta f_i}{\sum_{i} P(f_i) \Delta f_i} \ge \alpha

There is no interpolation between bins. If floating-point rounding means no bin
reaches :math:`\alpha`, the last bin is returned. Szeto (1990) used this
quantile for electrocortical maturation, with :math:`\alpha = 0.90`. The
quantile here is the ``percentile`` argument.

Spectral Entropy
----------------

Spectral entropy is the normalized Shannon entropy of the in-band power
distribution, in :math:`[0, 1]`.

.. math::

   \begin{aligned}
   p_i &= \frac{P(f_i)}{\sum_j P(f_j)} \\[6pt]
   H &= -\frac{\sum_i p_i \ln(p_i)}{\ln(N)}
   \end{aligned}

:math:`N` is the number of bins. :math:`H = 1` is uniform power across the
band. :math:`H = 0` is power in a single bin. The undefined term
:math:`0 \ln 0` is omitted. The definition uses one count per bin and requires
an approximately uniform frequency grid. A non-uniform grid is rejected.
Interpolate onto a uniform grid in hertz first. Because the denominator is
:math:`\ln(N)`, values from bands with different bin counts are not comparable.

This is the entropy of the power-spectrum proportions in Inouye et al. (1991).

Aperiodic Fit
-------------

The aperiodic background is a line in log-log coordinates.

.. math::

   \log_{10} P(f) = \text{offset} + \text{slope} \cdot \log_{10} f

An initial least-squares line is fit on ``fit_range``. Residuals are
:math:`r(f) = \log_{10} P(f) - (\text{offset} + \text{slope} \log_{10} f)`.
Points with :math:`r(f) > z \cdot \mathrm{MAD}(r)` are removed and the line is
refit, up to ``max_iterations`` times. The default is :math:`z = 2.5`. Only
positive residuals are removed. Oscillatory peaks sit above the aperiodic
component, and leaving them in flattens the slope. The threshold is
:math:`z \cdot 1.4826\,\mathrm{MAD}(r)`, the normal-consistent MAD. The residual
median enters the MAD but is not subtracted from :math:`r` in the comparison. The loop
stops when MAD is non-finite or below :math:`10^{-12}`, fewer than 5 points
remain, or the mask stops changing. Only positive finite power enters the fit.

The column ``r_squared`` is the coefficient of determination on the points that
survived rejection. Rejected peaks are left out of it. An alpha peak is not a
failure of the line, and scoring the line against that peak is low on ordinary
spectra. The statistic does respond to a knee inside ``fit_range``, and a straight
line through a knee biases the exponent. Donoghue et al. (2020) also report
goodness of fit, though for their full model including peaks.

On :math:`f > 0` the fitted power is
:math:`10^{\text{offset} + \text{slope} \log_{10} f}`, and aperiodic-adjusted
power is the ratio of the observed power to that curve. If the fit fails,
``aperiodic_ratio`` returns NaN for that cell and sets
``aperiodic_fit_failed``. The unadjusted spectrum is not relabelled as
adjusted.

The line is the model above. FOOOF (Donoghue et al., 2020) adds a knee
parameter and a different peak parameterization. Matching those settings is a
separate analysis.

Band Ratio and Asymmetry
------------------------

Ratios and asymmetry are computed from a band-power table. The arithmetic
follows the normalization already stored on that table.

For raw power the ratio of band :math:`A` to band :math:`B` is

.. math::

   R_{A/B} = \frac{P_A}{P_B}

Raw input is divided as stored. Percent input is refused, because a percent
change is a signed deviation and a quotient of two of them is not a power
ratio. For
``log10``, ``log_ratio``, and ``db`` the stored values are subtracted.

.. math::

   R_{A/B} = P_A^{(\mathrm{log})} - P_B^{(\mathrm{log})}

``log10`` input gives :math:`\log_{10}(P_A / P_B)`. ``log_ratio`` input gives
:math:`\log_{10}((P_A / B_A) / (P_B / B_B))`. ``db`` input is ten times that
baseline-relative difference.

For a homologous pair :math:`(L, R)`, raw asymmetry is

.. math::

   A_{L, R} = \frac{P_R - P_L}{P_R + P_L}

Percent input is refused. Logarithmic input subtracts the stored values.

.. math::

   A_{L, R} = P_R^{(\mathrm{log})} - P_L^{(\mathrm{log})}

``log10`` input gives :math:`\log_{10}(P_R / P_L)`. ``log_ratio`` input gives
:math:`\log_{10}((P_R / B_R) / (P_L / B_L))`. ``db`` input is ten times that
difference, in dB. A zero denominator or a zero sum returns NaN.

The output unit names the scale of that result.

.. list-table::
   :header-rows: 1

   * - Input normalization
     - Band ratio
     - Asymmetry
   * - ``"raw"``
     - ``ratio``
     - ``a.u.``
   * - ``"log10"``
     - ``log10 ratio``
     - ``log10 ratio``
   * - ``"log_ratio"``
     - ``log10 ratio``
     - ``log10 ratio``
   * - ``"db"``
     - ``dB``
     - ``dB``

Right-minus-left asymmetry follows the frontal-asymmetry convention of
Davidson, Chapman, Chapman, and Henriques (1990). Log-ratio asymmetry is reported in the
unit of its input, from the table above.

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
  media*. Geophysics, 47(2), 203--221. `doi:10.1190/1.1441328
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
