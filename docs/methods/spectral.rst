Spectral Methods
================

Power integrated over a band, normalization of that power, and summaries of
spectral shape. Each measure reads a :class:`~eegfeat.Spectra` container and
applies to Welch, multitaper, and Morlet input unless a section says otherwise.

Signatures are in :doc:`/api/spectral`.

Spectral Power
--------------

Band power is the PSD integrated over the band limits. The mean forms divide
that integral by the bandwidth, or average Morlet power over the band.

Integrated Band Power
~~~~~~~~~~~~~~~~~~~~~

``integrated_band_power`` integrates a PSD over the band limits.

.. math::

   P_B = \int_{f_{\min}}^{f_{\max}} S(f)\,df

- **Quadrature**: piecewise linear, with interpolated contributions at both
  boundaries. A linear grid and a logarithmic grid therefore represent the same
  interval.
- **Units**: for an EEG PSD in V²/Hz the integral is in V².
- **Missing values**: NaN if any weighted bin is non-finite.

Mean PSD and Mean TFR Power
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mean_psd`` divides the integral above by the bandwidth and stays in V²/Hz.
``mean_tfr_power`` is the same frequency-weighted mean of Morlet power, also in
V²/Hz.

- **Units**: V²/Hz for both. After the sampling-rate division described below,
  each ``mean_tfr_power`` value is a density in V²/Hz, smoothed over the wavelet
  bandwidth.
- **Averaging**: ``mean_tfr_power`` averages within the band rather than
  integrating, so the result is comparable to ``mean_psd``. Integrating it would
  give wavelet-smoothed band power in V².
- **Missing values**: both average over the finite bins.
- **Representation**: :class:`~eegfeat.Spectra` records which representation it
  holds. Each function rejects the other representation.

Morlet Scaling
~~~~~~~~~~~~~~

MNE scales its Morlet wavelets to energy 2 (norm :math:`\sqrt{2}`, unit norm for
the real part). The power it returns for a stationary signal is therefore the
one-sided density at the wavelet frequency, smoothed over the wavelet bandwidth,
times the sampling rate.

- **Correction**: :meth:`~eegfeat.Spectra.from_tfr` divides that factor out and
  therefore needs the sampling rate the TFR was computed at.
- **Decimated TFR**: it reports only the decimated rate.
- **Why**: without the division, resampling a recording from 250 Hz to 500 Hz
  doubles the raw power, and an aperiodic offset fitted to it shifts by
  :math:`\log_{10}` of the rate ratio.

Integration Weights
~~~~~~~~~~~~~~~~~~~

Weights are the piecewise-linear weights on the closed interval
:math:`[f_{\min}, f_{\max}]`. The grid points on either side of each bound carry
interpolated weight.

Sources
~~~~~~~

These functions summarize an estimate MNE has already computed.

- **Welch**: the short-segment modified periodogram of Welch (1967).
- **Multitaper**: Thomson (1982).
- **Morlet**: Morlet, Arens, Fourgeau, and Giard (1982).

Normalization
-------------

Normalization is applied per channel, then channels are aggregated. An ROI
value is the mean of the per-channel normalized values.

Here :math:`P` is the power in the analysis window and :math:`B` is the same
measure computed in the named baseline window.

.. list-table::
   :header-rows: 1

   * - Option
     - Definition
   * - ``log10``
     - :math:`\log_{10}(\max(P, \epsilon))`
   * - ``log_ratio``
     - :math:`\log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
   * - ``db``
     - :math:`10 \log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
   * - ``percent``
     - :math:`\frac{P - \max(B, \epsilon)}{\max(B, \epsilon)} \cdot 100`

Power Floor
~~~~~~~~~~~

:math:`\epsilon` is :math:`10^{-12}` of the largest finite power of that epoch
and channel, over every window and the baseline.

- **Why relative**: the floor does not depend on the recording units. A fixed
  floor would flatten a source estimate in A·m, whose power sits many decades
  below EEG in V².
- **Missing values**: an epoch and channel with no positive power has no floor,
  and every value is NaN.
- **Logarithmic forms**: the numerator and the denominator are floored.
- **Percent**: only the denominator is floored, so zero power is an exact
  :math:`-100\%` change.

Interpretation
~~~~~~~~~~~~~~

``percent`` is ERD/ERS% in the sense of Pfurtscheller and Lopes da Silva
(1999), where a negative value is desynchronization. ``log_ratio`` and ``db``
are logarithmic forms of the same baseline ratio.

Coverage and Metadata
~~~~~~~~~~~~~~~~~~~~~

For baseline-normalized power:

- **Coverage**: the minimum of the analysis-window and baseline coverage on that
  channel.
- **Flags**: flags from either window are copied to the result.
- **Metadata**: the baseline name and its time bounds are stored in the column
  metadata.

Wavelet Support Restriction
---------------------------

Only Morlet coefficients whose full wavelet support lies inside the analysis
window and the available TFR time range enter the window mean. Requested bounds
outside that range, including infinite whole-segment bounds, are intersected
with the available range before restricting wavelet support. Keep any padding
in the TFR passed to ``Spectra.from_tfr``; cropped-away padding cannot establish
support for the remaining coefficients.

A Morlet wavelet at frequency :math:`f` with :math:`n_{\text{cycles}}` cycles
has temporal half-support

.. math::

   \tau(f) = \frac{5 n_{\text{cycles}}}{2 \pi f}

The coefficient at time :math:`t` uses samples in
:math:`[t - \tau(f), t + \tau(f)]`. For the intersected window
:math:`[t_{\min}, t_{\max}]`, its support must satisfy

.. math::

   t_{\min} + \tau(f) \le t \le t_{\max} - \tau(f)

Coefficients outside this interval are excluded before the window mean.

**Missing values**

- If :math:`\tau(f) > (t_{\max} - t_{\min}) / 2`, the frequency has no usable
  coefficient in the window and the result is NaN.
- If no frequency in the band retains a coefficient, construction raises
  ``ValueError``.

Peak Frequency
--------------

``peak_frequency`` returns the frequency of the largest in-band power, after
three corrections to the argmax. Each can be turned off.

Aperiodic Adjustment
~~~~~~~~~~~~~~~~~~~~

The search runs on :math:`P(f) / P_{\text{ap}}(f)`, where a pure power law is
flat at one. On a steep spectrum the largest raw bin in the band is the low
edge, whatever the oscillation does.

- **Fit range**: ``fit_range``, default
  :math:`(\min(2, f_{\min}), \max(40, f_{\max}))`, which extends outside the
  analysis band. A slope estimated on a window a few hertz wide is not stable.
- **Measure name**: with this adjustment the measure name is
  ``peak_freq_adjusted``.
- **Fit**: the robust log-log line in `Aperiodic Fit`_. It is a straight line.
  FOOOF also estimates a knee and uses a different peak model.
- **Reference**: the split between aperiodic background and oscillatory peaks
  follows Donoghue et al. (2020).

Smoothing
~~~~~~~~~

Each frequency is replaced by the piecewise-linear mean over a centred window of
total width ``smoothing_hz`` (:math:`\pm` ``smoothing_hz``/2), truncated at the
band edges.

- **Short bands**: bands of three bins or fewer are not smoothed.
- **Grid**: distances are in hertz on both linear and logarithmic grids.
- **Missing values**: non-finite bins are left out of the mean. The gap is not
  filled.

Prominence Guard
~~~~~~~~~~~~~~~~

If the maximum is less than ``min_prominence`` above the band median, in
:math:`\log_{10}` units, the reported frequency is the centre of gravity
:math:`\sum f P(f) \Delta f / \sum P(f) \Delta f` and ``cog_fallback`` is set.

- **Power used**: :math:`P` here is the smoothed (and, if enabled,
  aperiodic-adjusted) power. Trapezoidal bin widths :math:`\Delta f` account
  for the frequency spacing, including on logarithmic grids.

Parabolic Interpolation
~~~~~~~~~~~~~~~~~~~~~~~

The retained maximum is refined by parabolic interpolation through the peak bin
and its two neighbours, using their actual frequency coordinates. Let
:math:`h_L=f_k-f_{k-1}` and :math:`h_R=f_{k+1}-f_k`.

.. math::

   \begin{aligned}
   s_L &= \frac{P(f_k)-P(f_{k-1})}{h_L}, \qquad
   s_R = \frac{P(f_{k+1})-P(f_k)}{h_R} \\[6pt]
   a &= \frac{s_R-s_L}{h_L+h_R}, \qquad b = s_L + a h_L \\[6pt]
   \delta &= -\frac{b}{2a}, \qquad f_{\text{peak}} = f_k + \delta
   \end{aligned}

:math:`k` is the discrete argmax.

- **Minimum bins**: interpolation needs at least three bins, which is also the
  requirement for an interior maximum. Narrower bands raise.
- **Disabling**: ``interpolate=False`` returns the bin frequency.
- **Accuracy**: the fitted parabola's vertex is exact on both uniform and
  non-uniform grids; the offset is bounded by the midpoints to neighbouring bins.
- **Degenerate cases**: a zero denominator, or a non-finite :math:`\delta`, is
  treated as :math:`\delta = 0`.
- **Clipping**: :math:`\delta` is clipped to :math:`[-h_L/2, h_R/2]` in Hz.
- **Flags**: a maximum on the first or last bin of the band is not interpolated,
  and ``edge_hit`` is set unless the centre-of-gravity fallback applies.
- **Stored resolution**: every column also stores ``freq_resolution_hz``, the
  median in-band bin spacing.

Spectral Centroid and Bandwidth
-------------------------------

The spectral centroid is the centre of mass of power in the band.

.. math::

   f_c = \frac{\sum_i f_i P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}

Bandwidth is the mass-weighted standard deviation about that centroid.

.. math::

   \text{BW} = \sqrt{\frac{\sum_i (f_i - f_c)^2 P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}}

- **Bin width**: :math:`\Delta f_i` is from :func:`numpy.gradient`, a central
  difference in the interior and one-sided at the band edges.
- **Why weight**: weighting by :math:`\Delta f_i` integrates a density. Unequal
  bins are not treated as equally probable samples.
- **Reference**: the two quantities are the spectral centroid and spread (the
  first moment and the square root of the second central moment) in Peeters
  (2004), adapted here to a PSD.

Spectral Edge Frequency
-----------------------

Spectral edge frequency is the lowest grid frequency at which the cumulative
band power reaches a fraction :math:`\alpha`. The default is 0.95.

.. math::

   \frac{\sum_{i=0}^{k} P(f_i) \Delta f_i}{\sum_{i} P(f_i) \Delta f_i} \ge \alpha

- **Interpolation**: none between bins.
- **Rounding**: if floating-point rounding means no bin reaches :math:`\alpha`,
  the last bin is returned.
- **Argument**: the quantile is the ``percentile`` argument.
- **Reference**: Szeto (1990) used this quantile for electrocortical maturation,
  with :math:`\alpha = 0.90`.

Spectral Entropy
----------------

Spectral entropy is the normalized Shannon entropy of the in-band power
distribution, in :math:`[0, 1]`.

.. math::

   \begin{aligned}
   p_i &= \frac{P(f_i)}{\sum_j P(f_j)} \\[6pt]
   H &= -\frac{\sum_i p_i \ln(p_i)}{\ln(N)}
   \end{aligned}

:math:`N` is the number of bins.

- **Range**: :math:`H = 1` is uniform power across the band. :math:`H = 0` is
  power in a single bin.
- **Zero terms**: the undefined term :math:`0 \ln 0` is omitted.
- **Grid**: the definition uses one count per bin and requires an approximately
  uniform frequency grid. A non-uniform grid is rejected. Interpolate onto a
  uniform grid in hertz first.
- **Comparability**: because the denominator is :math:`\ln(N)`, values from bands
  with different bin counts are not comparable.
- **Reference**: the entropy of the power-spectrum proportions in Inouye et al.
  (1991).

Aperiodic Fit
-------------

The aperiodic background is a line in log-log coordinates.

.. math::

   \log_{10} P(f) = \text{offset} + \text{slope} \cdot \log_{10} f

Robust Fitting
~~~~~~~~~~~~~~

An initial least-squares line is fit on ``fit_range``. Residuals are
:math:`r(f) = \log_{10} P(f) - (\text{offset} + \text{slope} \log_{10} f)`.
Points with :math:`r(f) > z \cdot \mathrm{MAD}(r)` are removed and the line is
refit, up to ``max_iterations`` times.

- **Threshold**: the default is :math:`z = 2.5`. The threshold is
  :math:`z \cdot 1.4826\,\mathrm{MAD}(r)`, the normal-consistent MAD.
- **Median**: the residual median enters the MAD but is not subtracted from
  :math:`r` in the comparison.
- **One-sided**: only positive residuals are removed. Oscillatory peaks sit above
  the aperiodic component, and leaving them in flattens the slope.
- **Input**: only positive finite power enters the fit.
- **Passband**: ``fit_range`` must stay within the recording's reported filter
  bounds. Including suppressed frequencies biases the fitted background, as
  illustrated in `FOOOF's filtering example
  <https://fooof-tools.github.io/fooof/auto_examples/processing/plot_line_noise.html>`_.
- **Stopping**: the loop stops when any of these holds.

  - MAD is non-finite or below :math:`10^{-12}`.
  - Fewer than 5 points remain.
  - The mask stops changing.

Goodness of Fit
~~~~~~~~~~~~~~~

The column ``r_squared`` is the coefficient of determination on the points that
survived rejection. Rejected peaks are left out of it.

- **Why exclude peaks**: an alpha peak is not a failure of the line, and scoring
  the line against that peak is low on ordinary spectra.
- **Sensitivity**: the statistic does respond to a knee inside ``fit_range``, and
  a straight line through a knee biases the exponent.
- **Reference**: Donoghue et al. (2020) also report goodness of fit, though for
  their full model including peaks.

Adjusted Power
~~~~~~~~~~~~~~

On :math:`f > 0` the fitted power is
:math:`10^{\text{offset} + \text{slope} \log_{10} f}`, and aperiodic-adjusted
power is the ratio of the observed power to that curve.

- **Missing values**: if the fit fails, ``aperiodic_ratio`` returns NaN for that
  cell and sets ``aperiodic_fit_failed``. The unadjusted spectrum is not
  relabelled as adjusted.
- **Units**: the ratio is dimensionless and records
  ``representation="aperiodic_ratio"``. Physical PSD and TFR power reducers
  reject this representation. The DC bin is undefined and has zero coverage.

Periodic Power
~~~~~~~~~~~~~~

``periodic_power`` is the frequency-weighted mean of that ratio over a band,
weighted as ``mean_psd`` weighs power.

- **Reference value**: a pure power law gives 1 in every band.
- **Broadband changes**: a broadband change the line can follow, a gain or a tilt
  of the whole spectrum as from a movement or muscle artifact, moves the fit and
  leaves the ratio unchanged. An oscillation changes the ratio.
- **Contrast with band power**: band power sums the two, so a broadband rise can
  hide an oscillatory fall.
- **Units**: the ratio is dimensionless, so a PSD and time-frequency power give
  the same value.
- **Normalization**: the normalizations are those above, with :math:`B` the
  periodic power of the baseline window, itself divided by that window's own fit.
- **Missing values**: a cell whose fit fails is NaN and carries
  ``aperiodic_fit_failed``.

Relation to FOOOF
~~~~~~~~~~~~~~~~~

The line is the model above. FOOOF (Donoghue et al., 2020) adds a knee parameter
and a different peak parameterization. Matching those settings is a separate
analysis.

Band Ratio and Asymmetry
------------------------

Ratios and asymmetry are computed from a band-power table. The arithmetic
follows the normalization already stored on that table.

Band Ratio
~~~~~~~~~~

For raw power the ratio of band :math:`A` to band :math:`B` is

.. math::

   R_{A/B} = \frac{P_A}{P_B}

Raw input is divided as stored. Percent input is refused, because a percent
change is a signed deviation and a quotient of two of them is not a power ratio.

For ``log10``, ``log_ratio``, and ``db`` the stored values are subtracted.

.. math::

   R_{A/B} = P_A^{(\mathrm{log})} - P_B^{(\mathrm{log})}

- ``log10`` input gives :math:`\log_{10}(P_A / P_B)`.
- ``log_ratio`` input gives :math:`\log_{10}((P_A / B_A) / (P_B / B_B))`.
- ``db`` input is ten times that baseline-relative difference.

Asymmetry
~~~~~~~~~

For a homologous pair :math:`(L, R)`, raw asymmetry is

.. math::

   A_{L, R} = \frac{P_R - P_L}{P_R + P_L}

Percent input is refused. Logarithmic input subtracts the stored values.

.. math::

   A_{L, R} = P_R^{(\mathrm{log})} - P_L^{(\mathrm{log})}

- ``log10`` input gives :math:`\log_{10}(P_R / P_L)`.
- ``log_ratio`` input gives :math:`\log_{10}((P_R / B_R) / (P_L / B_L))`.
- ``db`` input is ten times that difference, in dB.
- **Missing values**: a zero denominator or a zero sum returns NaN.
- **Reference**: right-minus-left asymmetry follows the frontal-asymmetry
  convention of Davidson, Chapman, Chapman, and Henriques (1990).

Output Units
~~~~~~~~~~~~

The output unit names the scale of that result. Log-ratio asymmetry is reported
in the unit of its input, from the table below.

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
