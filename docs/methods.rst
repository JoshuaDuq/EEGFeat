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

Peak frequency identifies the frequency of maximum spectral power within a band, refined via parabolic interpolation through the discrete maximum and its two adjacent neighbors:

.. math::

   \delta = \frac{1}{2} \frac{P(f_{k-1}) - P(f_{k+1})}{P(f_{k-1}) - 2 P(f_k) + P(f_{k+1})}

.. math::

   f_{\text{peak}} = f_k + \delta \cdot \frac{f_{k+1} - f_{k-1}}{2}

where :math:`k` is the discrete argmax index. Parabolic interpolation requires at least three frequency bins in the band. If the discrete maximum falls on the first or last bin of the band, the ``edge_hit`` flag is set, indicating that the true peak may lie outside the evaluated band.

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
