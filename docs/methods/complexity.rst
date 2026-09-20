Complexity and Microstate Methods
=================================

Measures of signal regularity, and the segmentation of the topographic map
sequence into a small set of recurring states.

Signatures for these functions are in :doc:`/api/complexity`.

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

The sample-entropy estimator is the self-match-excluding statistic introduced
by `Joshua S. Richman and J. Randall Moorman (2000)
<https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__. The package retains the
strict Chebyshev matching rule and makes undefined pair counts visible as NaN
or infinity instead of replacing them with an arbitrary finite value.
``multiscale_entropy`` follows the coarse-graining construction of `Madalena
Costa, Ary L. Goldberger, and C.-K. Peng (2002)
<https://doi.org/10.1103/PhysRevLett.89.068102>`__. Recomputing the tolerance
from each coarse-grained standard deviation is exposed as a different estimator
because it is not the classical fixed-tolerance MSE definition.

The core sample-entropy computation is:

.. code-block:: python

   templates = sliding_window_view(signal, order + 1)
   templates = templates[np.isfinite(templates).all(axis=1)]
   finite_signal = signal[np.isfinite(signal)]
   tolerance = r * np.std(finite_signal)
   if not np.isfinite(tolerance) or tolerance <= 0.0:
       tolerance = max(np.finfo(float).eps, r * np.nanstd(finite_signal))
   short_matches = 0
   long_matches = 0
   for i, j in unordered_pairs(templates):
       short = np.max(np.abs(templates[i, :order] - templates[j, :order])) < tolerance
       if short:
           short_matches += 1
           long_matches += (
               np.max(np.abs(templates[i] - templates[j])) < tolerance
           )
   sampen = -np.log(long_matches / short_matches)

The implementation counts each unordered pair once, discards incomplete
templates, and returns NaN or infinity before the final line when the respective
denominator or numerator is zero. For multiscale entropy, the coarse series is
formed by averaging complete non-overlapping blocks and any trailing incomplete
block is discarded:

.. code-block:: python

   n_blocks = signal.size // scale
   blocks = signal[:n_blocks * scale].reshape(n_blocks, scale)
   coarse = np.where(np.isfinite(blocks).all(axis=1), blocks.mean(axis=1), np.nan)
   source = signal if tolerance_mode == "original_sd" else coarse
   tolerance = r * np.std(source[np.isfinite(source)])

If the finite standard deviation is non-finite or zero, the implementation uses
the smallest positive floating-point tolerance rather than manufacturing a
finite entropy from exact self-matches.

Higuchi Fractal Dimension
-------------------------

``higuchi_fractal_dimension`` retraces the signal at integer strides
:math:`k`, estimates the mean curve length :math:`L(k)` for each stride, and
fits the scaling relation :math:`L(k) \propto k^{-D}`. The reported dimension
:math:`D` is therefore the slope of :math:`\log L(k)` against
:math:`-\log k`: values near one are characteristic of smooth curves, whereas
larger values indicate structure that persists across finer scales. The
estimator is linear in the number of samples for a fixed ``k_max`` and is not a
substitute for sample entropy: it measures geometric scale dependence rather
than template-match predictability.

This is the estimator introduced by `T. Higuchi (1988)
<https://doi.org/10.1016/0167-2789(88)90081-4>`__. The package reports NaN
when a window contains non-finite samples or fewer samples than the largest
requested stride, so an unsuccessful estimate remains distinguishable from a
valid low-dimensional signal.

For :math:`N` samples and stride :math:`k`, the exact length normalization is:

.. math::

   L_m(k) = \frac{\sum_q |x_{m+(q+1)k} - x_{m+qk}|}{q_{\max} k^2}(N-1),
   \qquad L(k) = \frac{1}{k}\sum_{m=0}^{k-1} L_m(k),

where :math:`q_{\max}=\lfloor(N-m-1)/k\rfloor`. The reported dimension is the
slope of ``log(L(k))`` against ``-log(k)``. In code-like form:

.. code-block:: python

   for k in range(1, k_max + 1):
       lengths = [np.abs(np.diff(signal[m::k])).sum()
                  * (n_samples - 1) / (steps(m, k) * k * k)
                  for m in range(k)]
       L[k - 1] = np.nanmean(lengths)
   dimension = np.polyfit(-np.log(k_values), np.log(L), 1)[0]

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

The corresponding implementation is:

.. code-block:: python

   demeaned = epoch - np.mean(epoch, axis=0, keepdims=True)
   gfp = np.std(demeaned, axis=0)
   maps = normalize_rows(demeaned.T)
   model = KMeans(n_clusters=K, n_init=20, random_state=random_state)
   model.fit(maps[gfp_peaks])
   templates = normalize_rows(model.cluster_centers_)
   states = np.argmax(np.abs(templates @ maps.T), axis=0)

The row normalization subtracts the channel mean, divides by the Euclidean
norm, and flips the sign so the largest-magnitude channel is positive. The
fitted templates therefore use ordinary k-means on sign-normalized maps; they
are not the polarity-invariant modified k-means objective used by Pycrostates.

**Template fitting pools across trials; the measures do not.** ``fit_on`` names which trials may contribute topographies, as a cross-validation fold requires. The default uses every trial, which is right for description and leaks for prediction. Assignment and every measure derived from it are per epoch, so the returned feature tables have one row per epoch.

Cluster indices are arbitrary, so unmatched templates are labelled ``state1``
onward. A-D labels require one-to-one topographic matching to an identified
reference set and are not assigned automatically. The segmentation exposes its
templates and global explained variance. Feature identities include the fitted
templates, channel order, contributing rows, and segmentation settings, so
independently fitted ``state1`` columns cannot silently represent the same
feature. Segmentation rejects non-finite or spatially constant maps rather
than assigning them to an arbitrary state. This implementation uses ordinary
k-means on sign-normalized maps, not the polarity-invariant modified k-means
objective used by Pycrostates; these estimators are not interchangeable.
From the resulting sequence :math:`s(t)`,
four temporal statistics are derived:

- **coverage**: Fractional occupancy time: :math:`\frac{1}{T} \sum_t \mathbb{I}[s(t) = k]` (compositional, sums to 1 across states).
- **duration**: Mean continuous dwell time per visit in milliseconds (evaluates to NaN if state :math:`k` was never entered).
- **occurrence**: Number of distinct visits per second (evaluates to :math:`0.0` if never entered).
- **transitions**: Directional transition probability between successive segments:

.. math::

   T_{i \to j} = \frac{N_{i \to j}}{\sum_{m \ne i} N_{i \to m}} \quad (i \ne j)

Self-transitions (:math:`i = j`) are omitted from segment-based transition matrices.

The global explained variance exposed by the segmentation is the GFP-weighted
fit of the assigned template:

.. code-block:: python

   gfp = np.nanstd(epoch - np.nanmean(epoch, axis=0, keepdims=True), axis=0)
   correlations = np.abs(normalized_maps @ templates.T)
   assigned = correlations[np.arange(n_times), states]
   gev = np.sum(gfp ** 2 * assigned ** 2) / np.sum(gfp ** 2)

Coverage, duration, occurrence, and transitions are then computed from the
smoothed state sequence using ``mean(state == k)``, mean run length divided by
sampling frequency, run count divided by window seconds, and row-normalized
successive-run counts, respectively.

The microstate model descends from the quasi-stable scalp-map analysis of
`Dietrich Lehmann, H. Ozaki, and I. Pal (1987)
<https://doi.org/10.1016/0013-4694(87)90025-3>`__ and the GFP-peak clustering
and back-fitting formulation of `Roberto D. Pascual-Marqui, Christoph M. Michel,
and Dietrich Lehmann (1995) <https://doi.org/10.1109/10.391164>`__. The review
by `Christoph M. Michel and Thomas Koenig (2018)
<https://doi.org/10.1016/j.neuroimage.2017.11.062>`__ explains why GFP peaks,
topographic correlation, polarity handling, and temporal descriptors are
separate methodological choices.

This package intentionally differs from the classical polarity-invariant
modified k-means objective: it sign-normalizes the selected maps and then uses
ordinary k-means. Consequently, ``state1``--``stateK`` are estimator-local
cluster labels, not automatically the canonical A--D maps, and the temporal
features should not be compared across independently fitted segmentations
without a topographic matching step.

References
----------

* Higuchi, T. (1988). *Approach to an irregular time series on the basis of
  the fractal theory*. Physica D: Nonlinear Phenomena, 31(2), 277--283.
  `doi:10.1016/0167-2789(88)90081-4
  <https://doi.org/10.1016/0167-2789(88)90081-4>`__.
* Richman, J. S., & Moorman, J. R. (2000). *Physiological time-series analysis
  using approximate entropy and sample entropy*. American Journal of
  Physiology--Heart and Circulatory Physiology, 278(6), H2039--H2049.
  `doi:10.1152/ajpheart.2000.278.6.H2039
  <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__.
* Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). *Multiscale entropy
  analysis of complex physiologic time series*. Physical Review Letters, 89(6),
  068102. `doi:10.1103/PhysRevLett.89.068102
  <https://doi.org/10.1103/PhysRevLett.89.068102>`__.
* Lehmann, D., Ozaki, H., & Pal, I. (1987). *EEG alpha map series: Brain
  micro-states by space-oriented adaptive segmentation*. Electroencephalography
  and Clinical Neurophysiology, 67(3), 271--288.
  `doi:10.1016/0013-4694(87)90025-3
  <https://doi.org/10.1016/0013-4694(87)90025-3>`__.
* Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). *Segmentation of
  brain electrical activity into microstates: Model estimation and
  validation*. IEEE Transactions on Biomedical Engineering, 42(7), 658--665.
  `doi:10.1109/10.391164 <https://doi.org/10.1109/10.391164>`__.
* Michel, C. M., & Koenig, T. (2018). *EEG microstates as a tool for studying
  the temporal dynamics of whole-brain neuronal networks: A review*.
  NeuroImage, 180(B), 577--593.
  `doi:10.1016/j.neuroimage.2017.11.062
  <https://doi.org/10.1016/j.neuroimage.2017.11.062>`__.

Segmentation requires scikit-learn: ``pip install eegfeat[microstates]``.
