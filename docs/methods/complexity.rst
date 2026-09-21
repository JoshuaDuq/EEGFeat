Complexity and Microstate Methods
=================================

Regularity of a one-dimensional series, and segmentation of the scalp map into
recurring states.

Signatures are in :doc:`/api/complexity`.

Sample and Multiscale Entropy
-----------------------------

Sample entropy is the negative log probability that two templates matching over
``order`` samples (:math:`m`) still match over :math:`m + 1` (Richman and
Moorman, 2000).

.. math::

   H(x, m, r) = -\log \frac{C(m + 1, r)}{C(m, r)}

Two templates match when their Chebyshev distance is strictly below
:math:`r \sigma_x`. Each unordered pair is counted once. No match at length
:math:`m` returns NaN. Matches at length :math:`m` and none at length
:math:`m + 1` return infinity. Zero is the value for a perfectly regular
series, so those two outcomes are not replaced by zero.

``multiscale_entropy`` repeats the measure after coarse-graining by
non-overlapping block means, one column per scale (Costa, Goldberger, and Peng,
2002). ``tolerance_mode="original_sd"`` keeps :math:`r \sigma_x` from the
original series at every scale. ``tolerance_mode="scale_sd"`` recomputes
:math:`\sigma` after coarse-graining and is stored as a separate estimator.
A block that contains a non-finite sample stays non-finite. A trailing
incomplete block is dropped. An embedding template that crosses a gap is
excluded. Missing samples are not deleted in a way that would make new
neighbours.

If :math:`r` times the standard deviation of the finite samples is non-finite
or non-positive, the tolerance falls back to
:math:`\max(\epsilon, r \cdot \operatorname{nanstd})` of those samples, where
:math:`\epsilon` is the smallest positive float. The log is not evaluated when
the match count in the denominator or the numerator is zero.

Cost grows with the square of the window length.

.. code-block:: python

   n_blocks = signal.size // scale
   blocks = signal[:n_blocks * scale].reshape(n_blocks, scale)
   coarse = np.where(np.isfinite(blocks).all(axis=1), blocks.mean(axis=1), np.nan)
   source = signal if tolerance_mode == "original_sd" else coarse
   tolerance = r * np.std(source[np.isfinite(source)])

Higuchi Fractal Dimension
-------------------------

``higuchi_fractal_dimension`` retraces the series at integer strides
:math:`k`, estimates the mean curve length :math:`L(k)`, and fits
:math:`L(k) \propto k^{-D}` (Higuchi, 1988). :math:`D` is the slope of
:math:`\log L(k)` against :math:`-\log k`. Values near 1 are smooth curves.
Larger values keep structure at finer scales. For a fixed ``k_max`` the cost
is linear in the number of samples. The quantity is geometric scale dependence.
It is a different estimator from sample entropy.

For :math:`N` samples and stride :math:`k`,

.. math::

   L_m(k) = \frac{\sum_q |x_{m+(q+1)k} - x_{m+qk}|}{q_{\max} k^2}(N-1),
   \qquad L(k) = \frac{1}{k}\sum_{m=0}^{k-1} L_m(k),

with :math:`q_{\max} = \lfloor (N - m - 1) / k \rfloor`. The slope is
``np.polyfit(-log(k), log(L), 1)``. A window with any non-finite sample, or
with fewer samples than ``k_max``, returns NaN.

Microstates
-----------

Templates are clustered from the scalp maps at local maxima of the global field
power (Lehmann, Ozaki, and Pal, 1987; Pascual-Marqui, Michel, and Lehmann,
1995).

.. math::

   \text{GFP}(t) = \sqrt{\frac{1}{N_{\text{channels}}} \sum_{c=1}^{N_{\text{channels}}} \left( V_c(t) - \bar{V}(t) \right)^2 }

:math:`\bar{V}(t)` is the average reference at that sample. Each peak map is
oriented so that its largest-magnitude channel is positive. Clustering is the
polarity-invariant modified :math:`k`-means of Pascual-Marqui et al. (1995).
Assignment uses absolute spatial correlation.

.. math::

   s(t) = \arg\max_k \frac{|V(t)^T \mu_k|}{\|V(t)\| \|\mu_k\|}

Each template is the principal eigenvector of its members' scatter matrix, so
negating a member does not change the template. scikit-learn :math:`k`-means
supplies only the k-means++ start (``n_init=20``). Ordinary :math:`k`-means on
sign-normalized maps is a different procedure. Orientation by the strongest
channel jumps when two extrema have similar magnitude, and noise can then split
one state's maps across clusters.

Segments shorter than ``min_duration_ms`` are absorbed into the longer
neighbouring state, or split between the two neighbours on a tie.

.. code-block:: python

   demeaned = epoch - np.mean(epoch, axis=0, keepdims=True)
   gfp = np.std(demeaned, axis=0)
   maps = normalize_rows(demeaned.T)
   seeds = KMeans(n_clusters=K, n_init=20, random_state=random_state).fit(maps[gfp_peaks])
   templates = unit_rows(seeds.cluster_centers_)
   while labels change:
       labels = np.argmax(np.abs(maps[gfp_peaks] @ templates.T), axis=1)
       for k in range(K):
           members = maps[gfp_peaks][labels == k]
           templates[k] = principal_eigenvector(members.T @ members)
   templates = normalize_rows(templates)
   states = np.argmax(np.abs(templates @ maps.T), axis=0)

Row normalization subtracts the channel mean, divides by the Euclidean norm,
and flips the sign so the largest-magnitude channel is positive. The sign flip
is a reporting convention. The clustering objective does not use it.

``fit_on`` names the trials whose maps may enter the fit, as a
cross-validation fold would require. The default uses every trial. That pool
includes the trial later scored if these features are used for prediction.
Assignment and the measures below are computed per epoch, so each returned
table has one row per epoch.

Cluster indices have no anatomical meaning. Templates that are not matched to
a reference are labelled ``state1`` onward. Labels A–D require a one-to-one
match to an identified reference set and are not assigned here. Temporal
features from two fits are comparable after that topographic match, and not
before. The segmentation returns its templates and the global explained
variance. Column identity includes the templates, the channel order, the rows
that contributed, and the segmentation settings, so ``state1`` from two fits
is two features. Non-finite maps and spatially constant maps are rejected.

The objective is the modified :math:`k`-means that Pycrostates fits.
Initialisation, GFP-peak selection, and short-segment smoothing are the choices
above, and they are stored with the templates. Michel and Koenig (2018) review
why GFP peaks, topographic correlation, polarity, and the temporal summaries
are separate choices.

From the sequence :math:`s(t)`,

- **coverage** is occupancy, :math:`T^{-1} \sum_t \mathbb{I}[s(t) = k]`. Across
  states the values sum to 1.
- **duration** is the mean dwell of a visit, in milliseconds. A state that is
  never entered is NaN.
- **occurrence** is the number of visits per second. A state that is never
  entered is 0.
- **transitions** are probabilities between successive segments,

.. math::

   T_{i \to j} = \frac{N_{i \to j}}{\sum_{m \ne i} N_{i \to m}} \quad (i \ne j)

Self-transitions are omitted.

Global explained variance is the GFP-weighted squared correlation of the
assigned template.

.. code-block:: python

   gfp = np.nanstd(epoch - np.nanmean(epoch, axis=0, keepdims=True), axis=0)
   correlations = np.abs(normalized_maps @ templates.T)
   assigned = correlations[np.arange(n_times), states]
   gev = np.sum(gfp ** 2 * assigned ** 2) / np.sum(gfp ** 2)

Coverage, duration, occurrence, and transitions are then ``mean(state == k)``,
mean run length divided by the sampling rate, run count divided by the window
length in seconds, and row-normalized counts of successive runs.

Segmentation requires scikit-learn (``pip install eegfeat[microstates]``).

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
