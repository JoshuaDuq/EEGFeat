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

Segmentation requires scikit-learn: ``pip install eegfeat[microstates]``.
