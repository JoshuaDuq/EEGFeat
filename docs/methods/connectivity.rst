Phase and Connectivity Methods
==============================

Measures defined on phase relationships, either across trials at one sensor or
between pairs of sensors. Most are estimated across trials rather than within
one epoch, which is why they are written to a separate table — see
:ref:`concepts-row-kinds`.

Signatures for these functions are in :doc:`/api/connectivity`.

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
every pair of nodes. ``spectral_connectivity`` exposes several frequency-domain
estimators, including coherence, PLV, PPC, PLI, and wPLI. ``wpli`` is a shorthand
for the weighted phase lag index, which discounts zero-lag coupling and so is less
vulnerable to volume conduction.

**Spectral connectivity delegates to** ``mne_connectivity.spectral_connectivity_epochs``,
the canonical implementation for cross-spectral estimation. It is an optional dependency:
``pip install eegfeat[connectivity]``. At low trial counts, prefer
``method="wpli2_debiased"`` over ``method="wpli"`` when using
:func:`~eegfeat.spectral_connectivity`.

Every wPLI trial group must contain at least two epochs. This minimum prevents
the degenerate single-epoch estimate; it does not establish that the sample
size is sufficient for reliable estimation. Estimator warnings remain visible.

Nodes are channels, or ROIs when ``groups`` is given, in which case the
channel-level matrix is averaged within each ROI block and a node's own block
excludes the diagonal. These connectivity measures are estimated across trials,
so results have one row per trial group; see :func:`~eegfeat.itpc`.

Common Spatial Patterns
-----------------------

:class:`~eegfeat.CommonSpatialPattern` finds spatial filters that maximize the
variance ratio between two classes. Because the labels determine the filters,
:func:`~eegfeat.csp_features` fits them on each training fold and transforms only
the corresponding held-out rows. This assembled table is descriptive: even reusing
the same folds for a classifier leaks test labels through other folds' CSP fits
into the classifier's training features. ``build_design`` rejects these columns.
For prediction, fit CSP inside each training fold and use that same fit to
transform both training and test rows. Repeat this within inner tuning; for a
scikit-learn workflow, place ``mne.decoding.CSP`` inside the classifier pipeline.
The split signature is recorded in the feature computation metadata. CSP expects
band-restricted input, an even number of components, and exactly two classes.

Graph Measures
--------------

``global_efficiency`` and ``clustering_coefficient`` take a pairwise table and
reduce it to one summary value per band and window, analogous to how :func:`~eegfeat.band_ratio`
summarizes a power table. Both graph metrics are computed directly without
third-party network graph dependencies.

Global efficiency converts nonzero edge weights :math:`w_{ij}` into path distances
:math:`L_{ij} = 1/|w_{ij}|`, treats zero weights as absent edges,
computes all-pairs shortest paths via Floyd-Warshall, and averages inverse distance
across all node pairs. A disconnected pair contributes exactly zero:

.. math::

   E_{\text{global}} = \frac{2}{N (N - 1)} \sum_{i < j} \frac{1}{d_{ij}}

where :math:`N` is the number of network nodes.

A non-finite edge makes either graph summary NaN with zero output coverage.
An unmeasured connection cannot be interpreted as an observed disconnection.

The clustering coefficient **binarizes** the connectivity matrix at user-specified ``threshold`` :math:`\theta` (:math:`A_{ij} = 1` if :math:`|w_{ij}| > \theta`, else :math:`0`), and computes the average local clustering coefficient over nodes with degree :math:`k_i \ge 2`:

.. math::

   \begin{aligned}
   C_i &= \frac{(A^3)_{ii}}{k_i (k_i - 1)} = \frac{2 T_i}{k_i (k_i - 1)} \\[6pt]
   C &= \frac{1}{|\{i : k_i \ge 2\}|} \sum_{i : k_i \ge 2} C_i
   \end{aligned}

where :math:`(A^3)_{ii}` is the diagonal entry of the cubed adjacency matrix (representing twice the number of triangles :math:`T_i` containing node :math:`i`), and :math:`k_i = \sum_j A_{ij}` is the node degree. Nodes with :math:`k_i < 2` are excluded from the average. The result is NaN only when no node has at least two neighbors; eligible nodes with no triangles contribute zero.


