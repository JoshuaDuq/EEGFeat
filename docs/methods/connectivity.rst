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

The ITPC expression is the length of the across-trial mean unit-phase vector,
the phase-locking quantity used by `Catherine Tallon-Baudry, Olivier Bertrand,
Claude Delpuech, and Jacques Pernier (1996)
<https://doi.org/10.1523/JNEUROSCI.16-13-04240.1996>`__ and documented in the
single-trial EEG framework of `Arnaud Delorme and Scott Makeig (2004)
<https://doi.org/10.1016/j.jneumeth.2003.10.009>`__. PPC is not a second name
for ITPC: it is the bias-free pairwise estimator introduced by `Martin Vinck,
Marijn van Wingerden, Thilo Womelsdorf, Pascal Fries, and Cyriel M. A.
Pennartz (2010) <https://doi.org/10.1016/j.neuroimage.2010.01.073>`__.

The implementation uses unit complex phase vectors and performs the reductions
in this order:

.. code-block:: python

   unit_phase = np.where(np.isfinite(phase), np.exp(1j * phase), np.nan)
   count = np.sum(np.isfinite(unit_phase), axis=0)
   itpc_per_time = np.abs(np.nansum(unit_phase, axis=0) / count)
   itpc = np.nanmean(itpc_per_time, axis=-1)
   resultant_squared = np.abs(np.nansum(unit_phase, axis=0)) ** 2
   ppc_per_time = (resultant_squared - count) / (count * (count - 1))
   ppc = np.nanmean(ppc_per_time, axis=-1)

The package withholds time points with fewer than ``min_valid_trials`` finite
phases. The PPC expression above is algebraically identical to the pairwise
cosine form, but makes the finite-sample correction explicit.

Phase-Amplitude Coupling
------------------------

Mean vector length, the amplitude-weighted resultant of the slow band's phase:

.. math::

   \mathrm{MVL} = \frac{\left| \sum_t A(t) e^{i \phi(t)} \right|}{\sum_t A(t)}

Normalizing by the summed amplitude makes the value independent of overall power
and so comparable across channels and trials; without it the result scales with
amplitude. In this package, ``normalize=False`` returns the amplitude-weighted
resultant divided by the number of valid samples, not the unscaled complex
resultant.

Computed within each trial, so it carries no cross-trial leakage and has one row
per epoch. **No surrogate correction is applied.** A raw coupling value is biased
upward by amplitude and phase autocorrelation and should be evaluated against an
empirical null distribution (preferring within-trial circular time shifts over
trial-shuffling to preserve single-trial spectral structure).

The mean-vector-length construction follows `Ryan T. Canolty, E. Edwards, Sarang
S. Dalal, Maryam Soltani, S. S. Nagarajan, H. E. Kirsch, Mitchel S. Berger,
N. M. Barbaro, and Robert T. Knight (2006)
<https://doi.org/10.1126/science.1128115>`__. The methodological comparison by
`N. Tort, R. Komorowski, H. Eichenbaum, and N. Kopell (2010)
<https://doi.org/10.1152/jn.00106.2010>`__ motivates treating this raw MVL as
an effect-size estimator rather than as a significance test.

The exact implementation is:

.. code-block:: python

   unit_phase = np.exp(1j * phase)
   valid = np.isfinite(amplitude) & np.isfinite(unit_phase)
   resultant = np.abs(np.sum(np.where(valid, amplitude * unit_phase, 0.0), axis=-1))
   if normalize:
       denominator = np.sum(np.where(valid, amplitude, 0.0), axis=-1)
       mvl = np.where(denominator > 1e-20, resultant / denominator, np.nan)
   else:
       count = np.sum(valid, axis=-1)
       mvl = np.where(count > 0, resultant / count, np.nan)


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

Envelope correlation is the amplitude-correlation approach used in MEG
functional-connectivity work by `Matthew J. Brookes, Joanne R. Hale, Johanna M.
Zumer, Claire M. Stevenson, Susan T. Francis, Gareth R. Barnes, Julia P. Owen,
Peter G. Morris, and Srikantan S. Nagarajan (2011)
<https://doi.org/10.1016/j.neuroimage.2011.02.054>`__. When orthogonalization is
requested, its leakage-correction provenance is `G. L. Colclough, M. J.
Brookes, S. M. Smith, and M. W. Woolrich (2015)
<https://doi.org/10.1016/j.neuroimage.2015.03.071>`__. The remaining spectral
estimators and their exact parameter names are delegated to the official
`MNE-Connectivity spectral-connectivity documentation
<https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html>`__.

The phase-locking value was introduced by `Jean-Philippe Lachaux, Eugenio
Rodriguez, Jacques Martinerie, and Francisco J. Varela (1999)
<https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C>`__.
The imaginary-coherency estimator follows `Guido Nolte, Ou Bai, Lewis Wheaton,
Zoltan Mari, Sherry Vorbach, and Mark Hallett (2004)
<https://doi.org/10.1016/j.clinph.2004.04.029>`__. The corrected imaginary
phase-locking estimator is documented by `Ernesto Pereda, Ricardo Bruña, and
Fernando Maestú (2018) <https://doi.org/10.1088/1741-2552/aacfe4>`__. These
citations identify the scientific estimators; the exact implementation and
parameter behavior remain those of MNE-Connectivity.

For analytic node signals :math:`z_i(t)`, the raw branch is the Pearson
correlation of :math:`|z_i(t)|` and :math:`|z_j(t)|`. The pairwise
orthogonalized branch follows the asymmetric construction of `Joerg F. Hipp,
David J. Hawellek, Maurizio Corbetta, Markus Siegel, and Andreas K. Engel
(2012) <https://doi.org/10.1038/nn.3101>`__:

.. math::

   a_{i\perp j}(t) = \left|\operatorname{Im}\left(z_i(t)
   \frac{\overline{z_j(t)}}{|z_j(t)|}\right)\right|,
   \qquad r_{i\perp j} = \operatorname{corr}(a_{i\perp j}, |z_j|)

The implementation averages :math:`r_{i\perp j}` with its transpose, optionally
takes the absolute value before that average, and combines trials by Fisher's
transform:

.. code-block:: python

   trial_r = np.corrcoef(np.abs(analytic_trial))
   bounded = np.clip(trial_r, -0.999999, 0.999999)
   envelope_correlation = np.tanh(np.nanmean(np.arctanh(bounded), axis=0))

The orthogonalized branch substitutes the projected envelope in the first line
and symmetrizes the resulting asymmetric matrix.

Spectral estimator formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The delegated estimator is defined from epoch-specific cross-spectral and
power quantities :math:`S_{xy}^{(e)}`, :math:`S_{xx}^{(e)}`, and
:math:`S_{yy}^{(e)}`. Writing :math:`\langle\cdot\rangle_e` for the mean over
valid epochs, the delegated scalar estimators before this package's
unordered-pair post-processing are:

.. math::

   \begin{aligned}
   \mathrm{coh} &= \frac{|\langle S_{xy}\rangle_e|}
      {\sqrt{\langle S_{xx}\rangle_e\langle S_{yy}\rangle_e}} \\
   \mathrm{imcoh} &= \frac{\operatorname{Im}(\langle S_{xy}\rangle_e)}
      {\sqrt{\langle S_{xx}\rangle_e\langle S_{yy}\rangle_e}} \\
   \mathrm{PLV} &= \left|\left\langle S_{xy}/|S_{xy}|\right\rangle_e\right| \\
   \mathrm{ciPLV} &= \frac{|\langle\operatorname{Im}(S_{xy}/|S_{xy}|)\rangle_e|}
      {\sqrt{1-|\langle\operatorname{Re}(S_{xy}/|S_{xy}|)\rangle_e|^2}} \\
   \mathrm{PLI} &= |\langle\operatorname{sign}(\operatorname{Im}S_{xy})\rangle_e| \\
   \mathrm{WPLI} &= \frac{|\langle\operatorname{Im}S_{xy}\rangle_e|}
      {\langle|\operatorname{Im}S_{xy}|\rangle_e}.
   \end{aligned}

PPC is the unbiased estimator of squared PLV given above. ``wpli2_debiased``
is MNE-Connectivity's debiased estimator of squared WPLI. These definitions
follow the official `MNE-Connectivity spectral-connectivity documentation
<https://mne.tools/mne-connectivity/dev/generated/mne_connectivity.spectral_connectivity_epochs.html>`__;
the project does not reimplement their cross-spectral accumulation. MNE returns
signed imaginary coherency, but this package takes its absolute value before
the frequency reduction because the public table contains unordered node pairs:
the sign would reverse when the pair order is swapped.

After estimation, the package performs its own half-open band reduction:

.. code-block:: python

   keep = band.mask(result.freqs)  # fmin <= f < fmax
   dense = result.get_data(output="dense")
   selected = np.abs(dense[..., keep]) if method == "imcoh" else dense[..., keep]
   band_matrix = selected.mean(axis=-1)
   band_matrix = band_matrix + band_matrix.T

The phase-lag interpretation of PLI follows `C. J. Stam, G. Nolte, and A.
Daffertshofer (2007) <https://doi.org/10.1002/hbm.20346>`__. The weighted phase
lag index and its sample-size correction follow `Martin Vinck, Robert Oostenveld,
Marijn van Wingerden, Franscesco Battaglia, and Cyriel M. A. Pennartz (2011)
<https://doi.org/10.1016/j.neuroimage.2011.01.055>`__. ``wpli`` is therefore a
specific estimator, whereas ``spectral_connectivity`` is the package's common
interface to several estimators.

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

CSP is the supervised spatial-filtering method introduced to single-trial EEG
classification by `Herbert Ramoser, J. Müller-Gerking, and Gert Pfurtscheller
(2000) <https://doi.org/10.1109/86.895946>`__. The cross-fitting requirement in
this package is a statistical safeguard around that method: because the filters
are estimated from labels, no held-out epoch may influence the filters used to
transform it.

For class :math:`c`, the package first centers each finite epoch over time and
forms a trace-normalized covariance:

.. math::

   C_c = \frac{1}{|E_c|} \sum_{e \in E_c}
   \frac{(X_e-\bar X_e)(X_e-\bar X_e)^\mathsf{T}}
   {\operatorname{tr}\left((X_e-\bar X_e)(X_e-\bar X_e)^\mathsf{T}\right)}.

The generalized eigenvectors solve :math:`C_0 w = \lambda(C_0+C_1)w`.
Components are selected alternately from the largest and smallest eigenvalues.
For an epoch, the reported feature is the log relative projected variance:

.. code-block:: python

   projected = filters @ epoch
   variance = np.nanvar(projected, axis=-1)
   csp_log_power = np.log(variance / np.sum(variance))

This is the standard CSP log-variance feature convention; see the official
`MNE-CSP documentation
<https://mne.tools/stable/generated/mne.decoding.CSP.html>`__ for a compatible
reference implementation. The package additionally records its trace
normalization, component ordering, and cross-fitting split.

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

The global-efficiency definition is the weighted-network measure introduced by
`Vito Latora and Massimo Marchiori (2001)
<https://doi.org/10.1103/PhysRevLett.87.198701>`__. The local clustering
coefficient is the triangle-density convention of `Duncan J. Watts and Steven
H. Strogatz (1998) <https://doi.org/10.1038/30918>`__. This implementation
adds an explicit thresholding rule for weighted connectivity and treats missing
edges as undefined rather than as zero-weight disconnections.

The reductions are implemented directly as:

.. code-block:: python

   if not np.isfinite(matrix).all():
       return np.nan
   length = np.where(np.abs(matrix) > 0.0, 1.0 / np.abs(matrix), np.inf)
   np.fill_diagonal(length, 0.0)
   distance = floyd_warshall(length)
   upper = np.triu_indices_from(distance, k=1)
   pair_distance = distance[upper]
   global_efficiency = np.mean(np.where(np.isfinite(pair_distance),
                                        1.0 / pair_distance, 0.0))
   adjacency = (np.abs(matrix) > threshold).astype(float)
   np.fill_diagonal(adjacency, 0.0)
   degree = adjacency.sum(axis=1)
   triangles = np.diag(adjacency @ adjacency @ adjacency)
   eligible = degree >= 2
   clustering = np.mean(triangles[eligible]
                        / (degree[eligible] * (degree[eligible] - 1.0)))

The actual implementation averages only upper-triangular node pairs for global
efficiency and returns NaN when no node is eligible for clustering.

References
----------

* Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J. (1996).
  *Stimulus specificity of phase-locked and non-phase-locked 40 Hz visual
  responses in human*. The Journal of Neuroscience, 16(13), 4240--4249.
  `doi:10.1523/JNEUROSCI.16-13-04240.1996
  <https://doi.org/10.1523/JNEUROSCI.16-13-04240.1996>`__.
* Delorme, A., & Makeig, S. (2004). *EEGLAB: An open source toolbox for
  analysis of single-trial EEG dynamics including independent component
  analysis*. Journal of Neuroscience Methods, 134(1), 9--21.
  `doi:10.1016/j.jneumeth.2003.10.009
  <https://doi.org/10.1016/j.jneumeth.2003.10.009>`__.
* Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). *Measuring
  phase synchrony in brain signals*. Human Brain Mapping, 8(4), 194--208.
  `doi:10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C
  <https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C>`__.
* Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).
  *Identifying true brain interaction from EEG data using the imaginary part of
  coherency*. Clinical Neurophysiology, 115(10), 2292--2307.
  `doi:10.1016/j.clinph.2004.04.029
  <https://doi.org/10.1016/j.clinph.2004.04.029>`__.
* Pereda, E., Bruña, R., & Maestú, F. (2018). *Phase locking value revisited:
  Teaching new tricks to an old dog*. Journal of Neural Engineering, 15(5),
  056011. `doi:10.1088/1741-2552/aacfe4
  <https://doi.org/10.1088/1741-2552/aacfe4>`__.
* Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., & Pennartz, C. M.
  A. (2010). *The pairwise phase consistency: A bias-free measure of rhythmic
  neuronal synchronization*. NeuroImage, 51(1), 112--122.
  `doi:10.1016/j.neuroimage.2010.01.073
  <https://doi.org/10.1016/j.neuroimage.2010.01.073>`__.
* Canolty, R. T., Edwards, E., Dalal, S. S., Soltani, M., Nagarajan, S. S.,
  Kirsch, H. E., Berger, M. S., Barbaro, N. M., & Knight, R. T. (2006). *High
  gamma power is phase-locked to theta oscillations in human neocortex*.
  Science, 313(5793), 1626--1628.
  `doi:10.1126/science.1128115 <https://doi.org/10.1126/science.1128115>`__.
* Tort, N., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010). *Measuring
  phase-amplitude coupling between neuronal oscillations of different
  frequencies*. Journal of Neurophysiology, 104(2), 1195--1210.
  `doi:10.1152/jn.00106.2010 <https://doi.org/10.1152/jn.00106.2010>`__.
* Brookes, M. J., Hale, J. R., Zumer, J. M., Stevenson, C. M., Francis, S. T.,
  Barnes, G. R., Owen, J. P., Morris, P. G., & Nagarajan, S. S. (2011).
  *Measuring functional connectivity using MEG: Methodology and comparison
  with fcMRI*. NeuroImage, 56(3), 1082--1104.
  `doi:10.1016/j.neuroimage.2011.02.054
  <https://doi.org/10.1016/j.neuroimage.2011.02.054>`__.
* Colclough, G. L., Brookes, M. J., Smith, S. M., & Woolrich, M. W. (2015). *A
  symmetric multivariate leakage correction for MEG connectomes*. NeuroImage,
  117, 439--448. `doi:10.1016/j.neuroimage.2015.03.071
  <https://doi.org/10.1016/j.neuroimage.2015.03.071>`__.
* Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel, A. K. (2012).
  *Large-scale cortical correlation structure of spontaneous oscillatory
  activity*. Nature Neuroscience, 15(6), 884--890.
  `doi:10.1038/nn.3101 <https://doi.org/10.1038/nn.3101>`__.
* Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). *Phase lag index:
  Assessment of functional connectivity from multi channel EEG and MEG with
  diminished bias from common sources*. Human Brain Mapping, 28, 1178--1193.
  `doi:10.1002/hbm.20346 <https://doi.org/10.1002/hbm.20346>`__.
* Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C.
  M. A. (2011). *An improved index of phase-synchronization for
  electrophysiological data in the presence of volume-conduction, noise and
  sample-size bias*. NeuroImage, 55(4), 1548--1565.
  `doi:10.1016/j.neuroimage.2011.01.055
  <https://doi.org/10.1016/j.neuroimage.2011.01.055>`__.
* Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). *Optimal spatial
  filtering of single trial EEG during imagined hand movement*. IEEE
  Transactions on Rehabilitation Engineering, 8(4), 441--446.
  `doi:10.1109/86.895946 <https://doi.org/10.1109/86.895946>`__.
* Latora, V., & Marchiori, M. (2001). *Efficient behavior of small-world
  networks*. Physical Review Letters, 87(19), 198701.
  `doi:10.1103/PhysRevLett.87.198701
  <https://doi.org/10.1103/PhysRevLett.87.198701>`__.
* Watts, D. J., & Strogatz, S. H. (1998). *Collective dynamics of small-world
  networks*. Nature, 393, 440--442. `doi:10.1038/30918
  <https://doi.org/10.1038/30918>`__.
