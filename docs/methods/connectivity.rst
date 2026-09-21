Phase and Connectivity Methods
==============================

Phase relations across trials at one sensor, and coupling between sensors.
ITPC, PPC, envelope correlation, spectral connectivity, and wPLI are estimated
on a set of trials, so they are written to a group-row table. See
:ref:`concepts-row-kinds`.

Signatures are in :doc:`/api/connectivity`.

Inter-Trial Phase Coherence
---------------------------

.. math::

   \mathrm{ITPC} = \frac{1}{T} \sum_t
   \left| \frac{1}{N} \sum_n e^{i \phi_n(t)} \right|

The value is 1 when the phase matches on every trial at that latency, and 0
when the phase is uniform across trials. The mean across trials is taken at
each time, and those values are then averaged over time. Averaging over time
first is a different quantity. A phase that drifts within the trial but matches
across trials is 1 in the order above and near 0 if time is averaged first.

Under a uniform-phase null the expected ITPC is about :math:`1/\sqrt{N}`.
Small :math:`N` is biased high, and values from different trial counts are not
comparable. The default requires at least two valid trials. Cells below
``min_valid_trials`` are flagged, and time points with fewer finite phases than
that minimum are omitted from the mean.

Pairwise phase consistency (Vinck et al., 2010) estimates squared population
phase locking. Its expectation is 0 under uniform phase, so values from
different trial counts can be compared.

.. math::

   \mathrm{PPC} = \frac{1}{T} \sum_t \left( \frac{2}{N(N - 1)} \sum_{j < k} \cos(\phi_j(t) - \phi_k(t)) \right)

The implementation uses the equivalent form
:math:`(|\sum_n e^{i\phi_n}|^2 - N) / (N(N - 1))` at each time, then averages
over time. ITPC is the length of the across-trial mean unit-phase vector, as
used by Tallon-Baudry, Bertrand, Delpuech, and Pernier (1996) and described in
Delorme and Makeig (2004).

ITPC and PPC have one row per trial group. The same summary copied onto each
member epoch would repeat one number across rows. :func:`~eegfeat.itpc` and
:func:`~eegfeat.ppc` return ``row_labels``. :func:`~eegfeat.concat` rejects a
join of group rows with per-epoch rows. Pass ``trials`` to compute the measure
inside groups such as experimental condition.

Phase-Amplitude Coupling
------------------------

Mean vector length is the amplitude-weighted resultant of the phase of the
slower band (Canolty et al., 2006).

.. math::

   \mathrm{MVL} = \frac{\left| \sum_t A(t) e^{i \phi(t)} \right|}{\sum_t A(t)}

Division by the summed amplitude removes the scale of the envelope.
``normalize=True`` uses that denominator, and returns NaN when it is at most
:math:`10^{-20}`. ``normalize=False`` divides the modulus of the weighted sum
by the number of finite samples. It returns that real mean, not the unscaled
complex sum.

The measure is computed inside each trial, so the table has one row per epoch.
No surrogate distribution is computed. Autocorrelation in amplitude and in
phase biases the raw value upward. Tort, Komorowski, Eichenbaum, and Kopell
(2010) compare coupling estimators and treat raw mean vector length as an
effect size. A null for this value can be built from circular time shifts
inside the trial, which keep the single-trial spectrum.

Connectivity
------------

``envelope_correlation`` is the Pearson correlation of band envelopes between
every pair of nodes. ``spectral_connectivity`` calls
``mne_connectivity.spectral_connectivity_epochs`` for coherence, imaginary
coherency, the phase-locking value, pairwise phase consistency, the
phase-lag index, and wPLI. ``wpli`` is the weighted phase-lag index, which
down-weights zero-lag coupling and therefore reduces the contribution of volume
conduction (Vinck et al., 2011).

Spectral connectivity requires the ``connectivity`` extra
(``pip install eegfeat[connectivity]``). At low trial counts,
``method="wpli2_debiased"`` applies the sample-size correction in Vinck et al.
(2011). ``method="wpli"`` does not. Every wPLI group needs at least two epochs.
Warnings raised by the MNE estimator are left visible.

Nodes are channels, or ROIs when ``groups`` is given. With ROIs, the
channel-level matrix is averaged inside each ROI block, and a node's own block
excludes the diagonal. These measures have one row per trial group, as
:func:`~eegfeat.itpc` does.

Envelope correlation of analytic amplitudes is the approach used by Brookes et
al. (2011). Optional orthogonalization follows the leakage correction of
Colclough, Brookes, Smith, and Woolrich (2015) and the pairwise projection of
Hipp, Hawellek, Corbetta, Siegel, and Engel (2012). For analytic signals
:math:`z_i(t)`,

.. math::

   a_{i\perp j}(t) = \left|\operatorname{Im}\left(z_i(t)
   \frac{\overline{z_j(t)}}{|z_j(t)|}\right)\right|,
   \qquad r_{i\perp j} = \operatorname{corr}(a_{i\perp j}, |z_j|)

The implementation averages :math:`r_{i\perp j}` with its transpose. It can
take the absolute value before that average. Trials are combined by a Fisher
transform. Correlations are clipped to :math:`[-0.999999, 0.999999]` before
:math:`\operatorname{arctanh}`.

.. code-block:: python

   trial_r = np.corrcoef(np.abs(analytic_trial))
   bounded = np.clip(trial_r, -0.999999, 0.999999)
   envelope_correlation = np.tanh(np.nanmean(np.arctanh(bounded), axis=0))

The orthogonalized branch replaces one envelope with the projected envelope and
then symmetrizes.

The phase-locking value is Lachaux, Rodriguez, Martinerie, and Varela (1999).
Imaginary coherency is Nolte et al. (2004). The corrected imaginary
phase-locking value is Pereda, Bruña, and Maestú (2018). Parameter names and
cross-spectral accumulation are those of
`MNE-Connectivity <https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html>`__.

Spectral estimator formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`S_{xy}^{(e)}` is the epoch cross-spectrum and :math:`\langle\cdot\rangle_e`
is the mean over valid epochs. Before pair symmetrization the delegated
estimators are

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

PPC in this list is the unbiased estimator of squared PLV given in the ITPC
section. ``wpli2_debiased`` is MNE-Connectivity's debiased estimator of squared
WPLI. Cross-spectral accumulation is computed by MNE-Connectivity. MNE returns
a signed imaginary coherency. This package takes the absolute value before the
frequency average, because the published pairs are unordered and the sign would
flip if the node order were swapped.

The band reduction applied afterwards is

.. code-block:: python

   keep = band.mask(result.freqs)  # fmin <= f < fmax
   dense = result.get_data(output="dense")
   selected = np.abs(dense[..., keep]) if method == "imcoh" else dense[..., keep]
   band_matrix = selected.mean(axis=-1)
   band_matrix = band_matrix + band_matrix.T

The phase-lag index is Stam, Nolte, and Daffertshofer (2007). ``wpli`` is the
Vinck et al. (2011) estimator. ``spectral_connectivity`` is the common wrapper
around the estimators above.

Common Spatial Patterns
-----------------------

:class:`~eegfeat.CommonSpatialPattern` finds spatial filters that maximize the
variance ratio between two classes (Ramoser, Müller-Gerking, and Pfurtscheller,
2000). The labels determine the filters. :func:`~eegfeat.csp_features` fits on
each training fold and transforms the held-out rows of that fold. The resulting
table is a description of those held-out rows. Reusing the same folds as a
classifier's outer split still leaks. A fold's filters are estimated with
labels from epochs that fall in the classifier's training set through another
fold. :func:`eegfeat.model.build_design` rejects these columns.

To use CSP as a predictor, fit it inside each training fold and transform the
training rows and the test rows with that fit. Repeat the fit inside inner
tuning. In a scikit-learn pipeline, put ``mne.decoding.CSP`` in the pipeline.
The split that produced a column is stored in its computation metadata. Input
must be band-limited. The number of components must be even. There must be two
classes.

For class :math:`c`, each finite epoch is centered in time and its covariance
is normalized by the trace.

.. math::

   C_c = \frac{1}{|E_c|} \sum_{e \in E_c}
   \frac{(X_e-\bar X_e)(X_e-\bar X_e)^\mathsf{T}}
   {\operatorname{tr}\left((X_e-\bar X_e)(X_e-\bar X_e)^\mathsf{T}\right)}.

The filters solve :math:`C_0 w = \lambda(C_0+C_1)w`. Components are taken
alternately from the largest and the smallest eigenvalues. The feature is the
log of relative projected variance. Trace normalization, component order, and
the cross-fitting split are stored with the column. A compatible reference
implementation is
`mne.decoding.CSP <https://mne.tools/stable/generated/mne.decoding.CSP.html>`__.

.. code-block:: python

   projected = filters @ epoch
   variance = np.nanvar(projected, axis=-1)
   csp_log_power = np.log(variance / np.sum(variance))

Graph Measures
--------------

``global_efficiency`` and ``clustering_coefficient`` reduce a pairwise table to
one value per band and window, in the same way :func:`~eegfeat.band_ratio`
reduces a power table. Both are computed in this package.

Global efficiency (Latora and Marchiori, 2001) turns a nonzero weight
:math:`w_{ij}` into a distance :math:`L_{ij} = 1/|w_{ij}|`, treats a zero
weight as a missing edge, and runs Floyd–Warshall. A disconnected pair
contributes 0.

.. math::

   E_{\text{global}} = \frac{2}{N (N - 1)} \sum_{i < j} \frac{1}{d_{ij}}

:math:`N` is the number of nodes. The sum is over unordered pairs.

A non-finite edge makes either summary NaN, with output coverage 0. A missing
edge is left undefined. It is not entered as a zero-weight disconnection.

The clustering coefficient binarizes at ``threshold`` :math:`\theta`
(:math:`A_{ij} = 1` when :math:`|w_{ij}| > \theta`) and averages the local
clustering of Watts and Strogatz (1998) over nodes with degree
:math:`k_i \ge 2`.

.. math::

   \begin{aligned}
   C_i &= \frac{(A^3)_{ii}}{k_i (k_i - 1)} = \frac{2 T_i}{k_i (k_i - 1)} \\[6pt]
   C &= \frac{1}{|\{i : k_i \ge 2\}|} \sum_{i : k_i \ge 2} C_i
   \end{aligned}

:math:`(A^3)_{ii}` is twice the number of triangles at node :math:`i`, and
:math:`k_i = \sum_j A_{ij}`. Nodes with :math:`k_i < 2` are omitted. The result
is NaN when no node has two neighbours. An eligible node with no triangles
contributes 0.

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
