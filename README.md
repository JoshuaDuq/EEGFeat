# eegfeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Typing: Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](docs/)

Labelled spectral, temporal, oscillatory burst, connectivity, complexity, and microstate feature extraction for MNE-Python objects, with group-disjoint predictive modeling and statistical inference (`eegfeat.model`).

`eegfeat` maps precomputed MNE structures (`Spectrum`, `EpochsTFR`, and `Epochs`) to self-describing `FeatureTable` outputs: numeric value matrices paired with column-level `FeatureMeta` records (measure, numerical window bounds, frequency bands or band pairs, channel/ROI or node pair, normalization, units, computation parameters, and stable hashes) and parallel finite-data `coverage` matrices.

---

## Methodological Safeguards

`eegfeat` eliminates common electrophysiological feature extraction and predictive modeling failure modes:

- **Support-restricted wavelets**: Per-frequency temporal support masks ($`5 n_{\text{cycles}} / (2 \pi f)`$) match MNE's Morlet extent and prevent information outside a window from entering its features.
- **Aperiodic-whitened peaks**: Iteratively fitted robust linear $1/f$ baselines and parabolic interpolation remove low-frequency spectral tilt bias.
- **Baseline-calibrated thresholds**: Burst detection and ERDS baselines are calibrated on unperturbed reference windows to avoid stimulus-induced circularity.
- **Strict row semantics**: Cross-trial measures (ITPC, wPLI, AEC) return one row per trial group with explicit labels, preventing single-trial pseudo-replication.
- **Finite-data accounting**: Parallel coverage matrices report numerical finiteness, not artifact rejection. Morlet spectra separately expose the fraction of each requested window with complete wavelet support.
- **Group-disjoint cross-validation**: Subjects and runs never appear in both training and evaluation splits; inner tuning refuses fewer than two training groups.
- **Fold-local preprocessing**: Imputation, scaling, feature selection, ComBat harmonization, and target residualization fit strictly on training folds to prevent target leakage.
- **Subject-level primary metrics**: Continuous predictions are aggregated across subjects in Fisher $z$-space with equal weighting rather than pooled across trials.
- **Unconditioned permutation nulls**: Multi-level label permutations (within-subject, run-wise, circular shift) include all sampled draws without selective filtering.
- **Conformal prediction intervals**: Distribution-free coverage intervals calibrated at trial or subject levels (split, CV+, CQR).

---

## Installation

```bash
pip install eegfeat
```

### Optional Extras

- `pip install "eegfeat[model]"`: `scikit-learn` for predictive modeling, cross-fitting, and statistical evaluation.
- `pip install "eegfeat[importance]"`: `scikit-learn` + `shap` for SHAP and permutation feature importance.
- `pip install "eegfeat[connectivity]"`: `mne-connectivity` for weighted phase lag index (wPLI).
- `pip install "eegfeat[microstates]"`: `scikit-learn` for GFP-peak topography clustering.
- `pip install "eegfeat[knee]"`: `specparam` for spectral knee fitting.
- `pip install "eegfeat[docs]"`: Sphinx, Furo theme, and doc extensions.
- `pip install "eegfeat[dev]"`: `pytest`, `mypy`, `ruff`, and stubs.

---

## Quick Start & Setup

Extracting features begins with standard MNE objects wrapped into `eegfeat` containers:

```python
import mne
import eegfeat as ef

# 1. Load MNE epochs
epochs = mne.read_epochs("sub-01_task-rest_epo.fif", preload=True)

# 2. Define frequency bands and analysis windows
theta = ef.Band("theta", 4.0, 8.0)
alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
gamma = ef.Band("gamma", 30.0, 45.0)

base_win = ef.Window("baseline", -0.5, 0.0)
task_win = ef.Window("stimulus", 0.0, 1.0)

# 3. Wrap MNE structures into eegfeat containers
spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0)
spectra = ef.Spectra.from_spectrum(
    spectrum,
    recording="sub-01_task-test",
    estimator_parameters={"method": "welch", "fmin": 1.0, "fmax": 45.0},
)
raw_sig = ef.Signal.from_epochs(epochs, recording="sub-01_task-test")
alpha_sig = ef.BandSignal.from_epochs(
    epochs, band=alpha, recording="sub-01_task-test"
)
theta_sig = ef.BandSignal.from_epochs(
    epochs, band=theta, recording="sub-01_task-test"
)
gamma_sig = ef.BandSignal.from_epochs(
    epochs, band=gamma, recording="sub-01_task-test"
)
```

---

## Computing Features

### 1. Spectral Power & Ratios

```python
# Absolute or normalized band power across channels and global average
power = ef.integrated_band_power(
    spectra, bands=[theta, alpha, beta], include_global=True
)

# Inter-band power ratio (e.g., theta / beta)
ratio = ef.band_ratio(power, numerator="theta", denominator="beta")

# Hemispheric asymmetry between homologous channel pairs
asym = ef.asymmetry(power, pairs=[("F3", "F4"), ("P3", "P4")])
```

### 2. Spectral Descriptors & Aperiodic Fitting

```python
# 1/f-whitened peak frequency with parabolic refinement & CoG fallback
peak_freq = ef.peak_frequency(spectra, band=alpha, fit_range=(2.0, 40.0))

# Spectral centroid (center of gravity) and spectral bandwidth (dispersion)
centroid = ef.spectral_centroid(spectra, band=alpha)
bandwidth = ef.spectral_bandwidth(spectra, band=alpha)

# Spectral edge frequency (e.g. 95% power boundary) and normalized Shannon entropy
edge = ef.spectral_edge(spectra, band=alpha, percentile=0.95)
entropy = ef.spectral_entropy(spectra, band=alpha)

# Robust linear 1/f background parameters (slope & offset) and residual ratio
ap_fit = ef.aperiodic(spectra, fit_range=(2.0, 40.0))
ap_ratio = ef.aperiodic_ratio(spectra, fit_range=(2.0, 40.0))
```

### 3. Time-Domain Metrics

```python
# Windowed signal variance and mean amplitude
var = ef.variance([raw_sig], windows=[task_win])
mean_amp = ef.mean_amplitude([raw_sig], windows=[task_win])

# Peak-to-peak amplitude and area under the curve (trapezoidal, skipping gaps)
p2p = ef.peak_to_peak([raw_sig], windows=[task_win])
auc = ef.area_under_curve([raw_sig], windows=[task_win])

# Signed extremum amplitude and peak latency (seconds from epoch onset)
peak_amp = ef.peak_amplitude([raw_sig], windows=[task_win], polarity="positive")
peak_lat = ef.peak_latency([raw_sig], windows=[task_win], polarity="positive")
```

### 4. Oscillatory Bursts

```python
# Baseline-calibrated suprathreshold envelope excursion metrics
burst_cnt = ef.burst_count([alpha_sig], baseline=base_win, windows=[task_win], threshold=0.75)
burst_rt = ef.burst_rate([alpha_sig], baseline=base_win, windows=[task_win], threshold=0.75)
burst_dur = ef.burst_duration([alpha_sig], baseline=base_win, windows=[task_win], threshold=0.75)
burst_amp = ef.burst_amplitude([alpha_sig], baseline=base_win, windows=[task_win], threshold=0.75)
frac_above = ef.fraction_above_threshold(
    [alpha_sig], baseline=base_win, windows=[task_win], threshold=0.75
)
```

### 5. ERDS Dynamics

```python
# Mean ERD/ERS (% or dB relative to baseline) and linear rate of change (slope)
erds_m = ef.erds_mean([alpha_sig], baseline=base_win, windows=[task_win], normalize="percent")
erds_s = ef.erds_slope([alpha_sig], baseline=base_win, windows=[task_win])

# Desynchronization (ERD) and synchronization (ERS) peak magnitudes & durations
erd_m = ef.erd_magnitude([alpha_sig], baseline=base_win, windows=[task_win])
erd_d = ef.erd_duration([alpha_sig], baseline=base_win, windows=[task_win])
ers_m = ef.ers_magnitude([alpha_sig], baseline=base_win, windows=[task_win])
ers_d = ef.ers_duration([alpha_sig], baseline=base_win, windows=[task_win])

# Event-related latencies: onset (departure from baseline), peak, and rebound
onset = ef.erds_onset_latency([alpha_sig], baseline=base_win, windows=[task_win])
peak = ef.erds_peak_latency([alpha_sig], baseline=base_win, windows=[task_win])
rebound = ef.erds_rebound_latency([alpha_sig], baseline=base_win, windows=[task_win])
```

### 6. Phase, Connectivity & Graph Metrics

```python
# Inter-trial phase coherence across trial groups (one row per group)
coh = ef.itpc([theta_sig], windows=[task_win])

# Phase-amplitude coupling (theta phase modulating gamma amplitude envelope)
phase_amp = ef.pac(theta_sig, gamma_sig, windows=[task_win])

# Pairwise amplitude envelope correlation (AEC) and weighted phase lag index (wPLI)
conn_aec = ef.envelope_correlation([alpha_sig], windows=[task_win])
conn_wpli = ef.wpli(raw_sig, bands=[alpha], windows=[task_win])

# Graph topology derived from pairwise connectivity matrices
efficiency = ef.global_efficiency(conn_aec)
clustering = ef.clustering_coefficient(conn_aec, threshold=0.2)
```

### 7. Complexity & Entropy

```python
# Chebyshev sample entropy and coarse-grained multiscale entropy
samp_ent = ef.sample_entropy([raw_sig], windows=[task_win], m=2, r=0.2)
mse = ef.multiscale_entropy([raw_sig], windows=[task_win], scales=range(1, 6))
```

### 8. Microstate Segmentation

```python
# Cluster GFP peak topographies into discrete state templates
seg = ef.segment(raw_sig, n_states=4, min_duration_ms=20.0)

# Windowed microstate statistics: fractional coverage, mean duration, occurrence, and transitions
m_cov = ef.microstate_coverage(seg, windows=[task_win])
m_dur = ef.microstate_duration(seg, windows=[task_win])
m_occ = ef.microstate_occurrence(seg, windows=[task_win])
m_trans = ef.microstate_transitions(seg, windows=[task_win])
```

---

## Table Operations & BIDS-style I/O

```python
from eegfeat.io import read_table, write_table

# Concatenate tables horizontally (validates row semantics)
features = ef.concat([power, peak_freq, var, burst_rt])

# Query columns by structured metadata fields
alpha_cols = features.select(band=alpha, space_kind="channel")

# Export to pandas DataFrame with canonical structured column names
df = features.to_dataframe()

# Write BIDS-style values TSV, finite-data coverage TSV, and JSON sidecar
paths = write_table(features, "sub-01_features.tsv", rows=epochs.metadata)

# Restore FeatureTable losslessly with metadata, flags, and row labels
restored = read_table("sub-01_features.tsv")
```

---

## Predictive Modeling (`eegfeat.model`)

`eegfeat.model` is a scikit-learn subpackage for fitting, cross-validating, and evaluating predictive models on `FeatureTable`s with strict leakage controls.

### 1. Design Matrices from Feature Tables

Filter features by structured metadata queries, extract target variables, and isolate grouping identifiers:

```python
import eegfeat.model as efm

# Build (X, y, groups) from FeatureTable and trial metadata
design = efm.build_design(
    table=features,
    metadata=epochs.metadata,
    target="reaction_time",
    selection=efm.Selection(band=("theta", "alpha"), space_kind=("channel",)),
    group_by="subject_id",
)
X, y, groups = design.X, design.y, design.groups
```

### 2. Group-Disjoint Cross-Fitting & Inner Tuning

Fit regression or classification pipelines across leave-one-subject-out (LOSO) or within-subject folds with fold-local inner tuning:

```python
import numpy as np

# Create outer LOSO folds and declare inner cross-validation structure
folds = efm.loso_folds(groups)
inner = efm.InnerSplit(grouping="subject", n_splits=5)

# Standard estimator pipelines and parameter grids (ridge, elasticnet, rf, svm, logistic)
pipe = efm.ridge_pipeline(include_scaling=True)
grid = efm.ridge_grid()

# Outer cross-validation loop with fold-local inner hyperparameter tuning
results = efm.cross_fit_regression(
    folds=folds,
    X=X,
    y=y,
    groups=groups,
    pipeline=pipe,
    grid=grid,
    inner=inner,
)

y_pred = np.concatenate([r.y_pred for r in results])
y_true = np.concatenate([r.y_true for r in results])
eval_groups = np.concatenate([groups[r.rows] for r in results])
```

### 3. Subject-Level Metrics & Hypothesis Testing

Evaluate performance with equal subject weighting via Fisher $z$-transformation:

```python
# Primary subject-level metrics and per-subject correlation scores
metrics, per_subject = efm.regression_metrics(y_true, y_pred, groups=eval_groups)
print(f"Subject-level r: {metrics['subject_level_r']:.3f}")

# Bootstrap confidence intervals and non-parametric sign-flip test
subj_r = np.array([r for _, r in per_subject])
ci_low, ci_high = efm.bootstrap_mean_ci(subj_r, iterations=10_000, seed=42)
p_signflip = efm.paired_signflip_p_value(subj_r, iterations=10_000, seed=42)
```

### 4. Permutation Null Distributions

Test exchangeability against multi-level null schemes without selective filtering:

```python
# Hierarchical label permutation null (within-subject, run-wise, or circular shift)
null = efm.permutation_test(
    folds=folds,
    X=X,
    y=y,
    groups=groups,
    runs=None,
    pipeline=pipe,
    grid=grid,
    observed=metrics["subject_level_r"],
    inner=inner,
    config=efm.NullConfig(n_permutations=100, scheme="within_subject"),
    seed=42,
)
print(f"Permutation p-value: {null.p_value:.4f}")
```

### 5. Conformal Prediction Intervals

Construct distribution-free prediction intervals with finite-sample coverage guarantees:

```python
# Conformal prediction intervals (split, cv_plus, or quantile/CQR)
intervals = efm.prediction_intervals(
    model=pipe,
    X_train=X,
    y_train=y,
    X_test=X[:10],
    groups=groups,
    alpha=0.10,
    method="cv_plus",
)
print(f"Coverage: {intervals.coverage:.1%}, calibration unit: {intervals.calibration_unit}")
```

### 6. Feature Importance & Metadata Aggregation

Compute test-set permutation or SHAP importance and aggregate scores by anatomical and spectral attributes:

```python
# Fold-aggregated permutation importance on held-out test sets
importance = efm.permutation_importance_over_folds(
    folds=folds,
    X=X,
    y=y,
    groups=groups,
    pipeline=pipe,
    grid=grid,
    inner=inner,
)

# Sum feature importance by metadata field (band, space, or measure)
band_importance = efm.aggregate_by(importance, design.meta, field="band")
```

---

## Command Line Runner

For batch extraction across whole datasets without writing custom loops, use the declarative recipe runner:

```bash
# 1. Generate a commented recipe template
eegfeat init recipe.toml

# 2. Validate recipe syntax and test execution on the first recording
eegfeat check recipe.toml

# 3. Process the complete dataset with parallel MNE jobs
eegfeat run recipe.toml --n-jobs 4
```

---

## Verification & Documentation

The test suite checks continuous values against analytic derivations and selected independently executed reference implementations. Formula-derived fixtures are regression evidence, not by themselves evidence of scientific validity.

```bash
# Run unit & regression test suite
pytest

# Static type verification
mypy src
```

Comprehensive documentation is available under `docs/` and built via Sphinx:
- **[Quick Start](docs/quickstart.rst)**: Walkthroughs and end-to-end extraction recipes.
- **[Methods](docs/methods.rst)**: Mathematical formulations and algorithm specifications.
- **[Command Line Runner](docs/runner.rst)**: Recipe syntax, batch configuration, and output schemas.
- **[API Reference](docs/api.rst)**: Complete function signatures and class specifications.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
