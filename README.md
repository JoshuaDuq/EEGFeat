# eegfeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Typing: Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](docs/)

Labelled spectral, temporal, oscillatory burst, connectivity, complexity, and microstate feature extraction for MNE-Python objects.

`eegfeat` maps precomputed MNE structures (`Spectrum`, `EpochsTFR`, and `Epochs`) to self-describing `FeatureTable` outputs: numeric value matrices paired with column-level `FeatureMeta` records (measure, frequency band, channel/ROI, window, normalization, unit, and frequency resolution) and parallel `coverage` quality matrices.

---

## Methodological Safeguards

`eegfeat` eliminates common electrophysiological feature extraction failure modes:

- **Support-restricted wavelets**: Per-frequency temporal support masks ($`n_{\text{cycles}} / (2 f)`$) prevent edge artifacts and pre-stimulus leakage into task windows.
- **Aperiodic-whitened peaks**: Iteratively fitted robust linear $1/f$ baselines and parabolic interpolation remove low-frequency spectral tilt bias.
- **Baseline-calibrated thresholds**: Burst detection and ERDS baselines are calibrated on unperturbed reference windows to avoid stimulus-induced circularity.
- **Strict row semantics**: Cross-trial measures (ITPC, wPLI, AEC) return one row per trial group with explicit labels, preventing single-trial pseudo-replication.
- **Coverage accounting**: Parallel coverage matrices track the exact fraction of finite, artifact-free samples contributing to every feature cell.

---

## Installation

```bash
pip install eegfeat
```

### Optional Extras

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
spectra = ef.Spectra.from_spectrum(epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0))
raw_sig = ef.Signal.from_epochs(epochs)
alpha_sig = ef.BandSignal.from_epochs(epochs, band=alpha)
theta_sig = ef.BandSignal.from_epochs(epochs, band=theta)
gamma_sig = ef.BandSignal.from_epochs(epochs, band=gamma)
```

---

## Computing Features

### 1. Spectral Power & Ratios

```python
# Absolute or normalized band power across channels and global average
power = ef.band_power(spectra, bands=[theta, alpha, beta], include_global=True)

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

## Table Operations & BIDS I/O

```python
from eegfeat.io import read_table, write_table

# Concatenate tables horizontally (validates row semantics)
features = ef.concat([power, peak_freq, var, burst_rt])

# Query columns by structured metadata fields
alpha_cols = features.select(band=alpha, space_kind="channel")

# Export to pandas DataFrame with canonical structured column names
df = features.to_dataframe()

# Write BIDS-compliant values TSV, coverage TSV, and JSON sidecar
paths = write_table(features, "sub-01_features.tsv", rows=epochs.metadata)

# Restore FeatureTable losslessly with metadata, flags, and row labels
restored = read_table("sub-01_features.tsv")
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

The test suite validates continuous feature values against analytic derivations and standard reference algorithms to floating-point precision ($`r_{\text{tol}} \le 10^{-6}`$, $`a_{\text{tol}} \le 10^{-9}`$), with discrete edge searches matching bit-identically.

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
