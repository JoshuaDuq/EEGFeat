# Equivalence Against Reference Pipeline

This document records the equivalence verification between `eegfeat` and the reference
feature extraction implementation in `EEG_fMRI_Pipeline` (`eeg_pipeline.analysis.features.spectral`).

## 1. How to Regenerate Fixtures

Fixtures are extracted directly from preprocessed participant epochs. The KINGSTON external
volume must be mounted before running fixture generation.

```bash
# Ensure KINGSTON drive is mounted at /Volumes/KINGSTON
PYTHONPATH=/Users/joduq24/Desktop/EEG_fMRI_Pipeline .venv/bin/python scripts/make_fixtures.py --subject sub-0001
```

Generated fixture files stored in `tests/fixtures/`:
- `manifest.json`: Metadata, bands, windows, and reference git commit.
- `spectra_sub-0001.npz`: Compact Morlet TFR array and raw epoch data for 8 epochs across 6 channels.
- `reference_sub-0001.npz`: Baseline reference outputs for band power, Welch PSD, spectral descriptors, spectral edge, and robust aperiodic fits.

Total fixture size must stay under 50 MB to remain trackable in version control (current size ~20 MB).

## 2. Compared Measures and Tolerances

| Measure | Reference Function | `rtol` | `atol` | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Band Power** | `_compute_frequency_weighted_power` | `1e-6` | `0.0` | Exact trapezoidal frequency integration; slight difference from single-precision float representation. |
| **Spectral Centroid** | `compute_spectral_center` | `1e-6` | `1e-12` | Weighted mean frequency using `gradient_weights`; differences bounded by floating-point arithmetic. |
| **Spectral Bandwidth** | `compute_spectral_bandwidth` | `1e-6` | `1e-12` | Weighted standard deviation using `gradient_weights`. |
| **Spectral Entropy** | `compute_spectral_entropy` | `1e-6` | `1e-12` | Shannon entropy normalized by $\ln(K)$ using `gradient_weights`. |
| **Spectral Edge (95%)** | `compute_spectral_edge` | `0.0` | `1e-9` | Cumulative mass quantile search over discrete frequency bins; indices match exactly. |
| **Aperiodic Slope** | `_robust_aperiodic_fit` | `1e-6` | `1e-9` | Iterative Huber residual rejection fit in log10-log10 space. |
| **Aperiodic Offset** | `_robust_aperiodic_fit` | `1e-6` | `1e-9` | Intercept of robust linear fit in log10-log10 space. |

## 3. Known Deliberate Divergences

### Band Edges
`eegfeat` band masks follow the half-open convention $[f_{\min}, f_{\max})$, whereas the reference
spectral descriptors in `EEG_fMRI_Pipeline` used inclusive $[f_{\min}, f_{\max}]$. Half-open intervals
ensure standard bands cleanly tile frequency space without double-counting boundary bins across adjacent bands.
A bin falling exactly on a band boundary is included in the upper band in `eegfeat`, whereas the legacy
pipeline counted it in both bands.

### Two Frequency Weightings
The legacy pipeline used two distinct frequency weighting conventions:
- Band power: Trapezoidal rule weights ($\Delta f_i = \frac{f_{i+1} - f_{i-1}}{2}$ with half-weights at endpoints).
- Spectral descriptors: Centered differences (`np.gradient`), which doubles endpoint bin weights relative to trapezoid.

`eegfeat` reproduces both kernels (`trapezoid_weights` and `gradient_weights`) to ensure exact mathematical
compatibility with the reference pipeline while cleanly separating their definitions.

### Peak Frequency
`eegfeat.peak_frequency` is not equivalent to the reference `compute_peak_frequency` by design:
- `eegfeat`: Computes an interpolated parabolic argmax over raw or normalized band spectra, returning an `edge_hit` diagnostic flag.
- Reference: Fits and subtracts an aperiodic 1/f model, smooths the residual spectrum with a 1 Hz Gaussian filter, and identifies prominent peaks.

The reference peak estimator is a multi-step composite method whose port is planned as a future specialized addition.

### Grid Endpoints on Band Bounds
Frequency arrays generated via `np.logspace` do not reproduce mathematical endpoints exactly due to
floating-point roundoff (e.g. $10^{\log_{10}(1.0)}$ may land at $1.0 \pm 10^{-16}$). Band masks in
`eegfeat` use exact inequality comparisons rather than artificial tolerances ($\varepsilon$). Fixture frequency
grids must therefore not place sample points exactly on band boundaries to avoid boundary-bin ambiguity.

### Support Restriction in Equivalence Tests
The reference pipeline extraction compared in `test_equivalence.py` evaluated a 1-D time mask across all channels
without per-channel support masking. The equivalence test therefore instantiates `Spectra` with plain rectangular
windows. Channel-by-epoch support-restricted masking is verified independently in unit tests (`test_support_restricted_mask`).

## 4. Divergences Discovered During Verification

During Step 5 verification, the initial test run revealed a disagreement in spectral descriptors (centroid,
bandwidth, entropy, and edge) when using Welch PSD with default `n_fft = 1000` at $f_s = 500\text{ Hz}$.
The $0.5\text{ Hz}$ frequency resolution caused grid points to land precisely on band bounds ($8.0\text{ Hz}$,
$13.0\text{ Hz}$, and $30.0\text{ Hz}$), triggering the known half-open vs inclusive boundary divergence.

Setting `n_fft = 1024` during PSD extraction yields a frequency spacing of $500 / 1024 \approx 0.4883\text{ Hz}$,
preventing any grid point from coinciding with standard integer band boundaries. With this adjustment,
all descriptor values between `eegfeat` and the reference pipeline match to within machine precision ($< 10^{-14}$).
