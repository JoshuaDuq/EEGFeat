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
| **BandSignal Envelope** | `compute_band_data` | `1e-6` | `1e-10` | Hilbert transform and FIR bandpass filtering with reflect padding. |
| **ERDS Measures** | *transcribed formula, not a reference call* | `1e-6` | `1e-10` | Relative power change, slope, signed excursions, and onset/peak latencies. See the caveat below. |
| **Burst Features** | `_extract_burst_metrics` | `1e-6` | `1e-10` | Contiguous run count, rate, duration, amplitude, and fraction above threshold. |

### What the ERDS row does and does not establish

`extract_erds_from_precomputed` cannot be called in isolation: it requires a fully
populated `PrecomputedData` and `FeatureContext`. The fixture generator therefore
**transcribes** the per-channel expressions from
`eeg_pipeline/analysis/features/precomputed/erds.py` (around lines 500-620) rather
than calling them. Comparing against those values checks the implementation against
the documented formula; it does **not** establish agreement with a pipeline run, and
a misreading of the reference would be reproduced identically on both sides.

This is not hypothetical. The burst threshold below was calibrated on the wrong
window in both the implementation and the fixture, and the equivalence test passed
throughout, because the fixture repeated the same mistake.

The `BandSignal` envelope and the burst metrics **are** genuine reference calls
(`compute_band_data` and `_extract_burst_metrics`), and the envelope agrees
bit-for-bit on real recordings.

### Time-Domain and Entropy Measures

| Measure | Reference | Result |
|---|---|---|
| `area_under_curve` | `erp._compute_auc` | bit-identical, with and without gaps |
| `peak_amplitude`, `peak_latency` | `erp._find_peak_in_signal` | bit-identical for all three polarities, with and without prominence |
| `sample_entropy` | `antropy.sample_entropy` 0.2.2 via `signal_metrics.compute_sample_entropy` | bit-identical across 8 signal types and two embedding dimensions, including constant, too-short, NaN-laden and two-level inputs |
| `multiscale_entropy` | `signal_metrics.compute_multiscale_entropy` | bit-identical |

`sample_entropy` is implemented natively rather than depending on `antropy`, so the
library's runtime dependencies stay at numpy, scipy, pandas and mne. The cross-check
against `antropy` runs wherever it is installed and skips elsewhere.

`variance`, `peak_to_peak` and `mean_amplitude` are **not** compared: they are
`np.var`, `np.ptp` and `np.mean` over a window, and `eegfeat` computes them over the
finite samples while the reference returns NaN if any sample is non-finite. That is a
deliberate difference, not a tolerance.

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
`eegfeat.peak_frequency` now implements the reference estimator: the aperiodic
adjustment, the smoothing and the prominence-gated centre-of-gravity fallback are all
present, and `aperiodic_ratio` exposes the whitening step on its own. Four differences
remain, all deliberate:

- **Interpolation is on by default.** The reference returns the bin frequency; `eegfeat`
  refines it parabolically. Pass `interpolate=False` to match the reference exactly.
- **Smoothing handles gaps differently.** The reference drops non-finite bins and smooths
  the compacted array, so its window silently spans across a gap in index space.
  `eegfeat` keeps the frequency axis and renormalizes by how many bins were finite.
  The reference filter is a uniform boxcar (`scipy.ndimage.uniform_filter1d`), not a
  Gaussian; `eegfeat` uses the same.
- **Band and fit-range masks are half-open**, as everywhere else in `eegfeat`, against the
  reference's inclusive `<= fmax`.
- **Non-positive power is excluded from the aperiodic fit** rather than floored at
  `1e-20`. Flooring lets a zero bin sit at -20 in log space and drag the fit. Where the
  fit does not converge, `eegfeat` passes the cell through unwhitened; the reference falls
  back to an unweighted `polyfit`.

### Burst Threshold Calibration
A percentile threshold is calibrated on the window named by `baseline`, matching the
reference's `_resolve_burst_reference_envelope`, which uses the baseline mask and
declines to run on task data without one. When `baseline` is omitted, `eegfeat`
calibrates on the analysis windows instead, which is the reference's resting-state
path; that is correct for rest and wrong for task data, so the choice is recorded in
every column's unit.

The reference additionally offers `zscore` and `mad` threshold methods and a
cross-trial `threshold_reference` of `"subject"` or `"condition"`. `eegfeat`
implements only the percentile, and replaces the cross-trial modes with a
caller-supplied array. Values will differ from any pipeline run that used those
modes, by construction.

### Coverage Semantics
`BandSignal.coverage` is taken from the **filtered analytic signal**, not from the
input epochs. A single non-finite input sample propagates through the FIR
convolution and the Hilbert transform and renders the whole epoch non-finite, so
input-based coverage would report a channel as essentially intact while every output
sample is NaN. The reference has no equivalent field.

### Unmeasurable Channels in Burst Detection
For a channel whose envelope is wholly non-finite, `eegfeat` returns NaN for `count`,
`rate` and `fraction_above`, where the reference returns `0.0`. Zero bursts in a
channel that could not be measured is a false measurement; NaN with zero coverage is
the honest one.

### Channel Selection
`BandSignal.from_epochs` filters every channel present in the object, including
non-EEG channels and those in `info["bads"]`. The reference operates on
`PrecomputedData.picks`. Pass an already-picked `Epochs` if that matters.

### Measures Not Ported
- **`snr` and `muscle` ratio.** Named for an interpretation rather than a computation:
  the first asserts 1-30 Hz is signal and 40-80 Hz is noise, the second that high
  frequencies are muscle. Both reduce to a band-power ratio and are available through
  `band_power` and `band_ratio` with caller-chosen bands.
- **ERP component-label parsing.** The reference infers peak polarity from the first
  letter of a window's name and parses labels of the `N2`/`P300` form. `eegfeat` takes
  polarity as an explicit argument.
- **ERDS laterality.** Depends on which side was stimulated per trial.

### Grid Endpoints on Band Bounds
Frequency arrays generated via `np.logspace` do not reproduce mathematical endpoints exactly due to
floating-point roundoff (e.g. $10^{\log_{10}(1.0)}$ may land at $1.0 \pm 10^{-16}$). Band masks in
`eegfeat` use exact inequality comparisons rather than artificial tolerances ($\varepsilon$). Fixture frequency
grids must therefore not place sample points exactly on band boundaries to avoid boundary-bin ambiguity.

### Support Restriction in Equivalence Tests
The reference pipeline extraction compared in `test_equivalence.py` evaluated a 1-D time mask across all channels
without per-channel support masking. The equivalence test therefore instantiates `Spectra` with plain rectangular
windows. Channel-by-epoch support-restricted masking is verified independently in unit tests (`test_support_restricted_mask`).

### Rebound Latency
`rebound_latency` is reconstructed from its standard definition (maximum excursion occurring strictly after
the peak latency) rather than ported from the reference pipeline's laterality path, which was cut.
It is therefore verified by analytic unit tests rather than numerical fixture comparison.


## 4. Divergences Discovered During Verification

### ERDS baseline guard (resolved)
The implementation originally guarded the baseline at the `1e-20` power floor used
elsewhere in `eegfeat`, while the reference guards at `EPSILON_STD = 1e-12`. A channel
with a 0.1 microvolt band envelope therefore returned an ERDS of **999,900 percent**
instead of NaN. The reference's own comment names this failure: clamping instead of
invalidating "would produce artificially huge ERD/ERS ratios". `eegfeat` now uses
`_MIN_BASELINE_POWER = 1e-12`, and a regression test covers both sides of the guard.

### Burst threshold calibration (resolved)
The percentile threshold was calibrated on the whole epoch rather than the baseline
window, so on task data the stimulus response raised the very threshold used to detect
bursts within it. Measured on `sub-0001`, beta band: burst rate 0.467/s against 0.570/s
with the correct calibration, a 0.82x bias disagreeing in 33 of 64 channel-epochs. In a
synthetic case where the response is strong, active-window calibration puts the
threshold exactly at the burst amplitude and finds **zero** bursts where baseline
calibration finds three. Both the implementation and the fixture have been corrected,
and a regression test pins the difference.

During Step 5 verification, the initial test run revealed a disagreement in spectral descriptors (centroid,
bandwidth, entropy, and edge) when using Welch PSD with default `n_fft = 1000` at $f_s = 500\text{ Hz}$.
The $0.5\text{ Hz}$ frequency resolution caused grid points to land precisely on band bounds ($8.0\text{ Hz}$,
$13.0\text{ Hz}$, and $30.0\text{ Hz}$), triggering the known half-open vs inclusive boundary divergence.

Setting `n_fft = 1024` during PSD extraction yields a frequency spacing of $500 / 1024 \approx 0.4883\text{ Hz}$,
preventing any grid point from coinciding with standard integer band boundaries. With this adjustment,
all descriptor values between `eegfeat` and the reference pipeline match to within machine precision ($< 10^{-14}$).
