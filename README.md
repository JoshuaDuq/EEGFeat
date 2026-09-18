# eegfeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Typing: Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](docs/)

Labelled spectral, temporal, and connectivity feature extraction for MNE objects.

`eegfeat` maps precomputed MNE-Python objects (`Spectrum`, `EpochsTFR`, and `Epochs`) to a structured `FeatureTable`: a numeric array paired with column-level `FeatureMeta` records. Frequency band bounds, spatial units (channels or regions of interest), temporal windows, normalization baselines, physical units, and spectral resolutions are stored as typed record fields rather than encoded into column name strings.

The library computes no time-frequency decompositions or raw filtering passes of its own. It operates downstream of standard estimators, standardizing the extraction of univariate, multivariate, dynamic, and non-linear electrophysiological features.

---

## Scientific Motivation

Feature extraction pipelines in cognitive and clinical electrophysiology often rely on ad-hoc scripts that parse composite string labels and commit subtle signal-processing errors:

- **Wavelet temporal contamination**: Standard slicing of time-frequency representations (TFRs) over fixed time intervals disregards frequency-dependent wavelet duration. Low-frequency Morlet wavelets possess temporal supports spanning several seconds, allowing pre-stimulus baseline power or edge discontinuities to contaminate post-stimulus windows.
- **Aperiodic bias in peak estimation**: Discrete argmax searches for peak oscillatory frequencies (such as individual alpha frequency) over raw power spectral densities are systematically biased toward lower band boundaries by the background $1/f^\chi$ power law.
- **Circular threshold calibration in burst detection**: Defining amplitude thresholds on task windows allows stimulus-evoked responses to inflate the threshold, suppressing detection of sustained oscillatory bursts.
- **Pseudo-replication in cross-trial estimators**: Quantities inherently defined across ensembles of trials—such as inter-trial phase coherence (ITPC), envelope correlation (AEC), and weighted phase lag index (wPLI)—are frequently broadcast across single-trial rows in tabular datasets, inflating nominal sample sizes in downstream inferential models.
- **Unchecked sample loss**: Non-finite samples introduced by artifact rejection or boundary padding either propagate silently or are coerced to zero, confounding absent measurements with zero-power estimates.

`eegfeat` enforces mathematical and structural safeguards against each of these failure modes.

---

## Methodological Principles

- **Structured metadata**: Every column in a `FeatureTable` is governed by an immutable `FeatureMeta` instance detailing its measure, frequency band, channel or ROI pick, window, normalization convention, physical unit, estimator provenance, and frequency grid resolution. Column selection is performed via attribute queries rather than regular expressions.
- **Support-restricted integration**: When integrating Morlet TFRs, per-frequency temporal supports ($n_\text{cycles} / (2 f)$) define the boundary margins. Time-frequency coefficients that cannot be fully attributed to the designated window are masked out.
- **Aperiodic residual whitening**: Peak frequency estimation divides out an iteratively fitted robust linear $1/f$ baseline (using Huber residuals in log-log coordinates), refines local maxima via three-point parabolic interpolation, and verifies minimum peak prominence over the band median, falling back to spectral centre-of-gravity when no clear peak is identifiable.
- **Baseline-referenced burst detection**: Contiguous envelope excursions above an amplitude percentile are calibrated against an unperturbed reference baseline window, preventing stimulus-induced threshold inflation.
- **Strict row semantics**: Measures estimated across trials return one row per trial group and assign explicit `row_labels`. Horizontal table concatenation (`concat`) raises an exception if group-level and single-trial tables are merged without an explicit broadcasting strategy.
- **Per-cell coverage accounting**: Every feature value is accompanied by an entry in a parallel `coverage` matrix, recording the fraction of finite, valid time-frequency or time-domain samples that contributed to the calculation.

---

## Feature Scope

The library extracts nine families of electrophysiological measures:

- **Spectral power**: Band-integrated power using trapezoidal interval weighting (compensating for non-uniform or logarithmic frequency grids), inter-band power ratios, and hemispheric asymmetry indices. Accepts `Spectra` built from Welch/multitaper PSD or Morlet TFR.
- **Spectral descriptors**: Spectral centroid (weighted mean frequency), spectral bandwidth (weighted frequency dispersion), spectral Shannon entropy normalized by bin count, and spectral edge frequency (cumulative power quantile, default 95%). Accepts `Spectra`.
- **Aperiodic background**: Robust iterative linear fits in $\log_{10}(P) - \log_{10}(f)$ coordinates, returning aperiodic slope and offset parameters, as well as periodic residual spectra. Accepts `Spectra`.
- **Time-domain metrics**: Windowed variance, mean amplitude, peak-to-peak amplitude, trapezoidal area under the curve skipping non-finite gaps, signed peak amplitude, and peak latency. Accepts `Signal` or `BandSignal`.
- **Oscillatory bursts**: Contiguous suprathreshold envelope excursion count, occurrence rate (events per second), mean event duration, mean event peak amplitude, and total time fraction above threshold. Accepts `BandSignal` envelopes.
- **ERDS dynamics**: Mean relative power excursion (percentage or decibel scaling), excursion rate of change (linear slope), separate negative (ERD) and positive (ERS) magnitudes and durations, onset latency relative to baseline variance, peak latency, and rebound latency. Accepts `BandSignal`.
- **Phase and connectivity**: Inter-trial phase coherence (ITPC), amplitude-normalized phase-amplitude coupling (PAC mean vector length), orthogonalized/raw amplitude envelope correlation (AEC), weighted phase lag index (wPLI via MNE-Connectivity), and derived graph topology (weighted global efficiency, binarized clustering coefficient). Accepts `BandSignal` or `Spectra`.
- **Complexity**: Single-scale Chebyshev sample entropy and coarse-grained multiscale entropy, implemented natively without external C extensions. Accepts `Signal` or `BandSignal`.
- **Microstates**: Global field power (GFP) peak topography clustering, continuous sample assignment with minimum duration smoothing, mean state duration, state occurrence rate, fractional time coverage, and directional transition probabilities. Accepts `Signal`.

---

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/JoshuaDuq/eegfeat.git
cd eegfeat
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Optional Dependencies

The core package depends exclusively on `numpy`, `scipy`, `pandas`, and `mne`. Optional estimator dependencies can be specified during installation:

- `pip install -e ".[connectivity]"` installs `mne-connectivity` for weighted phase lag index calculations.
- `pip install -e ".[microstates]"` installs `scikit-learn` for GFP peak clustering.
- `pip install -e ".[knee]"` installs `specparam` for spectral knee parameterization.
- `pip install -e ".[docs]"` installs `furo`, `myst-parser`, `sphinx-design`, and `sphinx-copybutton`.
- `pip install -e ".[dev]"` installs testing harnesses (`pytest`), type checkers (`mypy`), and linters (`ruff`, `black`).

---

## Basic Usage

```python
import mne
import eegfeat as ef

# 1. Load epoched data
epochs = mne.read_epochs("sub-01_task-rest_proc-clean_epo.fif", preload=True)

# 2. Compute power spectral density via MNE and wrap in a Spectra container
spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, n_fft=1024)
spectra = ef.Spectra.from_spectrum(spectrum)

# 3. Extract trapezoidally integrated band power
bands = [
    ef.Band("theta", 4.0, 8.0),
    ef.Band("alpha", 8.0, 13.0),
    ef.Band("beta", 13.0, 30.0),
]
power_table = ef.band_power(spectra, bands=bands, include_global=True)

# 4. Extract aperiodic-whitened peak alpha frequency
peak_table = ef.peak_frequency(spectra, band=ef.Band("alpha", 8.0, 13.0))

# 5. Concatenate tables into a single FeatureTable
features = ef.concat([power_table, peak_table])

# 6. Query columns by structured metadata attributes
alpha_features = features.select(band=ef.Band("alpha", 8.0, 13.0))

# 7. Export to a pandas DataFrame with canonical column names
df = features.to_dataframe()

# 8. Export to BIDS-compliant TSV files with JSON sidecar
from eegfeat.io import write_table

write_table(features, "sub-01_features.tsv")
```

---

## Verification and Testing

The implementation is verified against empirical electrophysiological fixtures and synthetic signal models across all supported estimators down to numerical precision. All continuous features match analytic baselines and standard published algorithms within machine precision ($r_\text{tol} \le 10^{-6}$, $a_\text{tol} \le 10^{-9}$), with discrete edge searches and graph topology matching bit-identically.

```bash
# Run test suite
pytest

# Run strict static type validation
mypy
```

---

## Documentation

Complete documentation is built using Sphinx with the Furo theme:

- **Quick Start (`docs/quickstart.rst`)**: End-to-end workflows from MNE objects to feature matrices.
- **Methods Reference (`docs/methods.rst`)**: Mathematical definitions, weighting kernels, and algorithmic details.
- **API Reference (`docs/api.rst`)**: Class interfaces, function signatures, and parameter documentation.

To compile the documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

---

## License

MIT License. See [LICENSE](LICENSE) for terms.
