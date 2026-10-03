# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeat/)

**Spectral**, **temporal**, **connectivity**, and **complexity** features for **MNE** objects, returned as `FeatureTable`, with reproducible cohort analysis and grouped model evaluation.

Every column keeps its band, window, spatial unit, normalization, coverage, and computation parameters, and its name carries the same fields:

```text
eeg_erds-mean_alpha_central_stimulus_db_pd6572187a139
│   │         │     │       │        │  └─ parameter hash (first 12 hex of SHA-256)
│   │         │     │       │        └─ normalization (raw, log10, log_ratio, db, percent)
│   │         │     │       └─ time window, or "all"
│   │         │     └─ channel, ROI, or "global"
│   │         └─ band, or "broadband"
│   └─ measure
└─ domain
```

**Documentation:** [joshuaduq.github.io/EEGFeat](https://joshuaduq.github.io/EEGFeat/)

## Install

Python 3.11 or newer. Not on PyPI; install from source.

```bash
git clone https://github.com/JoshuaDuq/EEGFeat.git
cd EEGFeat
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[model]"
```

Core dependencies are `numpy`, `scipy`, `pandas`, and `mne`. Optional extras:

| Extra | Adds |
| :--- | :--- |
| `model` | Pipelines, grouped cross-fitting, metrics, nulls, uncertainty |
| `connectivity` | Spectral connectivity and wPLI via `mne-connectivity` |
| `microstates` | Microstate segmentation |
| `spectral-model` | Fixed/knee spectral parameterization via specparam |
| `irasa` | Aperiodic/oscillatory separation via NeuroDSP |
| `cycles` | Cycle waveform and burst features via ByCycle |
| `complexity` | Permutation entropy, Lempel–Ziv complexity and DFA via AntroPy |
| `pac` | PAC surrogate inference via Tensorpac |
| `riemann` | Training-fitted covariance and tangent-space decoding via pyRiemann |
| `bids` | Native BIDS discovery and loading via MNE-BIDS |
| `importance` | SHAP (permutation importance is in `model`) |
| `preprocessing` | Raw-to-epochs workflow for one recording or a cohort |
| `preprocessing-auto` | PyPREP, ICLabel, Picard, autoreject |
| `preprocessing-gui` | MNE Qt viewers for interactive review |
| `docs`, `dev` | Documentation build; tests, typing, lint |

## Quick start

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_epo.fif", preload=True)

alpha = ef.Band("alpha", 8.0, 13.0)
baseline = ef.Window("baseline", -0.5, -0.1)
stimulus = ef.Window("stimulus", 0.1, 0.6)
rois = {"central": ["C3", "Cz", "C4"]}

spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0)
spectra = ef.Spectra.from_spectrum(
    spectrum,
    recording="sub-01",
    estimator_parameters={"method": "welch", "fmin": 1.0, "fmax": 45.0},
)
power = ef.integrated_band_power(spectra, bands=[alpha], groups=rois, normalize="log10")

signal = ef.BandSignal.from_epochs(epochs, alpha, recording="sub-01", pad_sec=0.5)
erds = ef.erds_mean([signal], baseline=baseline, windows=[stimulus], groups=rois)

features = ef.concat([power, erds])
features.to_dataframe()
```

`estimator_parameters` goes into the column hash. MNE does not store Welch `n_per_seg`, `n_overlap`, or `window` on a `Spectrum`, so record every setting that varies. Otherwise two different estimates share a column name.

More workflows: [Quick Start](https://joshuaduq.github.io/EEGFeat/quickstart.html). Tables, files, and stacking: [Feature Tables and Files](https://joshuaduq.github.io/EEGFeat/guides/tables.html).

## Command line

**Batch runner.** One TOML recipe applied to a folder of epochs files. Each written table loads back as a `FeatureTable`.

```bash
eegfeat init recipe.toml     # commented recipe; --template task or resting
eegfeat check recipe.toml    # validate and time the first recording; writes nothing
eegfeat run recipe.toml      # every recording
eegfeat status recipe.toml   # what is done, and what to run next
eegfeat report recipe.toml quality.html --by recording
```

**Preprocessing.** Optional raw-to-epochs cleaning from one YAML recipe, with review gates you can require, auto-accept, or skip.

```bash
eegfeat preprocess init preprocessing.yaml --mode events
eegfeat preprocess check preprocessing.yaml
eegfeat preprocess run preprocessing.yaml
```

Recipes, flags, exit codes, and the review workflow are in the [Runner](https://joshuaduq.github.io/EEGFeat/guides/runner.html) and [Preprocessing](https://joshuaduq.github.io/EEGFeat/guides/preprocessing.html) guides. [`tui/`](tui) is an optional preprocessing terminal front end (Go 1.24+).

## Modeling

`eegfeat.model` builds a design from per-epoch tables and evaluates it with group-disjoint nested validation. Preprocessing and tuning are fit inside each training fold, and permutation nulls cover the full procedure. Ridge, elastic net, random forest, logistic regression, SVM, and ensembles are included. See [Predictive Modeling](https://joshuaduq.github.io/EEGFeat/guides/modeling.html).

Reusable microstate templates and fold-fitted CSP, microstate, covariance and tangent-space transforms support analysis across recordings without fitting on held-out data. `eegfeat.group` provides a separate design interface for trial-group estimates. Fixed quality policies preserve coverage, flags and exclusion evidence in the design.

```bash
eegfeat model init model.yaml
eegfeat model check model.yaml
eegfeat model run model.yaml
```

Saved feature bundles require checksummed schema 2 manifests. Input content, resolved defaults, software and source identity govern extraction status. Regenerate older outputs; modified or incomplete bundles raise errors.

## What it computes

| Family | Functions |
| :--- | :--- |
| Power | `mean_psd`, `integrated_band_power`, `mean_tfr_power`, `periodic_power`, `band_ratio`, `asymmetry` |
| Spectral shape | `peak_frequency`, `spectral_centroid`, `spectral_bandwidth`, `spectral_edge`, `spectral_entropy`, `aperiodic`, `aperiodic_ratio` |
| Spectral separation | `spectral_parameterization`, `irasa` |
| Time domain | `variance`, `mean_amplitude`, `peak_to_peak`, `area_under_curve`, `peak_amplitude`, `peak_latency`, `amplitude_quantile`, `kurtosis`, `skewness`, `line_length`, `root_mean_square`, `zero_crossing_rate`, `hjorth_mobility`, `hjorth_complexity` |
| Bursts and ERDS | `burst_count`, `burst_rate`, `burst_duration`, `burst_amplitude`, `fraction_above_threshold`, `erds_mean`, `erds_slope`, `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration`, `erds_onset_latency`, `erds_peak_latency`, `erds_rebound_latency` |
| Phase and connectivity | `itpc`, `ppc`, `envelope_correlation`, `spectral_connectivity`, `spectral_connectivity_time`, `wpli`, `pac`, `pac_surrogates`, `global_efficiency`, `clustering_coefficient` |
| Complexity | `sample_entropy`, `multiscale_entropy`, `higuchi_fractal_dimension`, `permutation_entropy`, `lempel_ziv_complexity`, `detrended_fluctuation` |
| Cycle waveforms | `cycle_features` |
| Microstates | `segment`, `microstate_coverage`, `microstate_duration`, `microstate_occurrence`, `microstate_transitions` |
| Spatial filters | `CommonSpatialPattern`, `csp_features` |

Definitions are in [Methods](https://joshuaduq.github.io/EEGFeat/methods/index.html). Signatures are in the [API Reference](https://joshuaduq.github.io/EEGFeat/api/index.html). Numerical and boundary tests cover the new APIs; public-dataset evidence for established methods is in [Validation](https://joshuaduq.github.io/EEGFeat/guides/validation.html). Each dataset validation run saves its own immutable evidence snapshot.

## Development

```bash
python -m pip install -e ".[dev,model,connectivity,microstates,importance,preprocessing,preprocessing-auto,bids,spectral-model,irasa,cycles,complexity,pac,riemann,docs]"
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy
EEGFEAT_DATASETS=1 python -m pytest tests/validation -ra   # public-dataset suite, ~350 MB on first run
```

[`paradigm_specific/`](paradigm_specific) holds this study's raw-to-BIDS conversion scripts. They need the `bids` extra, and `eegfeat` does not import them.

## License

MIT. See [LICENSE](LICENSE).
