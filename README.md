# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeatML/)

**Labelled EEG feature extraction and leakage-safe machine learning for preprocessed EEG.**

EEGFeat works downstream of preprocessing. It accepts epoched MNE data and MNE spectral or
time-frequency objects, returns auditable `FeatureTable` objects, and keeps per-epoch and
cross-trial estimates separate.

## Installation

For a published installation:

```bash
pip install eegfeat
```

For a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[model,connectivity,microstates,importance,docs,dev]"
```

| Extra | Adds |
| :--- | :--- |
| `model` | scikit-learn modeling, metrics, nulls, uncertainty, and tuning |
| `connectivity` | `mne-connectivity` for wPLI and spectral connectivity |
| `microstates` | scikit-learn microstate segmentation |
| `importance` | SHAP explanations and their scikit-learn dependency |
| `docs` | Sphinx documentation dependencies |
| `dev` | Tests, typing, linting, and formatting tools |

## Quickstart: feature extraction

The library separates containers, extractors, and table I/O:

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_task-pain_epo.fif", preload=True)

alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
baseline = ef.Window("baseline", -0.5, -0.1)
stimulus = ef.Window("stimulus", 0.1, 0.6)
rois = {"central": ["C3", "Cz", "C4"]}

# Spectral measures consume an explicit MNE Spectrum or EpochsSpectrum.
spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, tmin=0.0, tmax=0.8)
spectra = ef.Spectra.from_spectrum(
    spectrum,
    recording="sub-01",
    estimator_parameters={"method": "welch", "fmin": 1.0, "fmax": 45.0},
)
power = ef.integrated_band_power(
    spectra, bands=[alpha, beta], groups=rois, normalize="log10"
)
peak = ef.peak_frequency(spectra, band=alpha, groups=rois)

# Time-domain and ERDS measures consume broadband or analytic band signals.
alpha_signal = ef.BandSignal.from_epochs(
    epochs, alpha, recording="sub-01", pad_sec=0.5
)
erds = ef.erds_mean([alpha_signal], baseline=baseline, windows=[stimulus], groups=rois)
bursts = ef.burst_rate(
    [alpha_signal], baseline=baseline, windows=[stimulus], groups=rois
)

table = ef.concat([power, peak, erds, bursts])
print(table.n_rows, table.names)
print(table.select(band=alpha, space="central").to_dataframe())
```

`Signal.from_epochs` wraps broadband data without filtering. `BandSignal.from_epochs` applies
an FIR band-pass filter and Hilbert transform, retains the analytic signal, and exposes its
envelope, phase, and power. Both require a `recording` label so row identities remain stable.

For multitaper PSD, compute with `normalization="full"` and record the same parameter in
`estimator_parameters`; `Spectra.from_spectrum` rejects MNE's non-density default. For Morlet
time-frequency power, use `Spectra.from_tfr` with the original `n_cycles`; it masks coefficients
whose wavelet support extends outside each requested window.

## Tables and export

Feature metadata stores the measure, band, space, window, normalization, unit, parameters, and
provenance. `coverage` and `flags` remain separate from numeric values.

```python
from eegfeat.io import read_table, write_table

write_table(table, "sub-01_features.tsv", rows=epochs.metadata)
restored = read_table("sub-01_features.tsv")
```

This writes a values TSV, a `_coverage.tsv` matrix, and a JSON sidecar. `read_table` restores
the feature metadata; `read_dataset` combines runner-generated per-epoch tables for modeling.
Use `stack_rows(..., columns="union")` for in-memory cohorts whose recordings have different
channel sets. Cross-trial tables have group rows and must not be broadcast into per-epoch data.

## Available measures

| Category | Public functions | Row semantics |
| :--- | :--- | :--- |
| Spectral power | `mean_psd`, `integrated_band_power`, `mean_tfr_power` | Per epoch |
| Spectral descriptors | `peak_frequency`, `spectral_centroid`, `spectral_bandwidth`, `spectral_edge`, `spectral_entropy`, `aperiodic`, `aperiodic_ratio` | Per epoch |
| Derived power | `band_ratio`, `asymmetry` | Per epoch |
| Time domain | `variance`, `mean_amplitude`, `peak_to_peak`, `area_under_curve`, `peak_amplitude`, `peak_latency`, `amplitude_quantile`, `kurtosis`, `line_length`, `root_mean_square`, `skewness`, `zero_crossing_rate`, `hjorth_mobility`, `hjorth_complexity` | Per epoch |
| Bursts and ERDS | `burst_count`, `burst_rate`, `burst_duration`, `burst_amplitude`, `fraction_above_threshold`, `erds_mean`, `erds_slope`, `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration`, `erds_onset_latency`, `erds_peak_latency`, `erds_rebound_latency` | Per epoch |
| Phase and connectivity | `itpc`, `ppc`, `envelope_correlation`, `spectral_connectivity`, `wpli` | Cross-trial group rows |
| Phase-amplitude coupling | `pac` | Per epoch; no surrogate correction is applied |
| Graph summaries | `global_efficiency`, `clustering_coefficient` | Reduce a pairwise table |
| Complexity | `sample_entropy`, `multiscale_entropy`, `higuchi_fractal_dimension` | Per epoch |
| Microstates | `segment`, `microstate_coverage`, `microstate_duration`, `microstate_occurrence`, `microstate_transitions` | Per epoch; requires `microstates` |
| Supervised spatial filters | `CommonSpatialPattern`, `csp_features` | Cross-fitted per epoch |

`spectral_connectivity` and `wpli` require the `connectivity` extra. `csp_features` requires
folds supplied by the caller and fits each spatial filter on training rows only. Cross-trial
measures require at least two epochs per group and are intentionally kept out of modeling.

## Batch processing with `eegfeat`

The CLI reads FIF epochs files and applies one TOML recipe to every matching recording:

```bash
eegfeat init recipe.toml
eegfeat check recipe.toml
eegfeat run recipe.toml
```

`python -m eegfeat` is equivalent. `check` writes nothing and tries the first recording after
validating the recipe. Paths are relative to the recipe file; unknown sections and keys are
errors.

Minimal recipe:

```toml
[inputs]
root = "derivatives/preprocessed/eeg"
pattern = "**/*_epo.fif"
picks = "eeg"
exclude_bads = true

[output]
root = "derivatives/eegfeat"
epoch_metadata = true

[bands]
alpha = [8.0, 13.0]
beta = [13.0, 30.0]

[windows]
baseline = [-0.5, 0.0]
stimulus = [0.0, 1.0]

[rois]
central = ["C3", "Cz", "C4"]

[trials]
by = "event"

[[features]]
measure = "integrated_band_power"
bands = ["alpha", "beta"]
normalize = "log10"
spatial = ["rois", "global"]

[[features]]
measure = "erds_mean"
bands = ["alpha"]
baseline = "baseline"
windows = ["stimulus"]
spatial = ["rois", "global"]

[[features]]
measure = "itpc"
bands = ["alpha"]
spatial = ["rois"]
```

The input tree is mirrored under the output root. Each recording produces per-epoch
`*_features.tsv`, coverage, and JSON files; cross-trial entries produce a separate
`*_crosstrial.tsv` bundle. `eegfeat_run.json` records successes and failures. Use
`--overwrite` to replace an earlier run, `--n-jobs N` for MNE filtering/spectral work, and
`--progress-json` for machine-readable progress. The runner processes only FIF epochs files.

## Machine learning

Install `eegfeat[model]` for grouped, nested cross-fitting. Modeling accepts per-epoch tables
only; targets must contain matching `recording`, `epoch`, and `event` keys.

```python
from pathlib import Path
import numpy as np
import eegfeat.model as efm
from eegfeat.io import read_dataset

dataset = read_dataset(sorted(Path("derivatives/eegfeat").rglob("*_features.tsv")))
selection = efm.Selection(
    measure=("band_power", "erds_mean"),
    band=("alpha", "beta"),
)
design = efm.build_design(
    dataset.table,
    dataset.targets,
    target="rating",
    groups="subject",
    runs="run",
    selection=selection,
)

config = efm.PreprocessingConfig(
    max_feature_missingness=0.2,
    max_subject_missingness=0.5,
)
pipeline = efm.ridge_pipeline(config, seed=42, n_covariates=design.n_covariates)
folds = efm.loso_folds(design.groups)
inner = efm.InnerSplit(grouping="subject", n_splits=5)
predictions = efm.cross_fit_regression(
    folds,
    design.X,
    design.y,
    design.groups,
    pipeline,
    efm.ridge_grid(),
    inner=inner,
    seed=42,
    runs=design.runs,
    harmonization="intersection",
)

y_true, y_pred, eval_groups, _, _ = efm.fold_results(predictions, groups=design.groups)
metrics, _ = efm.regression_metrics(
    y_true, y_pred, groups=np.asarray(eval_groups, dtype=object)
)
print(metrics["subject_level_r"])
```

Use `within_subject_folds(groups, runs, inner_splits=3)` for run-disjoint within-subject
evaluation. Regression pipelines include ridge, elastic net, and random forest; classification
pipelines include logistic, SVM, random forest, and ensemble models, with labels coded `0` and
`1`. `permutation_test` refits the full cross-fitting procedure, while
`prediction_intervals` offers `split`, `cv_plus`, and `quantile` calibration. Calibration is
trial-pooled and does not provide a distribution-free guarantee for a new participant.

Permutation importance is available with the model extra; SHAP importance additionally needs
`eegfeat[importance]`. Aggregate importance only over metadata belonging to the exact feature
columns used by the fitted design.

## Scientific guardrails

- Frequency bands use half-open `[fmin, fmax)` selection semantics.
- Bands outside the recording passband raise; partially truncated bands warn and record coverage.
- Morlet windows retain only coefficients with complete temporal support.
- Baselines for bursts and ERDS are explicit and cannot be learned from the analysis window.
- Cross-trial summaries remain separate from per-epoch modeling tables.
- Preprocessing, harmonization, selection, and tuning are fitted inside training folds.
- Subject-level metrics weight subjects equally by default.

## Documentation and development

The full guides are in [`docs/`](docs/): [quickstart](docs/quickstart.rst),
[methods](docs/methods.rst), [modeling](docs/modeling.rst),
[runner](docs/runner.rst), and [API reference](docs/api.rst).

```bash
python3 -m pytest
python3 -m mypy src
python3 -m ruff check src tests
```

## License

MIT License. See [LICENSE](LICENSE).
