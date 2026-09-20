# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeatML/)

Labelled EEG feature extraction and leakage-safe modeling for preprocessed MNE data.
EEGFeat returns self-describing `FeatureTable` objects: values stay linked to their
band, window, spatial unit, normalization, coverage, and computation parameters.

## Choose a workflow

- Use the [Python API](#python-api) for interactive control or custom pipelines.
- Use the [batch runner](#batch-runner) to apply one recipe to many FIF epochs files.
- Use [modeling](#modeling) only with per-epoch tables; cross-trial summaries stay separate.

## Install

Requirements: Python 3.11 or newer and preprocessed MNE `Epochs` data.

```bash
python -m pip install eegfeat
```

For a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,model,connectivity,microstates,importance,docs]"
```

Optional extras:

| Extra | Enables |
| :--- | :--- |
| `model` | scikit-learn pipelines, grouped cross-fitting, metrics, nulls, and uncertainty |
| `connectivity` | `mne-connectivity` for spectral connectivity and wPLI |
| `microstates` | scikit-learn microstate segmentation |
| `importance` | SHAP-based importance; permutation importance is in `model` |
| `docs` | Sphinx documentation |
| `dev` | tests, typing, linting, and formatting |

## Python API

The usual sequence is: load epochs, define bands/windows, build a spectral or time-domain
container, compute tables, then concatenate compatible tables.

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_task-rest_epo.fif", preload=True)

alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
baseline = ef.Window("baseline", -0.5, -0.1)
stimulus = ef.Window("stimulus", 0.1, 0.6)
rois = {"central": ["C3", "Cz", "C4"]}

# PSD features use an explicit MNE Spectrum or EpochsSpectrum.
spectrum = epochs.compute_psd(
    method="welch", fmin=1.0, fmax=45.0, tmin=0.0, tmax=0.8
)
spectra = ef.Spectra.from_spectrum(
    spectrum,
    recording="sub-01",
    estimator_parameters={
        "method": "welch",
        "fmin": 1.0,
        "fmax": 45.0,
        "tmin": 0.0,
        "tmax": 0.8,
    },
)
power = ef.integrated_band_power(
    spectra, bands=[alpha, beta], groups=rois, normalize="log10"
)
peak = ef.peak_frequency(spectra, band=alpha, groups=rois)

# BandSignal filters, Hilbert-transforms, and retains the analytic signal.
alpha_signal = ef.BandSignal.from_epochs(
    epochs, alpha, recording="sub-01", pad_sec=0.5
)
erds = ef.erds_mean(
    [alpha_signal], baseline=baseline, windows=[stimulus], groups=rois
)
bursts = ef.burst_rate(
    [alpha_signal], baseline=baseline, windows=[stimulus], groups=rois
)

features = ef.concat([power, peak, erds, bursts])
print(features.to_dataframe())
```

Use `Signal.from_epochs` for broadband time-domain measures. Use `BandSignal.from_epochs`
for band envelopes, phase, power, bursts, ERDS, and band-limited time-domain measures. Use
`Spectra.from_tfr` for an uncorrected MNE `EpochsTFR`; pass the original `n_cycles` so
wavelet support can be checked. For multitaper PSD, compute with
`normalization="full"` and record the same setting in `estimator_parameters`.

### Tables and files

```python
from eegfeat.io import read_dataset, read_table, write_table

rows = epochs.metadata  # None is valid; otherwise it must have one row per epoch.
write_table(features, "sub-01_features.tsv", rows=rows)
restored = read_table("sub-01_features.tsv")

dataset = read_dataset([
    "sub-01_features.tsv",
    "sub-02_features.tsv",
])
```

`write_table` creates the values TSV, a `_coverage.tsv` matrix, and a JSON sidecar containing
metadata, flags, row identities, and provenance. `read_dataset` loads per-epoch bundles and
returns `dataset.table` plus descriptor columns in `dataset.targets`. Include target, subject,
and run columns in `rows` when the tables will be modeled. Use
`stack_rows(..., columns="union")` for in-memory cohorts with different channel sets.

## Batch runner

The runner reads FIF epochs files recursively and mirrors their input paths below the output
root. Start with a generated recipe:

```bash
eegfeat init recipe.toml
eegfeat check recipe.toml
eegfeat run recipe.toml
```

`python -m eegfeat` is equivalent. `check` validates the recipe, surveys the recordings, and
tries the first one without writing output. Run it before a long batch job.

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
windows = ["stimulus"]
spatial = ["rois"]
```

Recipe rules that matter most:

- Paths are relative to the recipe file. Unknown sections and keys are errors.
- Bands are `[low, high)` in Hz. Windows are in seconds relative to the epoch origin.
- `baseline` is consumed by normalization, burst, and ERDS measures; it is not also an
  analysis window unless the measure permits that combination.
- `spatial` accepts `channels`, `rois`, and, where supported, `global`.
- Cross-trial measures such as `itpc`, `ppc`, `envelope_correlation`, and
  `wpli` need trial groups with enough epochs and write group-row tables.
- `series = ["broadband", "alpha"]` selects raw and band-envelope time-domain inputs;
  `pairs` configures PAC; `ratios` and `asymmetry` configure derived power features.

Useful options:

```bash
eegfeat run recipe.toml --overwrite --n-jobs 4
eegfeat run recipe.toml --progress-json
```

`--overwrite` replaces an existing result bundle. `--n-jobs` is passed to MNE filtering and
spectral estimation. `--progress-json` emits one JSON event per line. The runner exits `0`
when all recordings succeed, `1` when a run completes with recording failures, and `2` when
the recipe or inputs prevent the run from starting.

Per-epoch measures are written to `*_features.tsv`; cross-trial measures are written to a
separate `*_crosstrial.tsv` bundle. The runner also writes coverage, JSON sidecars, and an
`eegfeat_run.json` summary. Only FIF epochs files are read.

## Measures

The public functions are exported from `eegfeat` and documented in the [API reference](docs/api.rst).

| Input or purpose | Examples | Row semantics |
| :--- | :--- | :--- |
| PSD/TFR power | `mean_psd`, `integrated_band_power`, `mean_tfr_power` | Per epoch |
| Spectral descriptors | `peak_frequency`, `spectral_centroid`, `spectral_bandwidth`, `spectral_edge`, `spectral_entropy`, `aperiodic`, `aperiodic_ratio` | Per epoch |
| Time domain | `variance`, `mean_amplitude`, `peak_to_peak`, `area_under_curve`, `peak_amplitude`, `peak_latency`, `amplitude_quantile`, `kurtosis`, `line_length`, `root_mean_square`, `skewness`, `zero_crossing_rate`, `hjorth_mobility`, `hjorth_complexity` | Per epoch |
| Bursts and ERDS | `burst_count`, `burst_rate`, `burst_duration`, `burst_amplitude`, `fraction_above_threshold`, `erds_mean`, `erds_slope`, `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration`, `erds_onset_latency`, `erds_peak_latency`, `erds_rebound_latency` | Per epoch |
| Cross-trial phase/connectivity | `itpc`, `ppc`, `envelope_correlation`, `spectral_connectivity`, `wpli` | One row per trial group |
| Phase-amplitude coupling | `pac` | Per epoch; no surrogate correction |
| Derived power | `band_ratio`, `asymmetry` | Per epoch |
| Graph summaries | `global_efficiency`, `clustering_coefficient` | Reduce a pairwise table |
| Complexity | `sample_entropy`, `multiscale_entropy`, `higuchi_fractal_dimension` | Per epoch |
| Microstates | `segment`, `microstate_coverage`, `microstate_duration`, `microstate_occurrence`, `microstate_transitions` | Per epoch |
| Supervised spatial filters | `CommonSpatialPattern`, `csp_features` | Cross-fitted per epoch |

`csp_features` produces descriptive held-out features and cannot be used as a fixed
classifier design, even with the same folds. For prediction, fit `CommonSpatialPattern`
inside each training fold and transform both train and test data with that same fit;
repeat this inside inner tuning. `build_design` rejects assembled CSP columns.
`spectral_connectivity` and `wpli` require the `connectivity` extra; microstate measures
require `microstates`.

## Modeling

Install the model extra and load only per-epoch feature bundles:

```bash
python -m pip install "eegfeat[model]"
```

Targets must contain matching `recording`, `epoch`, and `event` keys, plus the target and
grouping columns. Keep cross-trial tables out of the design: their rows are trial groups, not
independent epochs. Leave-one-group-out evaluation also requires at least two unique groups;
the checked-in example outputs show one recording, while the modeling example below expects a
multi-recording cohort.

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
predictions = efm.cross_fit_regression(
    efm.loso_folds(design.groups),
    design.X,
    design.y,
    design.groups,
    pipeline,
    efm.ridge_grid(),
    inner=efm.InnerSplit(grouping="subject", n_splits=5),
    seed=42,
    runs=design.runs,
    harmonization="intersection",
)

y_true, y_pred, eval_groups, _, _ = efm.fold_results(
    predictions, groups=design.groups
)
metrics, _ = efm.regression_metrics(
    y_true, y_pred, groups=np.asarray(eval_groups, dtype=object)
)
print(metrics["subject_level_r"])
```

Use `within_subject_folds` for run-disjoint within-subject evaluation. Regression pipelines
include ridge, elastic net, and random forest; classification includes logistic, SVM, random
forest, and ensemble pipelines with binary labels. Preprocessing and hyperparameter tuning
are fitted inside the training folds. `permutation_test` refits the full cross-fitting
procedure; SHAP importance requires the `importance` extra.

## Scientific guardrails

- Bands outside an input recording's passband raise; partially truncated bands warn and record
  coverage.
- Morlet windows keep only coefficients with complete temporal support.
- ERDS and burst baselines are explicit; they are not learned from the analysis window.
- Non-finite samples become coverage and flags, not silently dropped temporal neighbours.
- Cross-trial tables are never broadcast into per-epoch modeling data.
- Subject-level metrics weight subjects equally by default.

## Documentation and development

- [Quickstart](docs/quickstart.rst) — Python containers, measures, and table I/O
- [Runner guide](docs/runner.rst) — complete TOML schema and CLI behavior
- [Modeling guide](docs/modeling.rst) — design construction, cross-fitting, metrics, nulls,
  and importance
- [Methods](docs/methods.rst) — definitions and assumptions
- [API reference](docs/api.rst) — public signatures

From a source checkout:

```bash
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy
```

## License

MIT License. See [LICENSE](LICENSE).
