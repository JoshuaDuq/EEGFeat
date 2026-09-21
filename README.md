# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeat/)

Labelled EEG feature extraction and leakage-safe modeling for preprocessed MNE data.

Features come back as `FeatureTable` objects, not bare arrays: every value stays linked to its
band, window, spatial unit, normalization, coverage, and computation parameters. Column names
carry the same information, so a table reads on its own months later:

```text
eeg_erds-mean_alpha_central_stimulus_db_pd6572187a139
│   │         │     │       │        │  └─ parameter hash (first 12 hex)
│   │         │     │       │        └─ normalization: raw, log10, log_ratio, db, percent
│   │         │     │       └─ time window, or "all"
│   │         │     └─ channel, ROI, or "global"
│   │         └─ band, or "broadband"
│   └─ measure
└─ domain
```

## Where to start

| Goal | Start here |
| :--- | :--- |
| Explore interactively or build a custom pipeline | [Python API](#python-api) |
| Apply one recipe to many epochs files | [Batch runner](#batch-runner) |
| Predict a target from per-epoch features | [Modeling](#modeling) |
| See real output — tables, sidecars, model scores | [`examples/`](examples/) |

## Install

Requires Python 3.11+ and preprocessed MNE `Epochs`. Not yet on PyPI, so install from source:

```bash
git clone https://github.com/JoshuaDuq/EEGFeat.git
cd EEGFeat
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[model]"
```

Core dependencies are just `numpy`, `scipy`, `pandas`, and `mne`. The rest are extras:

| Extra | Enables |
| :--- | :--- |
| `model` | scikit-learn pipelines, grouped cross-fitting, metrics, nulls, and uncertainty |
| `connectivity` | `mne-connectivity` for spectral connectivity and wPLI |
| `microstates` | scikit-learn microstate segmentation |
| `importance` | SHAP-based importance; permutation importance is in `model` |
| `docs` | Sphinx documentation |
| `dev` | tests, typing, linting, and formatting |

To contribute, take them all: `pip install -e ".[dev,model,connectivity,microstates,importance,docs]"`

## Python API

Load epochs, define bands and windows, build a spectral or time-domain container, compute
tables, concatenate.

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_task-rest_epo.fif", preload=True)

alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
baseline = ef.Window("baseline", -0.5, -0.1)
stimulus = ef.Window("stimulus", 0.1, 0.6)
rois = {"central": ["C3", "Cz", "C4"]}

# PSD features take an explicit MNE Spectrum or EpochsSpectrum.
spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, tmin=0.0, tmax=0.8)
spectra = ef.Spectra.from_spectrum(
    spectrum,
    recording="sub-01",
    estimator_parameters={
        "method": "welch", "fmin": 1.0, "fmax": 45.0, "tmin": 0.0, "tmax": 0.8
    },
)
power = ef.integrated_band_power(spectra, bands=[alpha, beta], groups=rois, normalize="log10")
peak = ef.peak_frequency(spectra, band=alpha, groups=rois)

# BandSignal filters, Hilbert-transforms, and retains the analytic signal.
alpha_signal = ef.BandSignal.from_epochs(epochs, alpha, recording="sub-01", pad_sec=0.5)
erds = ef.erds_mean([alpha_signal], baseline=baseline, windows=[stimulus], groups=rois)
bursts = ef.burst_rate([alpha_signal], baseline=baseline, windows=[stimulus], groups=rois)

features = ef.concat([power, peak, erds, bursts])
print(features.to_dataframe())
```

`estimator_parameters` is not bookkeeping: it goes into each column's parameter hash, along
with the frequency axis and sampling rate read off the object itself. Two tables whose grids,
ranges or rates differ therefore cannot land on the same column name.

The declared half of that is on you. MNE keeps none of the estimator's own keyword arguments
on a `Spectrum` — there is no `n_per_seg`, `n_overlap` or `window` to recover — so a setting
you vary without recording it here is a setting the hash cannot see. Two subjects computed
with different Welch segment lengths but the same declaration will stack into one variable,
with the processing difference left inside it. Record every parameter you varied.

### Choosing a container

| Container | Use for |
| :--- | :--- |
| `Spectra.from_spectrum` | PSD-based power and spectral descriptors |
| `Spectra.from_tfr` | An uncorrected MNE `EpochsTFR`; pass the original `n_cycles` so wavelet support can be checked, and the original `sfreq` so Morlet power is scaled to a V²/Hz density independent of sampling rate |
| `Signal.from_epochs` | Broadband time-domain measures |
| `BandSignal.from_epochs` | Band envelopes, phase, power, bursts, ERDS, and band-limited time-domain measures |

For multitaper PSD, compute with `normalization="full"` and record the same setting in
`estimator_parameters`.

### Tables and files

```python
from eegfeat.io import read_dataset, read_table, write_table

rows = epochs.metadata  # None is valid; otherwise one row per epoch.
write_table(features, "sub-01_features.tsv", rows=rows)
restored = read_table("sub-01_features.tsv")

dataset = read_dataset(["sub-01_features.tsv", "sub-02_features.tsv"])
```

`write_table` writes three files: the values TSV, a `_coverage.tsv` matrix giving the finite
fraction behind each cell, and a JSON sidecar with column metadata, flags, row identities, and
provenance. `read_dataset` loads per-epoch bundles into `dataset.table`, with descriptor columns
in `dataset.targets`.

Put target, subject, and run columns in `rows` if the tables will be modeled; they carry through
to `dataset.targets`. For in-memory cohorts with different channel sets, use
`stack_rows(..., columns="union")`.

## Batch runner

The runner reads FIF epochs files recursively and mirrors their paths below the output root.
Start from a generated recipe:

```bash
eegfeat init recipe.toml   # write a commented starting recipe
eegfeat check recipe.toml  # validate, survey recordings, try the first one, write nothing
eegfeat run recipe.toml    # compute features for every recording
```

`python -m eegfeat` is equivalent. Always `check` before a long batch job.

<details>
<summary>Minimal recipe</summary>

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

</details>

Each `[[features]]` entry names one eegfeat measure; its other keys are that function's own
parameters. The rules that matter most:

- Paths resolve against the recipe file. Unknown sections and keys are errors, not warnings.
- Bands are `[low, high)` in Hz; omit `[bands]` for delta through gamma (1–45 Hz).
- Windows are seconds from the epoch origin; omit `[windows]` to measure the whole epoch. The
  name `all` is reserved.
- `baseline` is consumed by normalization, burst, and ERDS measures. It is not also an analysis
  window unless the measure permits that combination.
- `spatial` accepts `channels`, `rois`, and, where supported, `global`.
- `[spectra]` selects `welch` (default), `multitaper`, or `morlet` estimation.
- `[trials]` groups epochs for cross-trial measures by `all`, `event`, or `metadata`. Those
  measures (`itpc`, `ppc`, `envelope_correlation`, `wpli`) need enough epochs per group and
  write group-row tables.
- `series = ["broadband", "alpha"]` selects raw and band-envelope time-domain inputs; `pairs`
  configures PAC; `ratios` and `asymmetry` configure derived power features.

**Options and exit codes.**

```bash
eegfeat run recipe.toml --overwrite --n-jobs 4
eegfeat run recipe.toml --progress-json
```

`--overwrite` replaces an existing result bundle; without it, existing results stop the run.
`--n-jobs` (default 1) goes to MNE filtering and spectral estimation. `--progress-json` emits
one JSON event per line, for a front end.

| Exit code | Meaning |
| :--- | :--- |
| `0` | All recordings succeeded |
| `1` | The run completed, but some recordings failed |
| `2` | The recipe or inputs prevented the run from starting |

**Output.** Per-epoch measures go to `*_features.tsv`, cross-trial measures to a separate
`*_crosstrial.tsv` bundle. Each bundle gets its coverage matrix and JSON sidecar, plus one
`eegfeat_run.json` summary per run. Only FIF epochs files are read.

## Modeling

Needs the `model` extra and per-epoch bundles only — cross-trial tables stay out of the design,
because their rows are trial groups, not independent epochs. Targets must carry matching
`recording`, `epoch`, and `event` keys plus the target and grouping columns.

```python
from pathlib import Path

import numpy as np
import eegfeat.model as efm
from eegfeat.io import read_dataset

dataset = read_dataset(sorted(Path("derivatives/eegfeat").rglob("*_features.tsv")))

# Selection filters on the measure *label* in the sidecar ("band_power"),
# not the function name (integrated_band_power).
selection = efm.Selection(measure=("band_power", "erds_mean"), band=("alpha", "beta"))
design = efm.build_design(
    dataset.table,
    dataset.targets,
    target="rating",
    groups="subject",
    runs="run",
    selection=selection,
)

config = efm.PreprocessingConfig(max_feature_missingness=0.2, max_subject_missingness=0.5)
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

y_true, y_pred, eval_groups, _, _ = efm.fold_results(predictions, groups=design.groups)
metrics, _ = efm.regression_metrics(y_true, y_pred, groups=np.asarray(eval_groups, dtype=object))
print(metrics["subject_level_r"])
```

`loso_folds` needs at least two unique groups, so this expects a multi-recording cohort; the
checked-in [`examples/`](examples/) outputs cover one recording. Use `within_subject_folds` for
run-disjoint within-subject evaluation.

Preprocessing and hyperparameter tuning are fitted inside the training folds. Regression
pipelines include ridge, elastic net, and random forest; classification includes logistic, SVM,
random forest, and ensemble pipelines with binary labels. `permutation_test` refits the whole
cross-fitting procedure; SHAP importance needs the `importance` extra.

## Measures

All of these are exported from `eegfeat` and documented in the
[API reference](https://joshuaduq.github.io/EEGFeat/api/index.html).

| Input or purpose | Functions | Row semantics |
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

`spectral_connectivity` and `wpli` require the `connectivity` extra; microstate measures require
`microstates`.

> [!IMPORTANT]
> `csp_features` gives descriptive held-out features and **cannot** serve as a fixed classifier
> design, even with the same folds. To predict, fit `CommonSpatialPattern` inside each training
> fold, transform both train and test with that same fit, and repeat inside inner tuning.
> `build_design` rejects assembled CSP columns.

## Design guarantees

- Bands outside an input recording's passband raise; partially truncated bands warn and record
  coverage.
- Morlet windows keep only coefficients with complete temporal support.
- ERDS and burst baselines are explicit; they are never learned from the analysis window.
- Non-finite samples become coverage and flags, not silently dropped temporal neighbours.
- Cross-trial tables are never broadcast into per-epoch modeling data.
- Subject-level metrics weight subjects equally by default.

## Documentation

- [Concepts](https://joshuaduq.github.io/EEGFeat/concepts.html) — containers, table anatomy, column names, missing values
- [Quickstart](https://joshuaduq.github.io/EEGFeat/quickstart.html) — containers, measures, and table I/O
- [Feature tables and files](https://joshuaduq.github.io/EEGFeat/guides/tables.html) — querying, TSV/JSON I/O, cohort stacking
- [Runner guide](https://joshuaduq.github.io/EEGFeat/guides/runner.html) — complete TOML schema and CLI behavior
- [Modeling guide](https://joshuaduq.github.io/EEGFeat/guides/modeling.html) — designs, cross-fitting, metrics, nulls, importance
- [Methods](https://joshuaduq.github.io/EEGFeat/methods/index.html) — definitions and assumptions
- [API reference](https://joshuaduq.github.io/EEGFeat/api/index.html) — public signatures
- [Example output](https://joshuaduq.github.io/EEGFeat/examples.html) — real files from a simulated cohort
- [Validation](https://joshuaduq.github.io/EEGFeat/guides/validation.html) — known effects recovered from public MNE datasets

## Development

```bash
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy
```

The validation suite checks the library against public MNE datasets (PhysioNet motor
movement, SSVEP, ERP CORE, Sleep-EDF): known physiology has to come out of real
recordings, and the formulas have to agree with direct computation on them. It downloads
about 350 MB on first use and is skipped unless asked for:

```bash
EEGFEAT_DATASETS=1 python -m pytest tests/validation -ra
```

## License

MIT License. See [LICENSE](LICENSE).
