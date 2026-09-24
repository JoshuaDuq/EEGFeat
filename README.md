# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeat/)

EEG feature extraction and grouped predictive modeling for preprocessed MNE data.

A result is a `FeatureTable`. Each column keeps its band, window, spatial unit, normalization, coverage, and computation parameters. The column name carries the same fields.

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

| Goal | Section |
| :--- | :--- |
| Call the library from Python | [Python API](#python-api) |
| Clean raw EEG into epochs first | [Preprocessing](#preprocessing) |
| Apply one recipe to many epochs files | [Batch runner](#batch-runner) |
| Predict a target from per-epoch features | [Modeling](#modeling) |
| Look up a measure | [Measures](#measures) |
| See written files | [`examples/`](examples/) |

## Install

Python 3.11 or newer. Feature extraction reads preprocessed MNE `Epochs`. The package is not on PyPI.

```bash
git clone https://github.com/JoshuaDuq/EEGFeat.git
cd EEGFeat
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[model]"
```

Core dependencies are `numpy`, `scipy`, `pandas`, and `mne`.

| Extra | Adds |
| :--- | :--- |
| `model` | scikit-learn pipelines, grouped cross-fitting, metrics, nulls, uncertainty |
| `connectivity` | `mne-connectivity` for spectral connectivity and wPLI |
| `microstates` | scikit-learn microstate segmentation |
| `importance` | SHAP. Permutation importance is included in `model` |
| `preprocessing` | raw-to-epochs workflow for one recording or a cohort: PyYAML, scikit-learn, h5io, h5py, filelock |
| `preprocessing-auto` | PyPREP, ICLabel, Picard, and autoreject |
| `preprocessing-gui` | MNE Qt viewers for interactive review |
| `bids` | This study's raw-to-BIDS script only. Not used by `eegfeat` |
| `docs` | This documentation, built with Sphinx |
| `dev` | tests, typing, lint, format |

Development install:

```bash
python -m pip install -e ".[dev,model,connectivity,microstates,importance,preprocessing,preprocessing-auto,docs]"
```

## Python API

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_task-rest_epo.fif", preload=True)

alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
baseline = ef.Window("baseline", -0.5, -0.1)
stimulus = ef.Window("stimulus", 0.1, 0.6)
rois = {"central": ["C3", "Cz", "C4"]}

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

alpha_signal = ef.BandSignal.from_epochs(epochs, alpha, recording="sub-01", pad_sec=0.5)
erds = ef.erds_mean([alpha_signal], baseline=baseline, windows=[stimulus], groups=rois)
bursts = ef.burst_rate([alpha_signal], baseline=baseline, windows=[stimulus], groups=rois)

features = ef.concat([power, peak, erds, bursts])
print(features.to_dataframe())
```

`estimator_parameters` is part of the column hash, together with the frequency axis and sampling rate read from the object. A different grid, range, or sampling rate produces a different column name.

MNE does not store the estimator keyword arguments on a `Spectrum`. `n_per_seg`, `n_overlap`, and `window` cannot be read back from it. Record every parameter that varies. Two recordings computed with different Welch segment lengths and the same `estimator_parameters` share a column name.

### Containers

| Constructor | Input |
| :--- | :--- |
| `Spectra.from_spectrum` | PSD from Welch or multitaper. Power and spectral descriptors. |
| `Spectra.from_tfr` | Morlet `EpochsTFR`. Pass the original `n_cycles` and the `sfreq` from before decimation. |
| `Signal.from_epochs` | Broadband time series. |
| `BandSignal.from_epochs` | Band-pass and Hilbert transform. Envelopes, phase, bursts, ERDS, band-limited time-domain measures. |

`n_cycles` is used to drop Morlet coefficients whose wavelet support falls outside the window. `sfreq` scales Morlet power to a density in V²/Hz.

For a multitaper PSD, call `compute_psd` with `normalization="full"` and store that setting in `estimator_parameters`. Other multitaper normalizations are rejected.

### Files

```python
from eegfeat.io import read_dataset, read_table, write_table

rows = epochs.metadata  # None, or one row per epoch.
write_table(features, "sub-01_features.tsv", rows=rows)
restored = read_table("sub-01_features.tsv")

dataset = read_dataset(["sub-01_features.tsv", "sub-02_features.tsv"])
```

`write_table` writes three files.

| File | Contents |
| :--- | :--- |
| `*_features.tsv` | Values |
| `*_features_coverage.tsv` | Fraction of finite input behind each cell |
| `*_features.json` | Column metadata, flags, row identity, provenance |

`read_dataset` returns per-epoch features in `dataset.table` and descriptor columns in `dataset.targets`. Put the target, subject, and run in `rows` when the tables will be modeled. For in-memory tables whose channel sets differ, use `stack_rows(..., columns="union")`.

## Preprocessing

Optional. One YAML recipe, one recording or a whole cohort. The feature runner reads the exported FIF and does not call this package.

```bash
python -m pip install -e ".[preprocessing]"
eegfeat preprocess init preprocessing.yaml --mode events
```

`--mode resting` writes 2 s fixed-length epochs instead of an event block. `init` will not replace an existing file. Edit the recipe before `check`.

- `input.path` names one recording. `input.root` and `input.pattern` select a cohort instead; the tree below `root` is mirrored under `output.directory` and each recording is named after its file.
- `epochs.events` must match the recordings. `init` writes annotation events `{stimulus: 1}`.
- `workflow` sets the review policy per gate. `required` stops the run until a decision is saved. `suggested` saves the detectors' own verdict and continues. `raw_review: disabled` skips the raw gate, `epoch_review` is skipped unless `required`, and an artifact gate runs whenever `artifact` is set.
- A FIF with inactive projectors fails while `channels.projections` is `error`. Set `apply` or `discard-inactive`.

```bash
eegfeat preprocess check preprocessing.yaml   # every recording; writes nothing
eegfeat preprocess run preprocessing.yaml     # every recording, up to export
```

With `raw_review: required`, `run` stops at the raw gate, prints the command to continue with, and exits 3:

```text
[1/2] sub-01
      ✓ awaiting review-raw · eegfeat preprocess review preprocessing.yaml --recording sub-01 raw
```

With a display, `pip install -e ".[preprocessing-gui]"` and run that command. It opens the Qt browser and saves the decision when the dialog is accepted.

Without a display, fill in the pending file the run wrote, `<bundle directory>/.preprocessing/<name>/decisions/review-raw.pending.yaml`. Every field is explained in a comment; replace each `null`. Then run the same `review` command, which reads the filled file. `eegfeat preprocess review preprocessing.yaml raw` with no `--recording` reviews each recording that awaits it.

```bash
eegfeat preprocess review preprocessing.yaml raw
eegfeat preprocess run preprocessing.yaml
```

That second `run` exports. The init recipe has no artifact block, so there is no second gate. An `artifact` block adds `review-artifact`, governed by `artifact_review`. [`examples/preprocessing.yaml`](examples/preprocessing.yaml) is a cohort recipe with ICA. Both of its gates are `suggested`, so that recipe runs to export without a stop.

Each bundle is `<name>_epo.fif`, `<name>_events.tsv`, `<name>_report.html`, `<name>_preprocessing.json`, and, when autoreject ran, `<name>_repairs.tsv`. When every recording is exported, `run` prints the `eegfeat init` command and the `inputs.root` to set. `status` shows one line per recording and the next command to run. `--recording LABEL` limits a command to one recording.

| Exit code | Meaning |
| :--- | :--- |
| `0` | Every selected recording reached the requested stage |
| `1` | At least one recording failed; the others still ran |
| `2` | The recipe or a prerequisite is wrong |
| `3` | No recording failed and at least one awaits a review |

[`tui/`](tui) is an optional terminal front end (Go 1.24+). Build it with `cd tui && go build -o eegfeat-tui .`. It lists recordings, starts `run`, and shows each gate as a checklist. `Tab` moves between recordings and stages, `Enter` runs the selected action, and `?` lists the keys. `l` opens the log. The Python package does not import it. Keys and the JSON commands it calls are in the [preprocessing guide](https://joshuaduq.github.io/EEGFeat/guides/preprocessing.html).

[`paradigm_specific/thermal_pain/eeg_raw_to_bids.py`](paradigm_specific/thermal_pain/eeg_raw_to_bids.py) converts this study's BrainVision or cleaned FIF to BIDS. It needs the `bids` extra. The command and the marker rules are in the script. `repair_markers.py` beside it writes those marker names back into files converted before this script.

The schema, the other review fields, and the stage order are in the [preprocessing guide](https://joshuaduq.github.io/EEGFeat/guides/preprocessing.html).

## Batch runner

The runner reads FIF epochs files recursively and mirrors that tree under the output root. `python -m eegfeat` is the same entry point. Run `check` before a long job. It writes nothing.

```bash
eegfeat init recipe.toml   # commented recipe; --template task or resting for every measure family
eegfeat check recipe.toml  # load the recipe and run the first recording
eegfeat run recipe.toml    # every recording
eegfeat status recipe.toml # done, missing, failed, stale or partial, and what to run next
```

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

Each `[[features]]` entry sets `measure` to one function name, or `measures` to a list of them sharing the entry's other keys. The other keys are that function's parameters. A `[defaults]` section sets `bands`, `windows`, `spatial`, or `series` for every entry that takes them.

| Item | Rule |
| :--- | :--- |
| Paths | Resolved against the recipe file. Unknown sections and keys are errors. |
| Bands | `[low, high)` in Hz. If `[bands]` is omitted, the bands are delta through gamma (1–45 Hz). |
| Windows | Seconds from the epoch origin. If `[windows]` is omitted, the measure uses the whole epoch. The name `all` is reserved. |
| `baseline` | Consumed by normalization, bursts, and ERDS. It is an analysis window only for measures that allow that combination. |
| `spatial` | `channels`, `rois`, and, where the measure supports it, `global`. |
| `[spectra]` | `welch` (default), `multitaper`, or `morlet`. |
| `[trials]` | Groups for cross-trial measures. `all`, `event`, or `metadata`. |
| Cross-trial measures | `itpc`, `ppc`, `envelope_correlation`, `wpli`. They need enough epochs per group and write group-row tables. |
| `series` | Time-domain and complexity input. `broadband` and band names (envelopes). |
| `pairs` | PAC only. `[["theta", "gamma"]]` is `[phase, amplitude]`. |
| `ratios`, `asymmetry` | Derived power features on a power measure. |

```bash
eegfeat run recipe.toml --overwrite --n-jobs 4
eegfeat run recipe.toml --resume
eegfeat run recipe.toml --workers 6
eegfeat run recipe.toml --progress-json
eegfeat status recipe.toml --json
```

| Flag | Effect |
| :--- | :--- |
| `--overwrite` | Replace an existing result bundle. Without it, an existing result stops the run. |
| `--resume` | Compute only the recordings `status` does not call `done`. Stale or partial results still need `--overwrite`. |
| `--workers` | Recordings computed at once, each in its own process. Default 1. |
| `--n-jobs` | Passed to MNE filtering and spectral estimation. Default 1. |
| `--progress-json` | One JSON object per line. |

| Exit code | Meaning |
| :--- | :--- |
| `0` | Every recording succeeded |
| `1` | The run finished, and at least one recording failed |
| `2` | The recipe, the inputs, or earlier results in the way stopped the run before computation |

Per-epoch measures are written to `*_features.tsv`. Cross-trial measures are written to `*_crosstrial.tsv`. Each bundle has a coverage TSV and a JSON sidecar. One `eegfeat_run.json` summarizes the run. Only FIF epochs files are read.

## Modeling

Requires the `model` extra. `build_design` takes per-epoch tables. Cross-trial rows are trial groups and are not part of the design. The target frame needs `recording`, `epoch`, and `event`, plus the target column and the grouping column.

```python
from pathlib import Path

import numpy as np
import eegfeat.model as efm
from eegfeat.io import read_dataset

dataset = read_dataset(sorted(Path("derivatives/eegfeat").rglob("*_features.tsv")))

# "band_power" is the label stored on the column. The function name is integrated_band_power.
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
    efm.ridge_grid(design.X),
    inner=efm.InnerSplit(grouping="subject", n_splits=5),
    seed=42,
    runs=design.runs,
    harmonization="intersection",
)

y_true, y_pred, eval_groups, _, _ = efm.fold_results(predictions, groups=design.groups)
metrics, _ = efm.regression_metrics(y_true, y_pred, groups=np.asarray(eval_groups, dtype=object))
print(metrics["subject_level_r"])
```

`loso_folds` needs at least two groups. The files in [`examples/`](examples/) are one recording. `within_subject_folds` holds out runs within each subject.

Preprocessing and hyperparameter tuning are fit on the training rows of each fold. Regression pipelines are ridge, elastic net, and random forest. Classification pipelines are logistic regression, SVM, random forest, and an ensemble, with labels in `{0, 1}`. `permutation_test` gives the null of the full procedure: refitted on every draw, or computed in closed form for a ridge pipeline. Held-out scores from different subjects share training data, so no interval is reported over them. SHAP requires the `importance` extra.

## Measures

Exported from `eegfeat`. Definitions and signatures are in the [API reference](https://joshuaduq.github.io/EEGFeat/api/index.html).

| Family | Functions | Rows |
| :--- | :--- | :--- |
| PSD and TFR power | `mean_psd`, `integrated_band_power`, `mean_tfr_power` | Per epoch |
| Spectral shape | `peak_frequency`, `spectral_centroid`, `spectral_bandwidth`, `spectral_edge`, `spectral_entropy`, `aperiodic`, `aperiodic_ratio` | Per epoch |
| Time domain | `variance`, `mean_amplitude`, `peak_to_peak`, `area_under_curve`, `peak_amplitude`, `peak_latency`, `amplitude_quantile`, `kurtosis`, `line_length`, `root_mean_square`, `skewness`, `zero_crossing_rate`, `hjorth_mobility`, `hjorth_complexity` | Per epoch |
| Bursts and ERDS | `burst_count`, `burst_rate`, `burst_duration`, `burst_amplitude`, `fraction_above_threshold`, `erds_mean`, `erds_slope`, `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration`, `erds_onset_latency`, `erds_peak_latency`, `erds_rebound_latency` | Per epoch |
| Cross-trial phase and connectivity | `itpc`, `ppc`, `envelope_correlation`, `spectral_connectivity`, `wpli` | One row per trial group |
| Phase-amplitude coupling | `pac` | Per epoch. No surrogate correction. |
| Derived power | `band_ratio`, `asymmetry` | Per epoch |
| Graph summaries | `global_efficiency`, `clustering_coefficient` | One value from a pairwise table |
| Complexity | `sample_entropy`, `multiscale_entropy`, `higuchi_fractal_dimension` | Per epoch |
| Microstates | `segment`, `microstate_coverage`, `microstate_duration`, `microstate_occurrence`, `microstate_transitions` | Per epoch |
| Supervised spatial filters | `CommonSpatialPattern`, `csp_features` | Per epoch, fit inside each training fold |

`spectral_connectivity` and `wpli` require the `connectivity` extra. Microstate measures require `microstates`.

`csp_features` fits inside each training fold and returns that fold's held-out rows. `build_design` rejects those columns. For classification, fit `CommonSpatialPattern` on the training rows of each fold, transform the training and test rows with that fit, and repeat the fit inside inner tuning. An assembled CSP table still leaks under the same folds, because each fit uses labels from epochs that another fold holds out.

## Behavior

- A band outside the recording passband raises. A band that is only partly inside the passband warns, and coverage records the portion used.
- A Morlet window average uses coefficients whose wavelet support lies inside the window.
- The ERDS and burst baseline is the window passed by the caller.
- Non-finite samples are omitted from the statistic and recorded in `coverage` and `flags`. Samples on either side of a gap do not become neighbours.
- A cross-trial value is not copied onto the epochs that formed it.
- Subject-level metrics give each subject equal weight by default.

## Documentation

| Page | Contents |
| :--- | :--- |
| [Concepts](https://joshuaduq.github.io/EEGFeat/concepts.html) | Containers, table fields, column names, missing values |
| [Quickstart](https://joshuaduq.github.io/EEGFeat/quickstart.html) | PSD, Morlet, and ERDS examples |
| [Feature tables and files](https://joshuaduq.github.io/EEGFeat/guides/tables.html) | `select`, TSV and JSON, stacking |
| [Preprocessing](https://joshuaduq.github.io/EEGFeat/guides/preprocessing.html) | YAML schema, review, and export |
| [Runner](https://joshuaduq.github.io/EEGFeat/guides/runner.html) | TOML schema and CLI |
| [Modeling](https://joshuaduq.github.io/EEGFeat/guides/modeling.html) | Designs, cross-fitting, metrics, nulls, importance |
| [Methods](https://joshuaduq.github.io/EEGFeat/methods/index.html) | Definitions |
| [API reference](https://joshuaduq.github.io/EEGFeat/api/index.html) | Signatures |
| [Example output](https://joshuaduq.github.io/EEGFeat/examples.html) | Files from a simulated cohort |
| [Validation](https://joshuaduq.github.io/EEGFeat/guides/validation.html) | Known effects on public MNE datasets |

## Development

```bash
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy
```

Beyond the unit tests, a validation suite runs every public function on four public
datasets (PhysioNet motor movement, MNE's SSVEP example, ERP CORE, Sleep-EDF) and checks
three things: that each formula matches an independent computation on real data, that
different estimators of the same quantity agree, and that textbook effects come out. The
[scorecard](https://joshuaduq.github.io/EEGFeat/guides/validation.html) is written by the
suite itself on each run. It downloads about 350 MB on first use and is skipped unless asked
for:

```bash
EEGFEAT_DATASETS=1 python -m pytest tests/validation -ra
```

## License

MIT. See [LICENSE](LICENSE).
