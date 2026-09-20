# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Typing: Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeatML/)

**Labelled feature extraction and leak-free machine learning for preprocessed EEG.**

EEGFeat operates downstream of preprocessing, accepting epoched MNE objects (`Epochs`, `Spectrum`, `EpochsSpectrum`, `EpochsTFR`). It extracts fully auditable, metadata-tracked features and provides a rigorous cross-validation modeling pipeline designed to prevent data leakage.

### Core Features

- **Self-Describing Features**: Every column preserves its measure type, frequency band, ROI/channel, time window, normalization method, physical units, and estimator parameters.
- **Strict Separation**: Epoch-level and cross-trial measures (e.g., ITPC, wPLI) are maintained in separate tables to prevent pseudo-replication.
- **Leak-Free Modeling**: Imputation, scaling, selection, tuning, and harmonization are fitted strictly inside training folds.
- **Subject-Aware Metrics**: Primary evaluation metrics weight subjects equally, avoiding trial-count dominance.
- **Example Pipeline**: Inspect sample outputs generated from synthetic data in [`examples/`](examples/).

---

## Installation

```bash
pip install eegfeat
```

### Optional Extras

| Extra | Description | Dependencies |
| :--- | :--- | :--- |
| `[model]` | Cross-validated modeling pipelines and evaluation | `scikit-learn>=1.3` |
| `[connectivity]` | Weighted Phase Lag Index (wPLI) | `mne-connectivity>=0.7` |
| `[microstates]` | Microstate segmentation and dynamics | `scikit-learn>=1.3` |
| `[importance]` | SHAP importance over cross-validation folds | `shap>=0.45`, `scikit-learn>=1.3` |
| `[dev]` | Testing, typing, and linting tools | `pytest`, `mypy`, `ruff`, `black` |

```bash
pip install "eegfeat[model,connectivity,microstates,importance]"
```

---

## Quickstart: Feature Extraction

### 1. Define Measurement Containers

Define frequency bands, time windows, and regions of interest (ROIs):

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_task-pain_epo.fif", preload=True)

# Define bands, windows, and ROIs
alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
baseline = ef.Window("baseline", -5.0, -1.0)
stimulus = ef.Window("stimulus", 0.0, 8.0)
rois = {"central": ["C3", "Cz", "C4"], "parietal": ["P3", "Pz", "P4"]}
```

### 2. Extract Spectral and Time-Domain Measures

- **Spectral measures** take an explicit MNE `Spectrum` or `EpochsSpectrum`:

```python
spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, tmin=0.0, tmax=8.0)
spectra = ef.Spectra.from_spectrum(
    spectrum,
    recording="sub-01",
    estimator_parameters={"method": "welch", "fmin": 1.0, "fmax": 45.0},
)

power = ef.integrated_band_power(spectra, bands=[alpha, beta], groups=rois, normalize="log10")
peak = ef.peak_frequency(spectra, band=alpha, groups=rois)
```

- **Time-domain, burst, and ERDS measures** cut designated windows directly from analytic or broadband signals:

```python
alpha_signal = ef.BandSignal.from_epochs(epochs, band=alpha, recording="sub-01")

# ERDS and burst rates calibrated against the baseline window
erds = ef.erds_mean([alpha_signal], windows=[stimulus], baseline=baseline, groups=rois)
bursts = ef.burst_rate([alpha_signal], windows=[stimulus], baseline=baseline, groups=rois)
```

### 3. Combine and Export Feature Tables

```python
table = ef.concat([power, peak, erds, bursts])

# Introspect table metadata
table.n_rows          # One row per epoch
table.names           # Column names containing full parameter hashes
table.meta[0].unit    # Physical unit (e.g. 'log10(V^2)')
table.meta[0].window  # Time window label (e.g. 'all' or 'stimulus')

# Query columns semantically
subset = table.select(band=alpha, space="central")
df = table.to_dataframe()
```

Export tables to disk with complete provenance:

```python
from eegfeat.io import read_table, write_table

# Write features alongside trial metadata
write_table(table, "sub-01_features.tsv", rows=epochs.metadata)
restored = read_table("sub-01_features.tsv")
```

`write_table` generates three complementary files:
1. `*_features.tsv`: Numerical values with trial metadata.
2. `*_features_coverage.tsv`: Fractional coverage indicating finite, unmasked data per cell.
3. `*_features.json`: Sidecar storing metadata, physical units, parameter hashes, and provenance for every column.

---

## Available Measures

| Category | Functions | Description |
| :--- | :--- | :--- |
| **Spectral Power** | `integrated_band_power`, `mean_psd`, `mean_tfr_power`, `band_ratio`, `asymmetry` | PSD/TFR band power with normalization (`raw`, `log10`, `dB`, `percent`, `log_ratio`), frequency ratios, and hemispheric asymmetry. |
| **Spectral Shape** | `peak_frequency`, `spectral_centroid`, `spectral_bandwidth`, `spectral_edge`, `spectral_entropy`, `aperiodic` | 1/f-corrected peak frequency with sub-bin refinement, spectral distribution moments, edge frequencies, entropy, and aperiodic slope/offset. |
| **Time Domain** | `variance`, `mean_amplitude`, `peak_to_peak`, `area_under_curve`, `peak_amplitude`, `peak_latency` | Statistical and waveform morphology metrics on broadband or filtered analytic envelope signals. |
| **Oscillatory Bursts** | `burst_rate`, `burst_count`, `burst_duration`, `burst_amplitude`, `fraction_above_threshold` | Dual-threshold burst detection calibrated against pre-stimulus baseline periods. |
| **Event-Related Dynamics** | `erds_mean`, `erds_slope`, `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration`, `erds_onset_latency`, `erds_peak_latency`, `erds_rebound_latency` | Temporal ERD/ERS dynamics, peak/onset/rebound latencies, and regression slopes relative to baseline. |
| **Phase & Connectivity** | `itpc`, `ppc`, `pac`, `envelope_correlation`, `wpli`, `global_efficiency`, `clustering_coefficient` | Cross-trial phase consistency, phase-amplitude coupling, pairwise connectivity, and graph theoretical network metrics. *(Written to cross-trial tables)* |
| **Complexity** | `sample_entropy`, `multiscale_entropy` | Signal regularity across multiple temporal scales. |
| **Microstates** | `segment`, `microstate_coverage`, `microstate_duration`, `microstate_occurrence`, `microstate_transitions` | Topographic clustering at GFP peaks and subsequent temporal dynamics analysis. |

---

## Batch Processing: `eegfeat` CLI

The CLI automates uniform feature extraction across full BIDS or derivative datasets using a single TOML recipe.

```bash
eegfeat init recipe.toml     # Generate a commented recipe template
eegfeat check recipe.toml    # Validate syntax and test extraction on the first recording
eegfeat run recipe.toml      # Process all recordings across the dataset
```

### Recipe Structure (`recipe.toml`)

```toml
[inputs]
root = "derivatives/preprocessed/eeg"
pattern = "**/*_proc-clean_epo.fif"
picks = "eeg"
exclude_bads = true

[output]
root = "derivatives/eegfeat"
epoch_metadata = true          # Include epoch metadata columns in outputs

[bands]
theta = [4.0, 8.0]
alpha = [8.0, 13.0]
beta = [13.0, 30.0]

[windows]
baseline = [-5.0, -1.0]
stimulus = [0.0, 8.0]

[rois]
central = ["C3", "Cz", "C4"]
parietal = ["P3", "Pz", "P4"]

[[features]]
measure = "integrated_band_power"
normalize = "log10"
spatial = ["channels", "rois", "global"]
ratios = [["theta", "beta"]]

[[features]]
measure = "erds_mean"
bands = ["alpha", "beta"]
baseline = "baseline"
spatial = ["rois", "global"]

[[features]]
measure = "itpc"
bands = ["theta"]
```

### Generated Files

For each recording (e.g., `sub-01_task-pain_epo.fif`), `eegfeat run` writes:
- `*_features.tsv`: Per-epoch feature table.
- `*_features_coverage.tsv`: Per-epoch finite data coverage table.
- `*_features.json`: Column metadata sidecar.
- `*_crosstrial.tsv`: Cross-trial measures (e.g., ITPC, wPLI) kept strictly separate from per-epoch rows.
- `eegfeat_run.json`: Execution log and reproducibility record for the batch.

---

## Machine Learning (`eegfeat.model`)

The `eegfeat.model` subpackage provides leak-free predictive modeling and validation tools.

### 1. Build the Design Matrix

Load multi-subject feature tables into unified matrices matched by metadata identifiers (`recording`, `epoch`, `event`):

```python
from pathlib import Path
import eegfeat.model as efm
from eegfeat.io import read_dataset

dataset = read_dataset(sorted(Path("derivatives/eegfeat").rglob("*_features.tsv")))

# Filter features semantically by measure and band
selection = efm.Selection(measure=("band_power", "erds_mean"), band=("alpha", "beta"))

design = efm.build_design(
    dataset.table,
    dataset.targets,
    target="pain_rating",
    groups="subject",
    runs="run",
    selection=selection,
)

# design.X: Feature matrix (epochs x features)
# design.y: Continuous or categorical target vector
# design.groups: Subject identifiers for grouped splitting
# design.runs: Run/session identifiers
```

### 2. Define Split Strategy and Cross-Fit

Choose an evaluation scheme matching your generalization target:
- `efm.loso_folds(design.groups)`: Leave-one-subject-out CV (between-subject generalization).
- `efm.within_subject_folds(design.groups, design.runs, inner_splits=3, seed=42)`: Within-subject CV across runs.

Preprocessing (imputation, scaling, selection) and hyperparameter tuning execute strictly within each training fold:

```python
config = efm.PreprocessingConfig(
    max_feature_missingness=0.2,
    feature_selection_percentile=25.0,
)
pipeline = efm.ridge_pipeline(config, seed=42)
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
```

*(Pipelines are also provided for `elasticnet_pipeline` and `random_forest_pipeline` with matching `*_grid()` definitions.)*

### 3. Evaluate Performance

Compute subject-weighted metrics (averaging within-subject Fisher $z$-transformed correlations) to prevent trial imbalance bias:

```python
import numpy as np
import pandas as pd

y_true, y_pred, eval_groups, _, _ = efm.fold_results(predictions, groups=design.groups)
eval_groups = np.asarray(eval_groups, dtype=object)

metrics, per_subject = efm.regression_metrics(y_true, y_pred, eval_groups)
print(f"Subject-level r: {metrics['subject_level_r']:+.3f}")
print(f"Pooled r:        {metrics['pearson_r']:+.3f}")

# Calculate 95% confidence interval
summary = efm.subject_level_r(
    pd.DataFrame({"subject_id": eval_groups, "y_true": y_true, "y_pred": y_pred})
)
print(f"95% CI: [{summary.ci_low:+.3f}, {summary.ci_high:+.3f}]")
```

### 4. Permutation Null Testing

Test significance by refitting the cross-validation loop on within-subject or within-run shuffled labels:

```python
null = efm.permutation_test(
    folds,
    design.X,
    design.y,
    design.groups,
    design.runs,
    pipeline,
    efm.ridge_grid(),
    metrics["subject_level_r"],  # Observed statistic
    config=efm.NullConfig(scheme="within_subject", n_permutations=1000),
    inner=inner,
    seed=42,
    harmonization="intersection",
)
print(f"Permutation p-value: {null.p_value:.3f}")
```

### 5. Uncertainty and Feature Importance

- **Prediction Intervals**: Compute trial-calibrated bounds. Formal conformal coverage requires exchangeability; group-disjoint splits alone do not guarantee coverage for dependent EEG trials or new subjects. CV+ uses `alpha` in each tail, so `alpha=0.10` is not a universal 90% coverage guarantee:

```python
intervals = efm.prediction_intervals(
    pipeline,
    design.X,
    design.y,
    design.X[:5],
    alpha=0.10,
    method="cv_plus",  # "cv_plus", "split", or "quantile"
    groups=design.groups,
    seed=42,
)
print(f"Intervals: [{intervals.lower}, {intervals.upper}]")
```

- **Out-of-Fold Feature Importance**: Assess feature attribution on validation folds and aggregate by metadata:

```python
importance = efm.permutation_importance_over_folds(
    folds,
    design.X,
    design.y,
    design.groups,
    pipeline,
    efm.ridge_grid(),
    inner=inner,
    feature_names=design.column_names,
    seed=42,
    runs=design.runs,
    harmonization="intersection",
)

# Aggregate importance scores across semantic tags
selected = efm.select(dataset.table, selection)
band_imp = efm.aggregate_by(importance, selected.meta, "band")
measure_imp = efm.aggregate_by(importance, selected.meta, "measure")
```

### 6. Classification

Binary classification models (`logistic_pipeline`, `svm_pipeline`, `random_forest_classifier_pipeline`, `ensemble_pipeline`) require labels coded as 0 and 1:

```python
labels = efm.build_design(
    dataset.table, dataset.targets, target="painful", groups="subject", runs="run",
    selection=selection,
)
answered = np.isin(labels.y, (0.0, 1.0))

classifications = efm.cross_fit_classification(
    efm.loso_folds(labels.groups[answered]),
    labels.X[answered],
    labels.y[answered].astype(np.intp),
    labels.groups[answered],
    efm.logistic_pipeline(config, seed=42),
    efm.logistic_grid(),
    inner=inner,
    seed=42,
    runs=labels.runs[answered],
    harmonization="intersection",
)

truth, predicted, class_groups, _, _ = efm.fold_results(
    classifications, groups=labels.groups[answered]
)
in_fold_order = sorted(classifications, key=lambda fold: fold.fold)
positive = np.concatenate([f.y_prob[:, f.classes.index(1)] for f in in_fold_order])

result = efm.classification_metrics(
    truth.astype(np.intp),
    predicted.astype(np.intp),
    y_prob=positive,
    groups=np.asarray(class_groups, dtype=object),
)
print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}, AUC: {result.auc:.3f}")
```

---

## Scientific Guardrails

EEGFeat enforces strict computational and statistical constraints by default:

1. **Boundary Isolation**: Morlet wavelets are strictly masked to frequencies whose temporal support fits entirely within the requested window to eliminate edge contamination.
2. **1/f Background Decoupling**: Peak frequency fitting subtracts the fitted aperiodic component across a broad band rather than searching local raw PSD maxima.
3. **Pre-Stimulus Threshold Calibration**: Burst detection thresholds and ERD/ERS baselines must be defined on dedicated reference windows; they cannot self-calibrate on test periods.
4. **Pseudo-Replication Prevention**: Multi-trial metrics (ITPC, wPLI, envelope correlation) are strictly separated into cross-trial tables and rejected by per-epoch modeling routines.
5. **Coverage Tracking**: Per-cell numeric finiteness and valid wavelet support fractions are recorded independently of data values.
6. **Leak-Free Cross-Fitting**: All preprocessing, imputation, scaling, feature selection, and tuning operations are computed exclusively within training folds.
7. **Balanced Subject Weighting**: Cohort metrics weight each subject equally to prevent subjects with higher trial counts from dominating effect sizes.
8. **Preserved Null Distributions**: Permutation tests fail fast if any fold fit fails, preventing biased null shapes from dropped iterations.

---

## Documentation and Development

Full API references, mathematical derivations, and tutorials are available in the [Documentation](https://joshuaduq.github.io/EEGFeatML/):

- [Quick Start](docs/quickstart.rst)
- [Methods & Mathematical Formulations](docs/methods.rst)
- [Modeling & Validation](docs/modeling.rst)
- [Runner & Recipe Specification](docs/runner.rst)
- [API Reference](docs/api.rst)

Run the test and verification suite:

```bash
pytest                  # Unit and regression tests
mypy src                # Strict static type checking
ruff check src tests    # Linting and style conformance
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
