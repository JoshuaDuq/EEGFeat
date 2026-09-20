# <img src="docs/_static/favicon.svg" width="32" height="32" valign="middle" alt="EEGFeat logo" /> EEGFeat

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MNE-Python ≥ 1.8](https://img.shields.io/badge/mne--python-≥1.8-blue.svg)](https://mne.tools/stable/)
[![Typing: Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/EEGFeatML/)

**Turn preprocessed EEG into a labelled feature table, then take that table through machine
learning without leaking the answer.**

Every number EEGFeat produces knows what it is: which measure, which frequency band, which
channel or region, which time window, which normalization, its unit, and the exact parameters
behind it. That description travels with the value — into the files you write, back out when you
read them, and on into the design matrix you model. A column is never just `feature_0347`.

EEGFeat starts where preprocessing ends. You bring epochs you already trust; it does not filter,
re-reference, or reject anything on your behalf.

### See the output before you install anything

Real files from the real pipeline, small enough to open and read, in [`examples/`](examples/):

- [`sub-01_task-pain_run-01_features.tsv`](examples/sub-01_task-pain_run-01_features.tsv) — one
  row per epoch: your trial metadata first, then every feature, each named for what it measures.
  Its [coverage file](examples/sub-01_task-pain_run-01_features_coverage.tsv) and
  [JSON sidecar](examples/sub-01_task-pain_run-01_features.json) sit beside it, and the
  [cross-trial table](examples/sub-01_task-pain_run-01_crosstrial.tsv) holds the measures that are
  defined over a group of trials rather than within one.
- [`example_model_scores.tsv`](examples/example_model_scores.tsv) — what the modelling gives back:
  a leave-one-subject-out correlation with its confidence interval and permutation *p*, plus each
  subject's own score. Every held-out prediction is in
  [`example_model_predictions.tsv`](examples/example_model_predictions.tsv).

The recordings behind them are simulated, so nothing here is participant data, and
`python examples/make_examples.py` reproduces the lot from
[`examples/recipe.toml`](examples/recipe.toml).

---

## Install

```bash
pip install eegfeat
```

The core install computes every spectral, temporal, burst, and ERDS measure. A few measures and
the modeling subpackage need extras:

- `pip install "eegfeat[model]"` — scikit-learn, for everything in the machine learning section.
- `pip install "eegfeat[connectivity]"` — mne-connectivity, for weighted phase lag index.
- `pip install "eegfeat[microstates]"` — scikit-learn, for microstate clustering.
- `pip install "eegfeat[importance]"` — adds SHAP. Permutation importance already comes
  with `[model]`.
- `pip install "eegfeat[dev]"` — pytest, ruff, black, mypy.

---

## Extract features from one recording

Three steps: describe the bands and windows you care about, wrap your MNE objects in EEGFeat
containers, then call measures on them.

```python
import mne
import eegfeat as ef

epochs = mne.read_epochs("sub-01_task-pain_epo.fif", preload=True)

# What you want measured, and where.
alpha = ef.Band("alpha", 8.0, 13.0)
beta = ef.Band("beta", 13.0, 30.0)
baseline = ef.Window("baseline", -5.0, -1.0)
stimulus = ef.Window("stimulus", 0.0, 8.0)
rois = {"central": ["C3", "Cz", "C4"], "parietal": ["P3", "Pz", "P4"]}
```

**Spectral measures read a spectrum you computed yourself.** EEGFeat never guesses how your
spectrum should be estimated — you pass it a finished MNE `Spectrum`, `EpochsSpectrum`, or
`EpochsTFR` and say what produced it.

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

**Time-domain, burst, and ERDS measures read the epochs and cut the windows themselves.** Pass
them a `Signal` (broadband) or a `BandSignal` (one band's analytic signal), and they slice each
window out for you.

```python
alpha_signal = ef.BandSignal.from_epochs(epochs, band=alpha, recording="sub-01")

# Percent change from the baseline window, calibrated on the baseline itself.
erds = ef.erds_mean([alpha_signal], windows=[stimulus], baseline=baseline, groups=rois)
bursts = ef.burst_rate([alpha_signal], windows=[stimulus], baseline=baseline, groups=rois)
```

Join everything into one table and look at it:

```python
table = ef.concat([power, peak, erds, bursts])

table.n_rows          # one row per epoch
table.names           # 'eeg_band-power_alpha_central_all_log10_p83d9…', …
table.meta[0].unit    # 'log10(V^2)'
table.meta[0].window  # 'all'

# Pull columns back out by what they mean, not by string matching.
table.select(band=alpha, space="central")
table.to_dataframe()
```

The spectral columns say `window='all'` because the spectrum you handed over covers one segment;
EEGFeat labels what it was given rather than inventing a window. To get several named windows in
one spectral table, compute one spectrum per window — or let the runner below do it for you.

Write it, and it comes back whole — values, per-cell coverage, flags, and all the metadata:

```python
from eegfeat.io import read_table, write_table

write_table(table, "sub-01_features.tsv", rows=epochs.metadata)
restored = read_table("sub-01_features.tsv")
```

`write_table` produces three files: the values as TSV, a matching `_coverage.tsv` saying which
cells were finite, and a `.json` sidecar holding every column's description. Passing
`rows=epochs.metadata` copies your per-trial variables — condition, stimulus intensity, response,
subject — alongside the features. **Put your prediction target and your subject label in there**,
because that is where the modeling stage will look for them.

---

## What you can measure

Every measure takes bands, windows, and an optional region mapping, and returns a table you can
`concat` with any other. Cross-trial measures return one row per trial group instead of one row
per epoch, and say so.

- **Spectral power** — `integrated_band_power` and `mean_psd` for a PSD, `mean_tfr_power` for a
  time-frequency decomposition. Normalize as raw, `log10`, dB, percent, or a log ratio against a
  baseline window. `band_ratio` and `asymmetry` derive theta/beta style ratios and left/right
  contrasts from a power table you already have.
- **Spectral shape** — `peak_frequency` finds the dominant oscillation after dividing out the
  1/f background, refines it between bins, and flags the cases where the maximum sat on a band
  edge or fell back to a centre of gravity. `spectral_centroid`, `spectral_bandwidth`,
  `spectral_edge`, and `spectral_entropy` describe the band's shape. `aperiodic` returns the 1/f
  slope and offset themselves.
- **Time domain** — `variance`, `mean_amplitude`, `peak_to_peak`, `area_under_curve`,
  `peak_amplitude`, and `peak_latency`, on the broadband signal or on any band's envelope.
- **Oscillatory bursts** — `burst_rate`, `burst_count`, `burst_duration`, `burst_amplitude`, and
  `fraction_above_threshold`. The threshold is calibrated on a baseline window you nominate, so
  the stimulus cannot set its own detection bar.
- **Event-related dynamics** — `erds_mean` and `erds_slope` for the average change and its trend;
  `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration` for the two directions
  separately; `erds_onset_latency`, `erds_peak_latency`, `erds_rebound_latency` for the timing.
- **Phase and connectivity** — `itpc` and `ppc` for phase consistency across trials, `pac` for
  phase-amplitude coupling, `envelope_correlation` and `wpli` for pairwise coupling, and
  `global_efficiency` and `clustering_coefficient` to summarize a connectivity table as a network.
- **Complexity** — `sample_entropy` and `multiscale_entropy`.
- **Microstates** — `segment` clusters the topographies at global field power peaks into states,
  then `microstate_coverage`, `microstate_duration`, `microstate_occurrence`, and
  `microstate_transitions` describe each window's sequence.

---

## Extract features for a whole dataset

Writing the loop yourself gets tedious across dozens of recordings. `eegfeat run` takes one
recipe file and applies it to every epochs file it finds.

```bash
eegfeat init recipe.toml     # a commented recipe to start from
eegfeat check recipe.toml    # validate it, try it on the first recording, write nothing
eegfeat run recipe.toml      # compute features for every recording
```

A recipe says once what you would otherwise repeat: where the epochs are, which bands and windows
and regions you mean, how spectra are estimated, and one entry per measure.

```toml
[inputs]
root = "derivatives/preprocessed/eeg"
pattern = "**/*_proc-clean_epo.fif"
picks = "eeg"
exclude_bads = true

[output]
root = "derivatives/eegfeat"
epoch_metadata = true          # copy each epoch's metadata into its row

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

Run `check` before `run`. It reads the recipe, reports every problem at once rather than the first
one, and then computes the first recording to catch what a recipe alone cannot know — a region
naming a channel your data lacks, a window outside your epochs, a wavelet longer than an epoch.

The output tree mirrors the input tree. For `sub-01/eeg/sub-01_task-pain_epo.fif` you get
`sub-01_task-pain_features.tsv` with one row per epoch, its coverage file and JSON sidecar,
`sub-01_task-pain_crosstrial.tsv` for the measures that are defined across trials rather than
within one, and an `eegfeat_run.json` recording what happened to every recording.

Per-epoch and cross-trial results are kept in separate files on purpose. ITPC, wPLI, and envelope
correlation estimate one value from a group of trials; copying that value onto each of its trials
would invent independent observations that do not exist.

---

## Machine learning

`eegfeat.model` (install with `pip install "eegfeat[model]"`) takes per-epoch feature tables into
cross-validated prediction. The hard part of modeling EEG is not fitting — it is not fooling
yourself, so most of what this subpackage does is refuse shortcuts that inflate scores.

### Load the cohort and build a design

```python
from pathlib import Path

import numpy as np
import pandas as pd

import eegfeat.model as efm
from eegfeat.io import read_dataset

dataset = read_dataset(sorted(Path("derivatives/eegfeat").rglob("*_features.tsv")))
```

`dataset.table` is every recording stacked into one matrix; `dataset.targets` holds the epoch
metadata the runner copied in. Recordings that excluded different bad channels have different
columns, so the default stacks them onto the union and marks a column a recording never measured
as missing, with zero coverage, rather than quietly aligning by position.

Now choose which features to model and which column is the answer:

```python
# Filter by meaning. These are the labels on the columns, not the function names:
# read them with sorted({m.measure for m in dataset.table.meta}).
selection = efm.Selection(measure=("band_power", "erds_mean"), band=("alpha", "beta"))

design = efm.build_design(
    dataset.table,
    dataset.targets,
    target="pain_rating",
    groups="subject",
    runs="run",
    selection=selection,
)

design.X.shape        # epochs by features
design.y              # the target, one value per epoch
design.groups         # the subject each epoch belongs to
design.runs           # the run each epoch belongs to
```

Features and targets are matched by recording, epoch, and event — never by row order — and a
mismatch is an error rather than a silent misalignment.

### Decide what "generalize" means, then cross-fit

This is the choice that determines what your score means.

- `efm.loso_folds(design.groups)` holds out **one subject at a time**. It answers: would this model
  work on someone it has never seen?
- `efm.within_subject_folds(design.groups, design.runs, inner_splits=3, seed=42)` holds out **runs
  within each subject**, so every subject appears on both sides. It answers: can we track this
  measure within a person across sessions?

The two usually give very different numbers. Report the one that matches your claim.

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

Imputation, scaling, feature selection, column harmonization, and any target residualization are
fitted **inside each training fold** and applied to the held-out fold. Hyperparameters are tuned in
an inner loop that never crosses the same grouping boundary as the outer one. Nothing about the
held-out subject reaches the model that predicts them.

Ridge is one option; `elasticnet_pipeline` and `random_forest_pipeline` work the same way, each
with a matching `*_grid()`.

### Score it at the subject level

Pooling every trial from every subject into one correlation lets a few talkative subjects speak for
the group. The primary metric averages each subject's correlation in Fisher *z* space instead, with
every subject weighted equally.

```python
y_true, y_pred, eval_groups, _, _ = efm.fold_results(predictions, groups=design.groups)
eval_groups = np.asarray(eval_groups, dtype=object)

metrics, per_subject = efm.regression_metrics(y_true, y_pred, eval_groups)
print(f"subject-level r {metrics['subject_level_r']:+.3f}")
print(f"pooled r        {metrics['pearson_r']:+.3f}")

# regression_metrics reports the point estimate; the aggregation behind it also has the interval.
summary = efm.subject_level_r(
    pd.DataFrame({"subject_id": eval_groups, "y_true": y_true, "y_pred": y_pred})
)
print(f"95% CI [{summary.ci_low:+.3f}, {summary.ci_high:+.3f}]")
```

This is what lands in [`examples/example_model_scores.tsv`](examples/example_model_scores.tsv):
`r = 0.41`, `95% CI [0.34, 0.48]`, and each subject's own correlation beside it.

### Ask whether it beats chance

A permutation test refits the whole cross-validated procedure on shuffled labels. Shuffling happens
within each subject (or within each run), so it destroys the brain-behaviour link without also
destroying the between-subject structure the model was never using anyway.

```python
null = efm.permutation_test(
    folds,
    design.X,
    design.y,
    design.groups,
    design.runs,
    pipeline,
    efm.ridge_grid(),
    metrics["subject_level_r"],          # the observed statistic
    config=efm.NullConfig(scheme="within_subject", n_permutations=1000),
    inner=inner,
    seed=42,
    harmonization="intersection",
)
print(f"p = {null.p_value:.3f}")
```

The observed statistic has to come from the *same* configuration — same folds, pipeline, grid,
seed, harmonization, and residualization. Pass a number computed some other way and the test says
so instead of quietly comparing incomparable things. Permutations that fail to fit abort the test
rather than being dropped, because dropping draws would reshape the null.

### Say how uncertain each prediction is

Conformal intervals give a range per prediction with a distribution-free coverage guarantee.

```python
intervals = efm.prediction_intervals(
    pipeline,
    design.X,
    design.y,
    design.X[:5],
    alpha=0.10,
    method="cv_plus",        # or "split", or "quantile"
    groups=design.groups,
    seed=42,
)
print(intervals.lower, intervals.upper, intervals.calibration_unit)   # … 'trial'
```

Passing `groups` keeps the fitting and calibration splits subject-disjoint. Calibration scores stay
one per trial either way, which is why `calibration_unit` is always `"trial"`: the guarantee is
marginal over trials, not over participants.

### Find out what drove it

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

for index in np.argsort(importance.values)[::-1][:5]:
    print(importance.feature_names[index], importance.values[index])

# Because the columns are labelled, importance can be read at the level you think in.
selected = efm.select(dataset.table, selection)
efm.aggregate_by(importance, selected.meta, "band")      # {'alpha': 0.022, 'beta': 0.056}
efm.aggregate_by(importance, selected.meta, "measure")   # {'band_power': 0.078, …}
```

Importance is measured on held-out folds, not on the data the model was fitted to. SHAP is
available as `shap_importance_over_folds` with the `importance` extra.

### Classification

Binary targets follow the same path with classification pipelines. Labels must be coded 0 and 1 —
a third value, such as a `-1` for a missing response, is refused up front rather than fitted as a
silent third class.

```python
labels = efm.build_design(
    dataset.table, dataset.targets, target="painful", groups="subject", runs="run",
    selection=selection,
)
answered = np.isin(labels.y, (0.0, 1.0))      # drop trials with no response

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
# fold_results carries labels only, so AUC needs the positive-class probability rebuilt
# from the fold records, in the same fold order it sorts by.
in_fold_order = sorted(classifications, key=lambda fold: fold.fold)
positive = np.concatenate([f.y_prob[:, f.classes.index(1)] for f in in_fold_order])
result = efm.classification_metrics(
    truth.astype(np.intp), predicted.astype(np.intp),
    y_prob=positive, groups=np.asarray(class_groups, dtype=object),
)
print(result.balanced_accuracy, result.auc)
```

`svm_pipeline`, `random_forest_classifier_pipeline`, and `ensemble_pipeline` are drop-in
alternatives. Passing `groups` makes every reported scalar a mean over subjects with equal weight;
the confusion matrix stays pooled over trials, so accuracy recomputed from it will not match the
reported accuracy. Report one convention, not both.

---

## What it refuses to do

These are the defaults, and most of them are the reason a number here may be lower than the same
number computed elsewhere.

- **It will not let a window borrow data from outside itself.** Morlet coefficients are masked to
  the frequencies whose wavelets actually fit inside the requested window.
- **It will not report a peak that is only the 1/f slope.** Peak frequency is measured after
  dividing out a fitted aperiodic background, and the fit deliberately spans more than the band,
  because a 1/f slope estimated from five hertz is not a 1/f slope.
- **It will not calibrate a threshold on the data it is testing.** Burst detection and ERDS take
  their reference from a baseline window you nominate.
- **It will not turn a group estimate into per-trial rows.** ITPC, wPLI, and envelope correlation
  stay in their own table, and the modeling layer rejects them outright.
- **It will not confuse "missing" with "rejected".** Coverage reports numerical finiteness, and
  Morlet spectra separately report how much of a window had complete wavelet support.
- **It will not fit preprocessing on data it is about to predict.** Every fold-local step is fitted
  on training rows only, and inner tuning refuses fewer than two training groups.
- **It will not let loud subjects outvote quiet ones.** Subject-level aggregation weights each
  subject equally.
- **It will not filter its own null.** Permutation draws are all kept; a failed fit is an error.
- **It will not overstate conformal coverage.** Intervals are marginal over trials, and the result
  says so.

---

## Documentation and development

The [Sphinx documentation](https://joshuaduq.github.io/EEGFeatML/) is the canonical reference:

- [Quick Start](docs/quickstart.rst) — end-to-end walkthroughs.
- [Methods](docs/methods.rst) — the mathematics behind each measure.
- [Modeling](docs/modeling.rst) — designs, cross-fitting, metrics, nulls, uncertainty, importance.
- [Command Line Runner](docs/runner.rst) — recipe syntax and output formats.
- [API Reference](docs/api.rst) — every signature.

```bash
pytest          # unit and regression tests
mypy src        # strict type checking
ruff check src tests
```

Continuous values in the test suite are checked against analytic derivations and, where one
exists, an independently executed reference implementation. Those fixtures are regression
evidence; they are not by themselves evidence of scientific validity.

---

## License

MIT. See [LICENSE](LICENSE).
