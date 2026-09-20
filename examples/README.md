# Example output

Real files, produced by the real pipeline, from simulated recordings. Five subjects, two runs
each, sixteen trials per run. Regenerate everything with:

```bash
python examples/make_examples.py
```

## What the feature extraction writes

[`recipe.toml`](recipe.toml) is the input: bands, windows, regions, and one entry per measure.
`eegfeat run` applies it to every epochs file and writes, for each recording:

- [`sub-01_task-pain_run-01_features.tsv`](sub-01_task-pain_run-01_features.tsv) — one row per
  epoch. The first columns are the epoch's identity and the metadata you attached to it
  (`subject`, `run`, `trial`, `intensity`, `rating`, `painful`); everything after that is a
  feature, named for what it measures.
- [`sub-01_task-pain_run-01_features_coverage.tsv`](sub-01_task-pain_run-01_features_coverage.tsv)
  — the same shape, saying what fraction of each cell was finite.
- [`sub-01_task-pain_run-01_features.json`](sub-01_task-pain_run-01_features.json) — the sidecar.
  Every column's measure, band, space, window, unit, normalization, full computation parameters
  and hash, plus flags and the provenance of the run.
- [`sub-01_task-pain_run-01_crosstrial.tsv`](sub-01_task-pain_run-01_crosstrial.tsv) — measures
  defined across trials rather than within one, here inter-trial phase coherence. One row per
  trial group, kept out of the per-epoch table on purpose.

## What the modelling writes

Leave-one-subject-out ridge regression predicting `rating` from the band power and ERDS columns,
scored per subject and tested against 200 within-subject permutations.

- [`example_model_scores.tsv`](example_model_scores.tsv) — the cohort result with its confidence
  interval and permutation *p*, then each subject's own correlation.
- [`example_model_predictions.tsv`](example_model_predictions.tsv) — every held-out prediction,
  with the fold it came from, so you can plot or re-score it yourself.

The numbers are what this simulation deserves, not a benchmark: alpha really is suppressed in
proportion to stimulus intensity here, and the rating carries a lot of noise that no EEG feature
could explain.
