# Feature-to-Model Integration Design

## Goal

Make per-epoch feature computation outputs directly usable by the machine-learning
pipeline across multiple recordings, both in memory and after runner persistence,
without weakening row identity or scientific independence guarantees.

Cross-trial features remain outside the modeling interface. They represent group-level
estimates and must never be broadcast onto epochs as independent observations.

## Public interfaces

### Row-wise table stacking

Add `stack_rows(tables)` to `eegfeat.table` and the top-level `eegfeat` API. It accepts a
non-empty sequence of per-epoch `FeatureTable` objects and returns one table containing
their rows in input order.

Every input must:

- carry `row_ids` and no `row_labels`;
- have exactly the same ordered `FeatureMeta` records;
- have unique row identities across the combined dataset.

Values and coverage are concatenated on axis zero. Flags are combined by the union of
their names, with `False` cells where an input table did not contain a flag. Schema
mismatches, cross-trial inputs, and duplicate identities raise `ValueError`.

### Persisted datasets

Add an immutable `FeatureDataset` record in `eegfeat.io` with two fields:

- `table`: the vertically stacked per-epoch `FeatureTable`;
- `targets`: a `pandas.DataFrame` aligned one-to-one and in the same order as the table.

Add `read_dataset(paths)` to load one or more runner-generated feature TSV bundles. For
each bundle it reads the `FeatureTable`, reads only the descriptor columns declared by
the JSON sidecar, and constructs canonical `recording`, `epoch`, and `event` columns
from the table's `row_ids`. Those canonical keys are not inferred from filenames or
descriptor text. The per-file tables are combined with `stack_rows`, and descriptor
frames are concatenated in the same input order.

The loader rejects an empty path sequence, cross-trial tables, missing descriptor
columns, descriptor columns that conflict with canonical keys, and incompatible table
schemas. Unexpected file, JSON, and parsing errors surface unchanged.

## Modeling boundary

`build_design()` continues to accept one per-epoch `FeatureTable` and an aligned target
frame. Its error for a table without `row_ids` will explicitly state that modeling is
per-epoch only and cross-trial/group-row tables are unsupported.

No model function reads files. `read_dataset()` is the storage adapter, `stack_rows()`
is the table operation, and `build_design()` remains the design-matrix adapter.

## Data flow

In-memory usage:

1. Compute one `RecordingFeatures` object per recording.
2. Collect each non-null `.epochs` table.
3. Call `stack_rows()`.
4. Construct or obtain a target frame with canonical row keys.
5. Call `build_design()` and the existing cross-fitting API.

Persisted usage:

1. Run the feature recipe over recordings.
2. Pass the resulting `*_features.tsv` paths to `read_dataset()`.
3. Call `build_design(dataset.table, dataset.targets, ...)`.
4. Cross-fit with `groups="recording"` or another descriptor column.

## Documentation

Update the README modeling example to use the implemented signatures:

- `targets=` and `groups=` for `build_design()`;
- `PreprocessingConfig()` and an explicit seed for `ridge_pipeline()`;
- an explicit seed for `cross_fit_regression()`.

Document both in-memory stacking and persisted `read_dataset()` usage, and state that
the modeling interface accepts per-epoch tables only.

## Verification

Tests will be written before implementation and will cover:

- successful row stacking and preservation of values, coverage, flags, metadata, and
  row order;
- rejection of incompatible schemas, duplicate row identities, and cross-trial rows;
- loading several runner outputs into one aligned `FeatureDataset`;
- a complete three-recording
  `compute_features -> stack/read_dataset -> build_design -> LOSO regression` path;
- the explicit cross-trial rejection message;
- public API exports and executable README examples.

The complete pytest suite, Ruff, and strict mypy checks must pass before completion.
