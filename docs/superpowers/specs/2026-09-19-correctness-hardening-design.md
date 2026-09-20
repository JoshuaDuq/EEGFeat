# EEGFeat Correctness Hardening Design

## Goal

Make EEGFeat fail fast on invalid scientific inputs, prevent incomplete connectivity
tables from producing valid-looking graph metrics, make spectral entropy's grid
assumptions explicit, align advertised extras with implemented behavior, and test the
supported package surface in CI.

This phase does not add new feature families. Relative band power, Hjorth parameters,
line length, RMS, per-epoch PLV, and custom feature registration will be designed
separately after the current contracts are reliable.

## Scope

The change has five independent but small hardening units:

1. Remove the unimplemented `knee` extra and every claim that EEGFeat currently offers
   spectral knee fitting.
2. Reject graph construction unless each estimator/node-set group contains exactly one
   column for every unordered node pair.
3. Validate scientific ranges and discrete parameters at their public entry points.
4. Restrict normalized spectral entropy to uniformly spaced frequency grids.
5. Expand CI to cover supported Python endpoints, optional connectivity behavior, and
   installation from a built wheel.

Existing feature values for valid inputs remain unchanged, except that
`spectral_entropy` will now reject nonuniform frequency grids instead of returning a
grid-dependent value.

## Packaging Contract

`specparam` is unused by the package, there is no public knee function, and there is no
runner measure for a knee estimator. The `knee` optional dependency and its README and
installation-guide claims will therefore be removed. This is preferable to introducing
a new estimator during a correctness pass: a knee model requires separate decisions
about fixed-versus-knee model selection, minimum frequency span, parameter reporting,
and fit-quality diagnostics.

The package description will be updated from “spectral feature extraction” to reflect
the existing temporal, connectivity, and modeling functionality. Version handling is
out of scope because both current version declarations agree and changing the release
mechanism is not required by a demonstrated defect.

## Complete Graph Validation

Graph summaries accept pairwise `FeatureTable` inputs grouped by estimator definition
and node set. For each group, validation will derive the unique node names from
`FeatureMeta.nodes` and construct the expected set of all unordered pairs. It will
compare that set with the supplied edges before any row is reduced.

The validator will raise `ValueError` when:

- an edge is duplicated, including reversed duplicates;
- an edge connects a node to itself;
- one or more expected pairs are absent;
- metadata contains a malformed pair.

The error will name missing or duplicate edges. A present edge whose value is NaN is
structurally complete and remains a data-quality condition; the existing graph kernels
may continue treating non-finite weights as absent connections. This preserves the
distinction between a missing feature column and a measured column with unavailable
data.

## Scientific Input Validation

Validation belongs at public data and function boundaries:

- `FeatureTable.coverage`, `Spectra.coverage`, `Spectra.support`, `Signal.coverage`, and
  `BandSignal.coverage` must be finite and lie in `[0, 1]`.
- `Spectra.freqs` must be finite, non-negative, non-empty, and strictly ascending.
- spectral power arrays must reject finite negative values because both supported
  representations are power quantities. NaNs remain valid missing-data markers.
- signal and spectral containers must contain at least one epoch, channel, and sample or
  frequency bin. Channel names must be unique.
- `aperiodic` and `aperiodic_ratio` require finite positive `peak_rejection_z` and an
  integer `max_iterations >= 1`; booleans are not accepted as integers.
- `sample_entropy` and `multiscale_entropy` require an integer `order >= 1` and finite
  positive `r`.
- every multiscale entropy scale must be an exact positive integer, booleans are
  rejected, and scales must be unique. Input order remains output order.

Unexpected or invalid configuration raises immediately. NaN outputs remain reserved
for valid configurations where the supplied data cannot support an estimate.

## Spectral Entropy Contract

The current entropy formula is the normalized discrete Shannon entropy of PSD samples.
That estimator assumes equal-width frequency bins. Weighting unequal bins by their
width and normalizing by `log(n_bins)` does not yield one for a flat PSD and changes the
estimand with the sampling grid.

`spectral_entropy` will therefore require an approximately uniform frequency grid over
the selected band. Uniformity will be checked with a strict numerical tolerance against
the first spacing. On valid grids, probabilities will be formed directly from finite,
non-negative PSD values without gradient weights. A flat spectrum will equal one and a
single-bin spectrum will equal zero.

Centroid, bandwidth, and spectral edge will retain frequency-width weighting because
those are numerical integrals and correctly support nonuniform grids. The entropy
docstring and methods documentation will state the uniform-grid requirement and explain
why it differs from the other descriptors.

## CI Design

CI will be divided into focused jobs:

- Core tests and static checks on Python 3.11.
- Core test execution on the newest declared Python endpoint used by the project.
- Optional connectivity and microstate tests with their extras installed, ensuring the
  computational paths run rather than only the missing-dependency guards.
- Build an sdist and wheel, install the wheel into a clean environment, and run import,
  version, and CLI smoke checks.

The workflow will continue to require the modeling reference fixture and explicitly run
the modeling suite. Heavy SHAP testing remains optional because it is not needed to
validate the base or modeling installations.

## Test Strategy

Each behavior change starts with a focused failing test:

- incomplete graph tables and self-edges raise;
- valid complete graphs retain their existing values;
- every coverage/support boundary rejects values below zero, above one, NaN, and
  infinity;
- invalid aperiodic iteration settings and entropy parameters raise;
- fractional, boolean, duplicate, empty, and non-positive MSE scales raise;
- a flat PSD on a uniform grid has entropy one;
- a nonuniform grid is rejected with an actionable message;
- the package metadata no longer declares `knee` or `specparam`;
- CI configuration contains the optional-dependency and wheel-install jobs.

After each unit passes its focused tests, the full suite, Ruff, Black, strict mypy, and a
warning-as-error documentation build will run. Tests will use `MNE_DONTWRITE_HOME=true`
where needed so verification does not depend on writable user configuration.

## Compatibility and Migration

No fallback or compatibility shim will be added. Callers relying on invalid coverage,
fractional entropy scales, zero aperiodic iterations, incomplete graph schemas, or
nonuniform-grid spectral entropy will receive an explicit error. These cases currently
produce misleading results, so surfacing them is the intended correction.

Users needing entropy from a logarithmic frequency grid must recompute or interpolate
their PSD onto a uniform-Hz grid before constructing the entropy feature. The exception
message and documentation will say this directly.
