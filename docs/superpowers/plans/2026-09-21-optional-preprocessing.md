# Optional EEG Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute inline unless the user requests delegation.

**Goal:** Add an optional, reproducible EEG preprocessing workflow in which every supported stage is available individually, sequentially, and end to end, with inspectable checkpoints and validated FIF epochs for feature extraction.

**Architecture:** Keep preprocessing in `eegfeat.preprocessing`, separate from feature extraction and modeling. A small stage registry calls focused numerical functions; the CLI's `step`, `next`, and `run` commands share that execution path and checkpoint validation. Use MNE's native objects and established optional packages; the existing feature runner consumes only the exported final epochs.

**Tech stack:** Python ≥3.11; existing NumPy, SciPy, pandas, and MNE ≥1.8 dependencies; `preprocessing` extra with PyYAML ≥6.0, scikit-learn ≥1.3, h5io, h5py, and filelock; `preprocessing-auto` extra with PyPREP ≥0.9.0 and autoreject ≥0.4.2; optional `preprocessing-gui` extra with mne-qt-browser and PyQt6. Use pytest, Ruff, Black, mypy, and Sphinx. Test each optional integration against its declared minimum; no compatibility shims.

---

## 1. Scope and repository integration

The user approved continuous raw EEG → clean epochs, a separate preprocessing package, YAML configuration, optional reviewed ICA, and saving epochs with rejection information and provenance. They requested this implementation plan instead of a separate design document. This document plans implementation; it does not claim that the feature exists or that its tests have passed.

Repository observations, checked on 2026-09-21:

- `src/eegfeat/runner/cli.py` owns `eegfeat init`, `check`, and `run`.
- `src/eegfeat/runner/batch.py` reads FIF epochs and already excludes marked bad channels according to its feature recipe.
- Feature recipes are TOML. Preserve them; the new preprocessing recipe is YAML.
- The top-level public API is tested exactly in `tests/test_public_api.py`. Expose new names in the preprocessing subpackage, without expanding `eegfeat.__all__`.
- Package metadata advertises MNE ≥1.8. The local environment has MNE 1.13.2. Validate the new code against both endpoints; do not introduce version-specific branches or silent compatibility paths.
- Sphinx already excludes `docs/superpowers`, so this plan does not enter the public documentation.

The CLI initially processes one explicitly named recording per YAML file. Repeated invocations cover multiple recordings with their own channel/ICA decisions. Do not add a second batch scheduler, BIDS dataset discovery, raw-file concatenation, or subject-wide ICA pooling. The Python API accepts an MNE Raw object loaded by any appropriate MNE reader. The CLI supports FIF/FIF.GZ, EDF/BDF, BrainVision VHDR, and EEGLAB SET through `mne.io.read_raw`; use MNE's reader dispatch and propagate reader errors.

The latest user requirement expands this plan: each stage must be offered independently, sequential execution must be easy, and completion must cover the ordinary scalp-EEG workflow rather than a single fixed ICA recipe. Include manual/automated channel and segment quality assessment, cropping, optional stimulation repair, filtering, ICA/SSP/EOG-regression alternatives, manual/automated epoch rejection and repair, interpolation, reference, sampling changes, detrending, baseline, and QC/export. Detection and correction remain distinct operations. Alternatives are explicit choices, not operations stacked automatically.

Scope is scalp EEG for resting-state/event-related analysis. No honest implementation can claim every method for every acquisition modality. ASR, ICLabel, robust PREP rereferencing, CSD, source reconstruction, simultaneous EEG-fMRI gradient correction, and MEG Maxwell filtering are separate methods outside this plan. The step catalog must show the offered methods and prerequisites clearly; it must not advertise unsupported methods as completed pipeline steps. Do not substitute a MEG bad-channel detector for EEG quality assessment.

### Review findings addressed in this revision

1. The previous `prepare_raw`/`preprocess` interface hid individual stages; expose named numerical functions and a shared step runner.
2. Saving only final epochs and ICA did not support practical sequential work; save validated, immutable stage checkpoints.
3. The prior recipe omitted useful standard options; add the methods above with explicit selection and established library implementations.
4. Manual review was described mostly as editing files elsewhere; offer MNE viewers and equivalent headless review files directly.
5. Configuration and method changes need dependency-aware invalidation, so a saved ICA/rejection decision cannot silently attach to different data.
6. One-shot, stepwise, and resumed execution need numerical and metadata equivalence tests, not just parser tests.

## 2. Scientific basis and implementation decisions

These are primary documentation sources reviewed for this plan. The exact choices below are this project's policy, not a claim that MNE prescribes a universal EEG recipe.

| Reference | Use in this implementation |
| --- | --- |
| [MNE-BIDS-Pipeline processing steps](https://mne.tools/mne-bids-pipeline/stable/features/steps.html) | Separate data quality, filtering, ICA fitting/review, epoching, ICA application, and residual rejection. |
| [MNE filtering and resampling](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html) | Filter continuous data; downsample epochs only after an explicit anti-aliasing check. |
| [MNE-BIDS-Pipeline filtering](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/filter.html) | Optional notch before high/low-pass filtering; no universal 1 Hz analysis high-pass. |
| [MNE handling bad channels](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html) | Preserve bad-channel labels; exclude bad electrodes from fitting; use MNE interpolation with valid sensor geometry. |
| [MNE rejecting bad spans](https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html) | Preserve and extend annotations; use annotation-aware epoch rejection. |
| [MNE amplitude annotations](https://mne.tools/stable/generated/mne.preprocessing.annotate_amplitude.html) | Optional flat/abrupt-sample-change annotation and persistent-channel marking; not a general noisy-electrode detector. |
| [MNE break annotations](https://mne.tools/stable/generated/mne.preprocessing.annotate_break.html) | Optional explicitly configured task-break exclusion. |
| [MNE ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) | Fit on a separate high-pass-filtered copy; retain the analysis passband. |
| [MNE-BIDS-Pipeline ICA review](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/ssp_ica.html) | Separate artifact suggestions from reviewed component exclusions. |
| [MNE ICA API](https://mne.tools/stable/generated/mne.preprocessing.ICA.html) | Fit, score, save, reload, and apply the fitted decomposition without refitting it. |
| [MNE Epochs API](https://mne.tools/stable/generated/mne.Epochs.html) | Explicit baseline, projection, event, annotation, and rejection behavior. |
| [MNE-BIDS-Pipeline amplitude rejection](https://mne.tools/mne-bids-pipeline/stable/settings/preprocessing/artifacts.html) | Residual peak-to-peak rejection after ICA correction. |
| [MNE EEG referencing](https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html) | Explicit final reference, handling bad channels and an omitted physical reference electrode. |
| [MNE fixed-length epochs](https://mne.tools/stable/auto_tutorials/epochs/60_make_fixed_length_epochs.html) | Resting-state epochs with precise duration and overlap. |
| [MNE 1.8 ICA](https://mne.tools/1.8/generated/mne.preprocessing.ICA.html), [Report](https://mne.tools/1.8/generated/mne.Report.html), and [reader](https://mne.tools/1.8/generated/mne.io.read_raw.html) | Check the declared minimum supported API, including `random_state` and report signatures. |
| [PyPREP NoisyChannels](https://pyprep.readthedocs.io/en/stable/generated/pyprep.NoisyChannels.html), [published release](https://pypi.org/project/pyprep/) | Optional EEG channel-quality suggestions using the released 0.9.0 API, including annotation-aware input. Use detector output, not a hidden PREP rereferencing pipeline. |
| [MNE muscle annotations](https://mne.tools/stable/generated/mne.preprocessing.annotate_muscle_zscore.html) | Detect on a copy with sufficient original bandwidth, before analysis low-pass filtering. |
| [MNE bridged electrodes](https://mne.tools/stable/generated/mne.preprocessing.compute_bridged_electrodes.html) | Optional electrode-pair diagnostic for review; no automatic pair-to-channel exclusion rule. |
| [MNE stimulation repair](https://mne.tools/stable/generated/mne.preprocessing.fix_stim_artifact.html) | Explicit narrow event-locked repair, with synthetic samples recorded; not a complete TMS/EEG-fMRI denoising solution. |
| [MNE SSP](https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html) | Reviewed EOG/ECG projector fitting and separate application. |
| [MNE EOG regression](https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html) | Explicit reference before fitting and applying the same saved regression coefficients. |
| [autoreject example](https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html), [API](https://autoreject.github.io/stable/generated/autoreject.AutoReject.html) | Separate fit and transform, channel-by-epoch repair log, and persisted model. |

## 3. Stage catalog and scientific order

Every row below has a stable CLI stage name, a public numerical operation, a documented input/output contract, and tests. Conditional stages are visible as `disabled` with a reason. Mandatory integrity checks cannot be turned off. `steps` lists all available stages, whereas `status` shows the current recording's resolved path.

| Order / stage | Operation and scientific constraints | Checkpoint output |
| --- | --- | --- |
| 1 `load` | Load raw with SI units, preserve source metadata, validate finite physiology, and fingerprint the complete input. Never mutate the acquisition files. | Original Raw and input identity |
| 2 `prepare` | Explicit channel renaming/types/drop list, digitization, optional bipolar auxiliary channels, and inactive-projector policy. No guessed montage/units/reference. | Prepared Raw, channel inventory |
| 3 `events` | Resolve annotation/stim/file events on the original sampling grid; optionally apply a declared trigger-delay correction and attach epoch metadata. | EventData and original event ledger |
| 4 `crop-raw` | Optional acquisition-relative interval; preserve absolute event sample numbers and log events outside the retained interval. No arbitrary raw concatenation. | Cropped Raw, updated event eligibility |
| 5 `annotate` | Existing/manual BAD spans, breaks, flat/amplitude candidates, and optional muscle candidates on original-bandwidth data. Detector results remain labeled suggestions until the raw decision is applied. | Raw plus annotation candidates/scores |
| 6 `detect-bads` | Optional PyPREP channel suggestions and MNE electrode-bridge diagnostics; preserve per-method evidence and sampling/passband prerequisites. | Channel/pair quality report |
| 7 `review-raw` | Apply explicit channel/segment decisions. Offer MNE visual review or a validated decision YAML; record unreviewed status if a study explicitly disables this gate. | Reviewed Raw and decision identity |
| 8 `repair-stim` | Optional MNE event-locked repair on specified channels/IDs; record each repaired interval. Reject overlap/edge cases that cannot be repaired as configured. | Raw, repair ledger |
| 9 `notch` | Optional explicit mains/harmonic frequencies; never infer country or silently skip invalid harmonics. | Raw and actual filter specification |
| 10 `filter` | Optional continuous high/low-pass using fixed MNE FIR defaults; annotation-aware boundaries and recorded impulse response. | Analysis-band Raw |
| 11 `artifact-reference` | Optional explicit reference before model fitting. Required explicit choice for EOG regression; ICA/SSP default to existing reference. Final reference remains a separate stage. | Raw with fit/application reference identity |
| 12 `fit-artifact` | One of ICA, EEG EOG/ECG SSP, EOG regression, or disabled. Fit on good measured EEG with consistent reference; omit bad training spans. Save exact training identity/model. | ArtifactModel plus scores/training QC |
| 13 `review-artifact` | Inspect and select ICA exclusions or SSP vectors; accept/reject EOG-regression correction. Require a decision bound to the exact model. | ReviewedArtifact |
| 14 `epoch` | Event-related or fixed-length epochs on the acquisition grid; optional extra time padding for later resampling. baseline=None, detrend=None, proj=False, decim=1. Honor BAD annotations. | Uncorrected epochs and full event ledger |
| 15 `apply-artifact` | Apply the exact reviewed operator; no refitting, reference/channel changes, implicit SSP, or premature baseline. | Corrected full-rate epochs |
| 16 `fit-rejection` | For autoreject, fit good EEG at full rate using explicit cross-validation settings and seed. Fixed thresholds need no fitted model. | RejectionModel or disabled-stage record |
| 17 `reject` | Fixed peak-to-peak/flat rejection or autoreject transform; retain every rejection and per-epoch repair decision. Work before global interpolation, sampling changes, or final reference. | Retained epochs, reject/repair ledger |
| 18 `review-epochs` | Optional manual epoch exclusion by stable original event IDs; never reuse displayed row numbers across changed selections. | Reviewed retained epochs |
| 19 `interpolate` | Optional global bad-EEG interpolation with valid geometry. Keep globally bad electrodes separate from autoreject's transient per-epoch repairs. | Epochs, restored-vs-still-bad channel list |
| 20 `reference` | Final average/named reference or unchanged; explicitly restore missing acquisition reference electrodes here if requested. Preserve auxiliary channels. | Referenced epochs |
| 21 `resample` | None, guarded integer decimation, or MNE polyphase epoch resampling with explicit padding. No raw resampling in this workflow. | Epochs on declared output time grid |
| 22 `crop-epochs` | Remove processing padding and return to requested analysis bounds; record actual sample-aligned bounds. | Final analysis interval |
| 23 `detrend` | Optional constant/linear per-epoch EEG detrending via SciPy; explicit, separate from baseline, off by default. | Detrended epochs |
| 24 `baseline` | Optional mean subtraction on the final time grid; reject empty/out-of-window baseline. | Final numerical epochs |
| 25 `report` | Validate final data and build before/after QC, full provenance, and artifact/rejection summaries. No further numerical mutation. | HTML QC and validated result |
| 26 `export` | Export double-precision FIF, event/rejection TSV, per-epoch repair TSV where applicable, and manifest. | Final feature-runner input bundle |

The stage registry is a fixed dependency graph, not an arbitrary reorderable plugin system. A user can run each operation explicitly or request a target stage with its dependencies; the executor rejects scientifically incompatible orders. The rows show canonical execution order; independent fit/epoch branches use their actual declared dependencies, not an artificial dependency on the preceding row.

Scientific invariants:

- The default reference remains the acquisition reference through artifact correction/rejection. An explicit `artifact.reference` changes it **before both fitting and application**. Final referencing never precedes application of a model fitted in a different reference.
- Do not interpolate globally bad EEG before fitting artifact models or estimating epoch rejection. Exclude those channels from fitting; restore them later only when requested.
- Inactive projectors require an explicit input policy: error by default, apply, or discard-inactive. Previously applied projections remain part of data identity. SSP fitted by this workflow is stored separately until `apply-artifact`.
- Missing acquisition-reference channels are added after rejection/global interpolation, before final reference. They are not fed as flat synthetic measurements to channel detection or ICA.
- Muscle/high-frequency-noise detection runs before the analysis low-pass. Requested diagnostic bands must fit the source passband and Nyquist; fail rather than silently altering detection bands.
- Reject residual spikes before downsampling. Filter, fit, or reject on padded epochs only under the declared analysis/training windows; padding samples do not silently enlarge the PTP decision window.
- Keep original event samples, corrected trigger samples, original sample rate, current sample rate, and actual time bounds as distinct fields. Epoch resampling never makes original sample indices into output-grid indices.
- Raw filtering around excluded spans can introduce adjacent transients. Record FIR support and make annotation padding explicit; do not claim all neighboring samples are unaffected. The initial implementation rejects too-short filterable segments rather than silently changing filters.
- All fitted methods are per recording and label-free. Offer their fit/apply functions separately for callers who need training-only fits in predictive analysis; never label whole-recording unsupervised cleaning as proof of leakage-free evaluation.
- A notch creates an internal spectral hole. Record it and document the limits of the feature runner's broad passband checks.

### 3.1 Easy sequential use

```bash
eegfeat preprocess init preprocessing.yaml --mode events
eegfeat preprocess check preprocessing.yaml
eegfeat preprocess steps preprocessing.yaml
eegfeat preprocess run preprocessing.yaml --until review-raw
eegfeat preprocess review preprocessing.yaml raw
eegfeat preprocess next preprocessing.yaml
eegfeat preprocess run preprocessing.yaml
eegfeat preprocess status preprocessing.yaml
```

Edit input/output paths, event mapping, and the desired scientific settings before `check`. If an artifact method is configured, `run` stops at its review gate; continue with:

```bash
eegfeat preprocess review preprocessing.yaml artifact
eegfeat preprocess run preprocessing.yaml
```

`--mode resting` generates fixed-length epoch settings. These are structural starter recipes, not hidden physiological presets: study-specific filters, threshold values, reference, and event mapping are visible YAML values requiring explicit selection. The short README path uses `next`/`run`; advanced documentation shows individual stages, after enabling those filters in the recipe:

```bash
eegfeat preprocess step preprocessing.yaml notch
eegfeat preprocess inspect preprocessing.yaml notch
eegfeat preprocess step preprocessing.yaml filter
eegfeat preprocess inspect preprocessing.yaml filter
```

Command semantics:

- `step CONFIG STAGE`: run that single enabled stage, only if its required parents are current. Missing parents produce a precise `run CONFIG --until PARENT` instruction; do not run hidden prerequisites.
- `next CONFIG`: execute the next enabled pending stage in catalog order; never repeat an already completed numerical operation. At a review gate, display the required review command without pretending the run is complete.
- `run CONFIG [--until STAGE]`: run all needed dependencies to the target. Reuse only validated matching checkpoints. Stop at unresolved review decisions. A target review stage means produce its candidates/report and stop until its decision exists.
- `status CONFIG`: show completed/pending/disabled/needs-review/stale/failed states, reason, relevant artifact path, and the exact next command. Normal status reads metadata; `status --verify` hashes stored payloads. Every executing command verifies payloads it consumes regardless of status mode.
- `inspect CONFIG STAGE`: open the corresponding MNE viewer when GUI support is installed and explicitly requested, or the existing HTML report using `--report`. Inspection never changes signal/review state; `--report` may generate a separate inspection HTML from an immutable checkpoint if it does not exist. No web GUI needs to be invented.
- `review CONFIG raw|artifact|epochs`: offer MNE's appropriate viewer, then save a validated review decision. Headless users supply `--decisions FILE`. A missing GUI package is an explicit error with the extra to install; it never silently changes the workflow to headless.
- `reset CONFIG --from STAGE`: explicitly make that stage and its descendants pending. Keep previous immutable artifacts and decisions for audit. No recursive deletion of user data.
- `init`, `check`, and `steps` write no processing outputs (`init` writes only its requested YAML). `check` reports configuration/data feasibility; it does not claim fitted models or final QC passed.

Return codes: 0 requested target completed; 3 needs review (checkpointed, not a failed algorithm and not a completed export); 2 configuration/prerequisite/stale-review errors; 1 known computation or I/O failures. A disabled stage explicitly requested by `step` returns a clear explanation and changes nothing. Unexpected exceptions keep their traceback.

### 3.2 Checkpoint and invalidation contract

Store intermediate artifacts under `output.directory/.preprocessing/output.name/`, which the existing feature runner excludes as a hidden path. Only `export` writes discovery-visible final epochs. Each stage directory contains its manifest, immutable parent IDs, resolved relevant settings, versions, decision hashes, output payload hashes, QC summary, and status. Use native FIF for Raw/Epochs/projectors, ICA FIF, HDF5 via established model save/read functions, and JSON/TSV/YAML for metadata. No pickle checkpoint protocol.

Stage identity is derived from stage implementation version, parent artifact identities, only that stage's scientifically relevant settings, and applicable review/model identities. Serialize checkpoint numerical data in double precision. Compute parent identities from persisted canonical artifacts so in-memory and resumed execution do not disagree because of FIF metadata normalization.

- Verify original source identity and every consumed checkpoint. Check the actual recording contents/metadata; a filename or mtime alone is insufficient. When a reader's full companion-file inventory cannot be established through public APIs, re-read and fingerprint the loaded source instead of assuming companions are unchanged. Document this validation cost.
- A changed bad-channel decision invalidates filtering/model/epoch descendants; changed ICA fitting parameters invalidate fit/review/application and their descendants; changed final baseline invalidates baseline/report/export only. Formatting-only YAML changes do not alter numerical identity.
- Derive fit identity from the actual data/fit settings, not the entire recipe. A new output directory does not change the scientific model identity. No silent migration of old schema versions.
- Unchanged completed stages are reported as reused. Corrupt payloads and stale user decisions raise; they are not treated as a cache miss followed by an unnoticed refit. User-authorized reset/review starts a new version.
- Stage execution never modifies its parent's payload. Publish a successfully validated temporary stage directory by rename, then update a small current-manifest pointer. Interrupted writes are not valid predecessors. Only one writer per recording workspace; use a lock and fail clearly on conflicting writers.
- Resuming after a review or failure never filters, rereferences, baselines, or applies an artifact model a second time to the already-transformed data.
- Numerical API calls and the persisted runner share stage functions. No separate large one-shot implementation. Saving can be omitted in the direct Python API, but its stage order/results must match the checkpointed path within declared serialization tolerances.

## 4. Configuration contract

Use frozen dataclasses for validated settings and dedicated result dataclasses. Parse YAML only at the file boundary. No arbitrary MNE kwargs, compatibility aliases, guessed reference, guessed montage, automatic mains-frequency selection, or swallowed exceptions.

The complete example below is a **study-specific example**, not default cleaning advice. Amplitude values are volts; time values are seconds; frequencies are Hz. The source filename and channels must match the user's data.

```yaml
input:
  path: ../../../data/sub-01_task-example_raw.fif
output:
  directory: ../../../derivatives/preprocessed/sub-01
  name: sub-01_task-example
channels:
  rename: {}
  types: {VEOG: eog, ECG: ecg, STI: stim}
  drop: []
  bads: []
  montage: standard_1020
  interpolate_bads: false
  bipolar: []
  projections: error
crop: null
bad_channels: null
bridges: null
stimulation: null
workflow:
  raw_review: required
  epoch_review: optional
annotations:
  bad_spans: []
  amplitude: null
  breaks: null
  muscle: null
filter:
  l_freq: 0.1
  h_freq: 40.0
  notch_freqs: [60.0]
artifact: null
rejection:
  method: thresholds
  reject: {eeg: 0.00015}
  flat: null
  tmin: null
  tmax: null
sampling: null
epochs:
  kind: events
  events:
    source: annotations
    event_id: {stimulus: 1}
  tmin: -0.5
  tmax: 1.5
  baseline: null
  padding: 0.0
  detrend: null
reference:
  channels: average
  add_channels: []
```

Strict nested schema:

| Section/type | Fields, defaults, and constraints |
| --- | --- |
| `input` | Required `path`; accepted CLI entry-file suffixes `.fif`, `.fif.gz`, `.edf`, `.bdf`, `.vhdr`, `.set`. No reader kwargs. Related files are loaded by MNE. |
| `output` | Required `directory` and `name`; name must be a single safe filename stem, not a path. Output must not overwrite input or reader companion files. |
| `channels` / `ChannelSettings` | `rename={}`, `types={}`, `drop=()`, `bads=()`, `montage=None`, `interpolate_bads=False`, `bipolar=()`, `projections="error"`. Montage is a standard MNE montage name or `{path: montage.fif}` read by `mne.channels.read_dig_fif`; individualized digitization can be supplied through the input Raw/FIF. Never replace existing digitization unless an explicit montage was supplied. |
| `annotations` / `AnnotationSettings` | `bad_spans=()` containing `{onset, duration, description}` with a BAD-prefixed description and positive duration; onset is seconds from the original acquisition start, before optional cropping. `amplitude=None`, `breaks=None`, `muscle=None`. |
| `amplitude` / `AmplitudeSettings` | At least one of `peak`/`flat` EEG threshold mappings; `bad_percent=5.0`, `min_duration=0.005`. MNE compares successive samples, not epoch peak-to-peak. Require positive thresholds, `0 <= bad_percent <= 100`, and positive duration. Only EEG is detected in this stage. |
| `breaks` / `BreakSettings` | Required `min_break_duration`, `t_start_after_previous`, `t_stop_before_next`; positive minimum, nonnegative margins, sum of margins less than minimum duration. Use resolved task events; reject for fixed-length resting data. |
| `filter` / `FilterSettings` | `l_freq=None`, `h_freq=None`, `notch_freqs=()`. Positive finite cutoffs below Nyquist; high-pass below low-pass; unique ascending positive notch centers with valid stopband/transition support. Fixed MNE FIR design below. |
| `artifact.ica` / `ICASettings` | Present only with `artifact.method=ica`; `l_freq=1.0`, `n_components=None` or integer ≥2, `random_state=42`, `max_iter=1000`, `reject=None`, `flat=None`, `tstep=2.0`, `eog_channels=()`, `ecg_channel=None`. No fractional PCA retention option. Rank below two, excessive requested components, zero usable training samples, or nonconvergence must fail. |
| `epochs` / `EventEpochSettings` | `kind=events`, required `events`, `tmin`, `tmax`; `tmin<tmax`. Common fields below. |
| `events` / event source types | `source=annotations` requires explicit description→integer `event_id`; `source=stim` requires `stim_channel`, explicit label→integer `event_id`, `shortest_event=2`, `min_duration=0.0`; `source=file` requires a path to an MNE events file and explicit `event_id`. API also accepts an explicit integer `(n_events,3)` array. Exactly one source; no search-and-fallback behavior. |
| `epochs` / `FixedEpochSettings` | `kind=fixed`, required `duration>0`, `overlap=0.0` with `0<=overlap<duration`. No events/tmin/tmax keys. Use ID 1 named `segment`; require duration and stride to map to integer samples within floating-point tolerance. |
| Common epoch settings / `EpochCleaningSettings` | `baseline=None` or a finite ordered two-number tuple within the analysis epoch; `padding=0.0` nonnegative seconds; `detrend=None` or `constant`/`linear`. Sampling and rejection have their own sections below. |
| `reference` / `ReferenceSettings` | `channels=None`, `"average"`, or a nonempty tuple of EEG names; `add_channels=()`. Added names must not already exist, must be explicitly documented as acquisition-reference electrodes, and require a final reference choice. No implicit online-reference inference. |

Additional section contracts (strict tagged choices, not arbitrary kwargs):

| Section | Offered choices and validation |
| --- | --- |
| `workflow` | `raw_review=required|disabled`; `epoch_review=optional|required|disabled`. Starter recipes require raw review and offer optional epoch review. `optional` means the numerical path may continue without a manual epoch decision and records that fact; users can invoke it explicitly. Artifact review is required whenever an artifact method is configured. |
| `crop` | None or `{tmin, tmax}` in seconds from acquisition start; inclusive MNE sample bounds, positive retained length, within original duration. Crop all paired data/events consistently. |
| `channels.bipolar` | Sequence of `{name, anode, cathode, type}` for EOG/ECG auxiliary derivations only; unique resulting name, two existing channels of compatible voltage units. Use `mne.set_bipolar_reference(..., drop_refs=False)`; drop originals only via explicit channel settings. No automatic EEG bipolar transformation. |
| `channels.projections` | `error` (default for inactive projectors), `apply`, or `discard-inactive`; never discard already-applied signal changes. |
| `epochs.events.delay` | Optional finite seconds subtracted from event samples to correct a measured trigger delay. Round once on the original grid and record requested/realized shifts. The original uncorrected sample is retained. No drift correction is inferred. |
| `epochs.metadata` | Optional TSV path, one row per original input event in input order; validate count and preserve stable event identity through selection. |
| `annotations.muscle` | None or `{filter_freq: [low, high], threshold, min_length_good}`; explicit EEG type, finite ordered band within source passband/Nyquist; positive threshold and nonnegative gap length. No default 110–140 Hz on a 250 Hz input. |
| `bad_channels` | None or `{method: pyprep, random_state: 42, ransac: false}`. The fixed algorithm calls flat/deviation/correlation detection; high-frequency detection only when explicitly supported by the source bandwidth. If that test is requested through a `methods` list and unsupported, fail. Allow `snr` only together with `high_frequency` and `correlation`, its prerequisites. The parser expands the chosen method list into the resolved configuration. |
| `bridges` | None or `{}` to request MNE bridged-electrode diagnostics with recorded library parameters. Results are electrode pairs for review, not automatic excluded-channel guesses. |
| `stimulation` | None or `{event_ids, channels, tmin, tmax, mode, baseline}`. Support MNE linear/window/constant modes; constant requires its baseline. Windows require valid edge samples; no overlapping repairs. Record repaired spans separately from rejected BAD spans. |
| `artifact` | None or exactly one method: `ica`, `ssp`, `regression`. `reference=None`, `average`, or explicit good EEG names. Match fit and application reference; regression requires an explicit choice. Only that method's settings subsection is accepted. |
| `artifact.ssp` | `eog_channels=()`, `ecg_channel=None`, required positive `n_eeg`, `l_freq=1.0`, `h_freq=35.0`, `tmin=-0.2`, `tmax=0.5`, `reject=None`. At least one artifact channel; validate ordered filter/time bounds against data and total projector count against usable rank. EEG projectors only. Review selects included projector IDs. |
| `artifact.regression` | Required nonempty `eog_channels`, `tstep=2.0`, `reject=None`, `flat=None`. Positive training duration and valid EEG training thresholds. Use MNE EOGRegression; no automatic predictor selection. Review accepts or rejects the complete operator. |
| `rejection` | None; `{method: thresholds, reject, flat, tmin, tmax}`; or `{method: autoreject, n_interpolate, consensus, cv, random_state}`. Thresholds in volts for existing good EEG/EOG/ECG; flat < reject. AutoReject operates on EEG only with valid positions, explicitly configured feasible grids and integer `cv>=2`; fail on too few epochs/electrodes rather than shrinking grids. Manual epoch review is available with every mode. |
| `sampling` | None; `{method: decimate, factor}` with integer factor≥2 and explicit safe low-pass; or `{method: resample, sfreq, padding}` with positive target below input rate and positive epoch padding matching `epochs.padding`. Use MNE polyphase epoch resampling. Never enable both paths or silently change the target rate. |

Example optional automated choices (still requiring review for channel decisions):

```yaml
bad_channels:
  method: pyprep
  methods: [flat, deviation, correlation]
  random_state: 42
  ransac: false
rejection:
  method: autoreject
  n_interpolate: [1, 2]
  consensus: [0.2, 0.5, 0.8]
  cv: 5
  random_state: 42
sampling:
  method: resample
  sfreq: 200.0
  padding: 1.0
```

This sampling fragment requires `epochs.padding: 1.0`; it is an illustrative setting, not evidence that one second is sufficient for every source/target/filter. Validate the actual polyphase support against the configured padding.

Validation rules common to every section:

1. Reject unknown sections/keys, duplicate YAML keys, non-string mapping keys, wrong scalar/container types, booleans supplied as numbers, nonfinite numbers, duplicate channel names, and empty required values. Use a small `yaml.SafeLoader` subclass whose mapping constructor rejects duplicate keys; do not use unsafe YAML constructors.
2. Resolve file paths against the recipe directory with `Path.resolve()`. No environment-variable interpolation, executable expressions, or implicit file search.
3. Reject contradictory stage settings early, then data-dependent errors after loading. Include the dotted configuration path in each error.
4. CLI and Python callers must use the same validation. Dataclass construction alone must not allow invalid ranges to bypass checks.
5. Fixed choices in the implementation are algorithm choices, not extra user-facing knobs. Record their values in provenance. The packaged YAML contains all example study parameters.
6. With `workflow.raw_review=disabled`, explicitly configured bads/spans are still applied, but detector suggestions are only reported, never automatically accepted. The output records `raw_review: disabled`; running detectors alone must not be described as completing review.

ICA-enabled replacement for `artifact: null` (the other artifact methods are mutually exclusive):

```yaml
artifact:
  method: ica
  reference: null
  ica:
    l_freq: 1.0
    n_components: null
    random_state: 42
    max_iter: 1000
    reject: {eeg: 0.0003}
    flat: null
    tstep: 2.0
    eog_channels: [VEOG]
    ecg_channel: ECG
```

Resting-state replacement for `epochs`:

```yaml
epochs:
  kind: fixed
  duration: 2.0
  overlap: 0.0
  baseline: null
  padding: 0.0
  detrend: null
```

## 5. Files and interfaces

All paths below are relative to `/Users/joduq24/Desktop/eegfeat`.

| File | Responsibility |
| --- | --- |
| `src/eegfeat/preprocessing/__init__.py` | Explicit public exports only. |
| `src/eegfeat/preprocessing/stages.py` | Static stage definitions, dependencies, relevant config fields, input/output types. |
| `src/eegfeat/preprocessing/execution.py` | step/next/run-until/status/reset; one shared executor. |
| `src/eegfeat/preprocessing/checkpoints.py` | Native serialization, identities, locks, atomic stage publication, invalidation. |
| `src/eegfeat/preprocessing/quality.py` | Channel/bridge/muscle diagnostics and reviewed raw decisions. |
| `src/eegfeat/preprocessing/artifacts.py` | Explicit SSP/regression implementations and tagged artifact dispatch. |
| `src/eegfeat/preprocessing/rejection.py` | Fixed rejection and autoreject fit/transform with ledgers. |
| `src/eegfeat/preprocessing/sampling.py` | Decimation/resampling/padding/cropping validation. |
| `src/eegfeat/preprocessing/review.py` | Raw/artifact/epoch review files, MNE viewer adapters, headless path. |
| `tests/preprocessing/test_stages.py` | Registry order, dependencies, enablement, unknown-stage errors. |
| `tests/preprocessing/test_execution.py` | step/next/run equivalence, review state, resumption, reset. |
| `tests/preprocessing/test_checkpoints.py` | Round trips, invalidation, tamper detection, interrupted writes, locks. |
| `tests/preprocessing/test_quality.py` | PyPREP/bridge/muscle evidence and prerequisites. |
| `tests/preprocessing/test_artifacts.py` | SSP/regression fit-review-apply equivalence and reference identity. |
| `tests/preprocessing/test_rejection.py` | Autoreject fitting/transform, per-epoch repair, stable event identities. |
| `tests/preprocessing/test_sampling.py` | Target-rate resampling, padding, cropping, sampling provenance. |
| `tests/preprocessing/test_review.py` | GUI adapter boundaries and real headless decisions with stable IDs. |
| `src/eegfeat/preprocessing/config.py` | Settings dataclasses, shared validation, strict YAML loading, path resolution. |
| `src/eegfeat/preprocessing/raw.py` | Copy/validate Raw; channel preparation, annotation preparation, continuous filtering. Helpers above callers. |
| `src/eegfeat/preprocessing/events.py` | Resolve event sources; validate event arrays; fixed-length sample geometry. |
| `src/eegfeat/preprocessing/ica.py` | ICA fitting, scores, exact-fit review validation, application. |
| `src/eegfeat/preprocessing/epochs.py` | Epoch construction, residual rejection, interpolation, reference, decimation, baseline. |
| `src/eegfeat/preprocessing/pipeline.py` | Short orchestration functions and public result types. No file writes in numerical APIs. |
| `src/eegfeat/preprocessing/provenance.py` | Input/fit fingerprints and serializable processing records. |
| `src/eegfeat/preprocessing/io.py` | Read/write preprocessing bundles and ICA review artifacts; overwrite policy. |
| `src/eegfeat/preprocessing/report.py` | MNE HTML QC and ICA-review reports. |
| `src/eegfeat/preprocessing/cli.py` | Preprocessing subcommand registration and handlers. |
| `src/eegfeat/preprocessing/template.yaml` | Complete commented example configuration; scientific units explicit. |
| `tests/preprocessing/conftest.py` | Deterministic raw, event, artifact, and montage fixtures. |
| `tests/preprocessing/test_config.py` | Schema/range/path/optional dependency validation. |
| `tests/preprocessing/test_raw.py` | Preparation, annotations, filtering, immutability. |
| `tests/preprocessing/test_events.py` | Original sample timing and event-source errors. |
| `tests/preprocessing/test_ica.py` | Fit/review/apply lifecycle, rank, stale fits, artifact removal. |
| `tests/preprocessing/test_epochs.py` | MNE equivalence, rejection, reference, interpolation, baseline, decimation. |
| `tests/preprocessing/test_pipeline.py` | Full numerical workflow and independent signal checks. |
| `tests/preprocessing/test_io.py` | Round trips, provenance, interrupted writes, collision protection. |
| `tests/preprocessing/test_report.py` | Headless HTML report content and explicit dependencies. |
| `tests/preprocessing/test_cli.py` | Init/check/fit/run integration and feature-runner handoff. |
| `tests/validation/test_preprocessing.py` | Opt-in real raw EEG validation without changing existing loaders. |
| `docs/guides/preprocessing.rst` | Workflow, units, review, limitations, complete examples, citations. |
| `docs/api/preprocessing.rst` | Small public API reference. |
| `examples/preprocessing.yaml` | Runnable configuration after setting an actual recording path. |

Modify `src/eegfeat/runner/cli.py` only to register the new command. Also update `pyproject.toml`, `README.md`, `docs/install.rst`, `docs/index.rst`, `docs/api/index.rst`, `.github/workflows/ci.yml`, and relevant packaging tests. Add one feature-runner integration test; do not refactor feature computation or modeling.

Public interfaces are explicit at both levels:

```text
load_config(path: str | Path) -> PreprocessingConfig
open_workflow(config: PreprocessingConfig) -> Workflow
list_steps(workflow: Workflow) -> tuple[StepStatus, ...]
run_step(workflow: Workflow, stage: str) -> StepResult
run_next(workflow: Workflow) -> StepResult
run_until(workflow: Workflow, stage: str = "export") -> RunOutcome
read_checkpoint(workflow: Workflow, stage: str) -> Checkpoint
reset_from(workflow: Workflow, stage: str) -> tuple[str, ...]
```

`Workflow` contains explicit validated config and workspace paths, not mutable hidden “current Raw” state. Each call reads and validates its declared predecessor artifacts. `StepResult` identifies stage, state, artifact paths, QC summary, and next action. `RunOutcome` distinguishes complete, needs-review, and failed; it never disguises review as successful export.

Direct numerical functions accept typed settings and MNE objects/results and return new objects; no file writes or global settings. Export each offered operation by a clear verb: `prepare_channels`, `resolve_events`, `crop_raw`, `annotate_raw`, `detect_bad_channels`, `detect_bridges`, `apply_raw_review`, `repair_stimulation`, `notch_raw`, `filter_raw`, `reference_artifact_data`, `fit_ica`, `fit_ssp`, `fit_eog_regression`, `review_artifact`, `make_epochs`, `apply_artifact`, `fit_rejection`, `reject_epochs`, `apply_epoch_review`, `interpolate_channels`, `reference_epochs`, `resample_epochs`, `crop_epochs`, `detrend_epochs`, `baseline_epochs`, `build_report`, and `write_result`. `preprocess` is an in-memory convenience orchestrator over those functions, not a second implementation.

Use small results for distinct boundaries: `PreparedRaw` carries Raw plus original/current source metadata; `EventData` carries events/event_id/metadata/original sampling rate, `original_row`, and original-to-corrected event identity; `QualityCandidates` carries channel/pair/span suggestions; `ArtifactModel` is a tagged ICA/SSP/regression model with exact training identity; `ReviewedArtifact` binds a decision to that model; `RejectionModel` is an autoreject model and fitting identity; `PreprocessingResult` carries final epochs and complete ledgers/QC/provenance. `ProcessingSettings` groups the validated numerical sections; `PreprocessingConfig` adds input/output/workflow settings. Do not couple unrelated numerical functions to the entire configuration object.

Sequential Python usage must be as short as the CLI:

```python
from eegfeat.preprocessing import load_config, open_workflow, run_next, run_until

workflow = open_workflow(load_config("preprocessing.yaml"))
first = run_next(workflow)
raw_review = run_until(workflow, "review-raw")
print(raw_review.state, raw_review.next_action)
```

For full programmatic control, expose the same numerical functions directly. Native MNE viewers remain available on returned/checkpoint objects. A user must never need private attributes to mark channels, apply a reviewed artifact model, or pass a completed stage to the next one.

Numerical functions raise `TypeError`/`ValueError` with context and propagate unexpected library errors. File functions propagate `OSError`. CLI handlers catch only these expected boundary failures. Do not adopt the existing batch runner's broad per-recording exception catch in this single-recording command.

## 6. Implementation tasks

For each task: add its scientific/contract tests first, run the named test file to observe the expected missing-feature failure, implement the listed behavior, and rerun to green. Commit coherent tasks only after their checks pass; do not commit unrelated working-tree changes.

### Task 1: Establish fixtures and dependency boundaries

**Files:** Create `tests/preprocessing/conftest.py`, package `__init__.py`; modify `pyproject.toml` and `tests/test_packaging.py`.

- [ ] Record the working-tree state and baseline test results before implementation.
- [x] Add `preprocessing = ["PyYAML>=6.0", "scikit-learn>=1.3", "h5io", "h5py", "filelock"]` to optional dependencies. Add `preprocessing-auto = ["pyprep>=0.9.0", "autoreject>=0.4.2"]`, `preprocessing-gui = ["mne-qt-browser", "PyQt6"]`, and `types-PyYAML` to development dependencies. Install extras explicitly in their dedicated CI jobs. Import YAML only inside file parsing and sklearn-dependent ICA work only when requested. Base `import eegfeat` and the existing commands must not import the extra.
- [x] Create a deterministic 30-second, 250 Hz RawArray with `Fp1`, `Fp2`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `VEOG`, `ECG`, and `STI`. Use standard_1020 geometry for EEG; seed 42; independent small noise plus 10 Hz and 20 Hz EEG signals, and a separate 60 Hz contamination fixture. Use event samples `first_samp + [1250, 2500, 3750, 5000, 6250]` with event code 1. Provide variants with `first_samp=1000`, a known BAD interval, one bad EEG channel, and one amplitude-spike epoch.
- [x] Use a separate ≥90-second fixture for 0.1 Hz high-pass tests, whose FIR can exceed the 30-second fixture. Set each successful annotated-filter test's contiguous clean segments longer than its computed filter support. Keep a short-segment fixture specifically for failure testing.
- [x] Add a separate full-rank non-Gaussian mixture fixture for ICA; do not try to test component identification on identical sinusoidal channels.
- [x] Test core import in a subprocess that blocks `yaml` and `sklearn` imports; requesting YAML/ICA must produce the explicit installation instruction `pip install 'eegfeat[preprocessing]'`. Catch only a missing optional package, not arbitrary import failures inside it.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/test_packaging.py tests/test_public_api.py -q
```

Expected after implementation: existing public API remains unchanged and optional dependency boundary tests pass.

### Task 2: Implement strict configuration and validation

**Files:** Create `preprocessing/config.py`, `template.yaml`, and `tests/preprocessing/test_config.py`.

- [x] Implement exactly the settings/types and defaults in section 4; give every variant a single responsibility. Keep event/fixed epoch variants distinct so invalid mixed fields cannot reach computation.
- [x] Write parameterized failure cases for unknown/duplicate keys, string thresholds, booleans as integers, NaN/infinity, negative cutoffs, empty event mapping, duplicate event codes, unknown epoch kind, conflicting event fields, invalid overlaps, reference strings other than `average`, malformed bad spans, and path traversal in output names.
- [x] Implement a duplicate-key-rejecting SafeLoader, then small explicit section parsers. Do not build a generic schema framework or accept extra kwargs.
- [x] Verify both direct settings construction and YAML parsing reject the same invalid numeric values. Validate bad/ref/montage names against data later in Task 3.
- [x] Load the complete packaged example through `importlib.resources`; check paths resolve relative to the YAML file, independent of cwd.

Representative executable regression test:

```python
def test_unknown_filter_key_fails(tmp_path):
    from eegfeat.preprocessing import load_config

    path = tmp_path / "preprocessing.yaml"
    path.write_text(
        "input: {path: raw.fif}\n"
        "output: {directory: out, name: subject}\n"
        "filter: {highpass: 1.0}\n"
        "epochs: {kind: fixed, duration: 2.0}\n"
    )
    with pytest.raises(ValueError, match=r"filter.highpass"):
        load_config(path)
```

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_config.py -q
```

Expected: invalid settings fail at the configuration boundary, before data or output creation.

### Task 3: Prepare channel metadata and input quality

**Files:** Create `preprocessing/raw.py`; add `test_raw.py`.

- [x] Validate `isinstance(raw, mne.io.BaseRaw)`, positive finite sampling rate, nonzero time samples, EEG presence, and finite physiology samples in bounded chunks. Reject nonfinite input even in a bad span; never impute or replace it with zero. Copy before calling `load_data`.
- [x] Rename → set channel types → derive declared bipolar auxiliaries → drop explicitly requested channels → attach montage → merge bad names, preserving original labels. Reject missing requested names and rename collisions. Reject removal of a required event/artifact/reference channel.
- [x] Implement the explicit inactive-projector policy (error/apply/discard-inactive); default to error. Preserve applied-projector information and the original reference/filter metadata. Add declared bipolar EOG/ECG channels without dropping source channels implicitly. Support existing or explicitly supplied FIF digitization.
- [x] For requested interpolation or ICA topographies, validate finite nonzero EEG coordinates and sufficient usable geometry; let MNE surface numerical geometry errors. Do not guess a standard montage from channel labels. Basic filtering/epoching remains valid without a montage.
- [x] Verify no operation changes the caller's samples, info, channel names, annotations, preload state, or bad list. Public output objects must not alias mutable caller metadata.

Core channel-only operations for `ChannelSettings` are below. Add the explicitly configured bipolar derivation before dropping source channels, and handle a montage file with `read_dig_fif` in the same montage helper. Do not silently ignore the file variant:

```python
working = raw.copy().load_data()
working.rename_channels(dict(settings.rename))
working.set_channel_types(dict(settings.types))
if settings.drop:
    working.drop_channels(list(settings.drop))
if isinstance(settings.montage, str):
    working.set_montage(settings.montage, on_missing="raise")
working.info["bads"] = list(
    dict.fromkeys([*working.info["bads"], *settings.bads])
)
```

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_raw.py -q
```

Expected: preparation preserves the input and errors identify the offending channel or assumption.

### Task 4: Resolve events and annotate artifacts before filtering

**Files:** Create `preprocessing/events.py`, `test_events.py`; extend `raw.py` and `test_raw.py`.

- [x] Implement explicit annotation/stim/file event readers. Use `mne.events_from_annotations(..., event_id=..., use_rounding=True)`, `mne.find_events(..., stim_channel=..., shortest_event=..., min_duration=...)`, or `mne.read_events(path)` respectively. No automatic alternate source.
- [x] Validate integer shape `(n, 3)`, nonempty events, strictly increasing sample locations, no repeated samples, selected event-code presence, and sample indices inside `[first_samp, first_samp+n_times)`. Reject aliases/duplicate event codes in the initial API. Keep original event rows and their metadata; let unwanted event codes appear as IGNORED in the full event ledger. If cropping selects an event subset, retain an explicit original-row map: original IDs are `events.original_row[epochs.selection]`, not a fresh row enumeration.
- [x] Validate metadata row count equals input event count; reset its DataFrame index on a copy. Final selection must be obtained from MNE, not matched by rounded event time.
- [x] Create fixed events with `mne.make_fixed_length_events(..., first_samp=True)` using sample-aligned duration and stride. Store epoch endpoint `tmax=(n_samples-1)/sfreq`, not `duration`, to avoid an extra shared sample.
- [x] Append manual BAD spans with MNE time conversion. Construct a temporary Raw copy carrying only the new relative annotations with `orig_time=None`, let `set_annotations` normalize to its time origin, then append the normalized fields directly to `working.annotations`. Do **not** pass the combined, already-normalized annotations back through `working.set_annotations`: with `orig_time=None` this can add the recording offset twice. Reject spans outside the original acquisition before MNE can silently clip them. After an intentional crop, explicitly intersect valid acquisition-relative spans with the retained range and log the intersection/exclusion. Convert their absolute acquisition sample bounds to cropped-Raw-relative seconds before normalization. Test with/without measurement dates and nonzero first samples. Use the same normalization/append helper for MNE-generated amplitude/break/muscle annotations, honoring their returned `orig_time`; those detector outputs already refer to the current Raw, so do not reapply acquisition-to-crop conversion.
- [x] When amplitude annotation is configured, call `mne.preprocessing.annotate_amplitude` on explicitly selected good EEG channels, record returned annotations as candidates, and record returned bad channels as candidates. The review step merges explicitly accepted candidates; preserve previous bads unless review explicitly changes them. Show that `peak` describes adjacent-sample change; epoch PTP thresholds belong elsewhere.
- [x] When breaks are configured, use `mne.preprocessing.annotate_break(raw, events=events.events, ...)` with the explicitly selected task events. Reject break detection for fixed-length epochs. Record its annotations as candidates for raw review, preserving BAD and EDGE boundaries already present. Existing/manual annotations are already explicit decisions; automatic break suggestions require the same raw-decision handling as amplitude/muscle suggestions.
- [x] Test that original annotations retain their times/descriptions, BAD spans reject overlapping epochs, and neither channel selection nor annotation changes shift event samples.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_events.py tests/preprocessing/test_raw.py -q
```

Expected: exact original sample identity, no silent event dropping, and annotations aligned under every tested time origin.

Verified manual-span append operation (new annotations are already validated in bounds):

```python
normalized = working.copy().set_annotations(new_annotations).annotations
working.annotations.append(
    onset=normalized.onset,
    duration=normalized.duration,
    description=normalized.description,
    ch_names=normalized.ch_names,
)
```

Here `new_annotations` already uses the current (possibly cropped) Raw's relative time origin; acquisition-relative manual times were converted by the preceding step. The temporary copy is for MNE's public time normalization, not another signal-processing stage. Release it immediately. Existing annotations are never renormalized. The regression must assert an onset of `5.0 + raw.first_time` for a span beginning five seconds into a recording, both with and without `meas_date`.

### Task 5: Complete raw selection and quality assessment

**Files:** Extend `raw.py`, `events.py`; create `quality.py`, `test_quality.py`; extend configuration/event tests.

- [x] Implement acquisition-relative cropping after resolving events. Use MNE crop on a copy, preserve its new `first_samp`, and mark excluded event rows as `OUTSIDE_CROP` in the original ledger. Never renumber input-event identities. Test events on both crop boundaries, nonzero original first sample, and negative/too-large crop requests.
- [x] Implement declared trigger-delay correction before cropping and epoching. Preserve uncorrected and corrected samples, requested delay, and realized rounded delay. Fail if correction creates repeated/out-of-range events. Align metadata to original input rows before selection.
- [x] Add channel/segment candidates without automatically accepting them. Raw review can explicitly add/remove candidates and amend existing bad labels/spans; every change records its source and decision. Editing reviewed input invalidates descendants, not the original recording.
- [x] PyPREP uses the published 0.9.0 constructor with a fixed seed, good measured EEG, `do_detrend=True` on a disposable diagnostic copy, and `reject_by_annotation="omit"`. Call explicit selected methods rather than `find_all_bads` if that would run unrequested tests. Preserve per-method bad lists and parameters. Default methods are flat, deviation, and correlation. SNR requires both high-frequency and correlation tests. RANSAC is explicit, seeded, and requires sufficient valid electrode geometry.
- [x] Explain that PyPREP's omission of BAD spans concatenates clean data for diagnostic evaluation only; it never changes the analysis Raw or original event times. Record the clean intervals used. Compare detector results to a direct PyPREP call under the same omission policy. If the selected methods cannot operate on the available clean duration/channel set, fail clearly.
- [x] Run MNE muscle detection on the pre-analysis-low-pass Raw with `ch_type="eeg"` and user-specified band/threshold; preserve returned scores and candidate annotations. The source header's filter history and Nyquist must permit that diagnostic band. A low-rate recording does not automatically receive a different muscle band.
- [x] Run bridge diagnostics with `mne.preprocessing.compute_bridged_electrodes` on a diagnostic copy; store channel-pair results and electrical distances. Do not infer which of two bridged channels to remove or interpolate. Let review decide.
- [x] Implement optional `repair-stim` using `mne.preprocessing.fix_stim_artifact` before notch/filter. Require explicit event IDs, measured physiology picks, and valid nonoverlapping windows. Prevalidate needed neighboring/baseline samples, not just event centers. Record repaired sample spans and method as reconstructed samples, not clean measured signal. Do not modify stim/auxiliary event channels or global event timing.
- [x] Test candidate-vs-decision separation, preservation of original bads/annotations, direct-library detector equivalence, unsupported bands, impossible RANSAC geometry, and direct-MNE stimulation-repair agreement. Inject a known flat/noisy channel and a high-frequency burst into suitable ≥500 Hz synthetic data; do not test a 110–140 Hz detector at 250 Hz.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_quality.py tests/preprocessing/test_events.py -q
```

Expected: all raw preparation and quality steps are available independently, with inspectable evidence and no automatically accepted detector suggestions.

### Task 6: Implement continuous FIR filtering and anti-aliasing validation

**Files:** Extend `raw.py`, `test_raw.py`; add filtering tests in `test_pipeline.py`.

- [x] Select EEG/EOG/ECG channel names explicitly, including marked bad channels whose data may still be inspected; exclude stim and misc. Require at least one good EEG channel separately. Filtered bad channels remain marked bad.
- [x] Use MNE's zero-phase FIR, Hamming window, `firwin` design, `filter_length="auto"`, automatic transition bands, and `pad="reflect_limited"`. Set `skip_by_annotation=("edge", "bad")` for both notch and bandpass. Pass `n_jobs` explicitly; validate it as a nonzero integer supported by MNE.
- [x] Validate notch stopbands including default notch width and transition support against Nyquist. Do not silently remove out-of-range harmonics. Apply only the user-listed centers, once.
- [x] Precompute requested FIR coefficients with `mne.filter.create_filter` using the same parameters and sample rate. Record coefficient hashes, tap counts, cutoff values, and transition widths. Reject any nonempty filterable annotation segment shorter than its required filter; do not suppress MNE warnings or substitute an IIR/shorter filter.
- [x] For later integer decimation, require an explicit `h_freq` in this run. Let `target=sfreq/decim` and `transition=min(max(h_freq*0.25, 2.0), sfreq/2-h_freq)` for the selected MNE auto transition rule. Require both `h_freq <= target/3` and `h_freq+transition < target/2`. This deliberately conservative policy avoids trusting only an input header's low-pass number. Check equality/rounding boundaries in tests.
- [x] Keep decimation and target-rate polyphase resampling as mutually exclusive sampling methods. The sampling task below implements the target-rate path with explicit padded epochs and final cropping.

Exact filtering calls, after validation and coefficient/segment checks:

```python
if settings.notch_freqs:
    working.notch_filter(
        freqs=list(settings.notch_freqs), picks=physiology_channels,
        method="fir", phase="zero", fir_window="hamming",
        fir_design="firwin", filter_length="auto", pad="reflect_limited",
        skip_by_annotation=("edge", "bad"), n_jobs=n_jobs,
    )
if settings.l_freq is not None or settings.h_freq is not None:
    working.filter(
        l_freq=settings.l_freq, h_freq=settings.h_freq,
        picks=physiology_channels, method="fir", phase="zero",
        fir_window="hamming", fir_design="firwin", filter_length="auto",
        l_trans_bandwidth="auto", h_trans_bandwidth="auto",
        pad="reflect_limited", skip_by_annotation=("edge", "bad"),
        n_jobs=n_jobs,
    )
```

- [x] Check numerical equivalence against independent direct MNE calls on the same synthetic raw; test unchanged stim values, sfreq, first_samp, event samples, and sample count.
- [x] Add a long sinusoid test: away from FIR support at the ends, require >20 dB reduction of an injected 60 Hz component after a 60 Hz notch and <5% change in 10 Hz amplitude. Specify the analysis slice from the actual filter length; do not hard-code a too-short transient exclusion.
- [x] Test two clean segments separated by a BAD interval independently against MNE's annotated filtering. Include an interval too short for the requested filter and require a contextual error.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_raw.py tests/preprocessing/test_pipeline.py -q
```

Expected: direct-MNE agreement, measurable artifact attenuation, and unsafe decimation/filter lengths rejected.

### Task 7: Fit ICA with reproducible training and explicit artifact suggestions

**Files:** Create `ica.py`, `provenance.py`, `test_ica.py`.

- [x] Build `ArtifactModel` and its data fingerprint before fitting. Hash prepared sample values in bounded chunks, channel order/types/locations, sfreq/first_samp, bad labels, annotations/time origin, reference/projector state, preparation settings, ICA settings, and relevant package versions. Use canonical JSON with `allow_nan=False` for serializable metadata; normalize missing location metadata explicitly rather than encoding NaN. Do not hash only a path or mtime.
- [x] Form training data from a copy of the prepared analysis-band Raw. Its high-pass must be at least `artifact.ica.l_freq`; apply only the additional necessary high-pass. Never lower an existing high-pass or claim to restore removed frequencies. The analysis object keeps its original analysis passband.
- [x] Select good EEG only. Estimate numerical rank on the actual finite, annotation-excluded, amplitude-accepted training data with `mne.compute_rank`, using explicit EEG picks and `proj=False`. Use the same training segments for rank estimation and fitting. The clean implementation is fixed-duration MNE training Epochs of `tstep`, baseline=None, proj=False, reject_by_annotation=True, with configured training reject/flat thresholds and no decimation.
- [x] With `n_components=None`, use the estimated usable EEG rank; require rank ≥2. With an integer, require 2 ≤ requested components ≤ rank. ICA reconstruction must retain the remaining PCA dimensions, rather than introducing an unrequested PCA reduction.
- [x] Use FastICA, fixed random state, finite maximum iterations, and a targeted sklearn `ConvergenceWarning`→error conversion local to the fit call. Record the number of accepted/rejected training segments and estimated rank. Do not suppress unrelated warnings. Require at least one usable complete training segment and enough samples for the fitted dimensions; warn visibly in QC about limited training duration without pretending a universal minimum guarantees a good decomposition.

Core fit call after training selection/rank validation:

```python
ica = mne.preprocessing.ICA(
    n_components=n_components,
    method="fastica",
    random_state=settings.random_state,
    max_iter=settings.max_iter,
)
with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)
    ica.fit(training_epochs, picks=good_eeg_channels, reject_by_annotation=True)
```

- [x] Score only explicitly requested real EOG/ECG channels using MNE `find_bads_eog`/`find_bads_ecg`. Reject missing, bad, or incorrectly typed requested channels. Use `find_bads_ecg(..., method="correlation", ch_name=...)` to avoid synthetic-ECG inference for EEG-only data. Record method/threshold/score for each candidate; leave `ica.exclude` empty. No detections is a legitimate result, not an error.
- [x] Add tests for repeatable reconstruction, input immutability, no changes to analysis high-pass, exclusion of bad electrodes/annotations, rank-deficient data, stale fingerprints, and nonconvergence. Compare reconstructed samples rather than assuming independently fitted ICA component signs/order are scientifically fixed.
- [x] Add an artifact-injection test using the non-Gaussian fixture. Select the known artifact component by absolute correlation to the injected source for the **test only**, remove it, and verify reduced artifact contribution with retention of an independent neural source. Production exclusions still require review.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_ica.py -q
```

Expected: reproducible fits on valid data; no automatic component removal; failures never fall back to uncorrected “success.”

### Task 8: Implement exact-fit ICA review and persistence

**Files:** Extend `ica.py`, `provenance.py`; create `io.py`; extend `test_ica.py` and create `test_io.py`.

- [x] `review_artifact(fit, decision=...)` validates a unique integer sequence in `[0, ica.n_components_)`; reject booleans and removal of every component. Copy the fitted ICA when producing reviewed state. The ICA decision contains `exclude`; SSP contains `include`; regression contains `apply`. Use separate decision dataclasses rather than interpreting one flag differently across methods. An empty tuple is valid and must preserve data exactly by skipping `ICA.apply`, avoiding unnecessary reconstruction roundoff.
- [ ] Save the fitted model as `<name>_ica.fif`, training/preparation identity as `<name>_ica.json`, and a machine-generated `<name>_ica-review.yaml`. The review file references the SHA-256 of the saved ICA file and the preparation/fit identity, with `exclude: null` initially.
- [x] Include EOG/ECG candidate indices and scores in the fit JSON/report; do not prefill the review exclusions. `exclude: null` blocks final output; `exclude: []` explicitly approves keeping all components.
- [x] On final run, load the saved model and review, recompute input/preparation identity, and verify the model hash, channel order, bad set, reference state, ICA settings, and supported environment match. Never refit and then apply old component indices. Changed input/filter/channel/annotation/ICA settings invalidate the review. Changed output directory or final baseline need not invalidate the fit.
- [x] Add round-trip tests that change a raw sample, annotation, bad label, channel order, fit setting, ICA file, or review hash and require failure. Prove that final execution never calls `ICA.fit`.

Generated review-file shape:

```yaml
fit_id: "sha256 digest generated from the training identity"
ica_sha256: "sha256 digest generated from the saved ICA file"
exclude: null
```

Those digest values are generated data, not literal template defaults. Users edit only `exclude` after inspection.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_ica.py tests/preprocessing/test_io.py -q
```

Expected: model identity survives save/load; stale or unreviewed exclusions block ICA application.

### Task 9: Offer SSP and EOG regression as artifact alternatives

**Files:** Create `artifacts.py`, `test_artifacts.py`; extend `config.py`, `review.py`, `io.py`.

- [x] Represent artifact settings/models as tagged ICA/SSP/regression variants. Use a small explicit method-to-handler map, not a generic plugin framework. Exactly one correction method is selected for this workflow; do not stack ICA/SSP/regression automatically.
- [x] Implement the optional `artifact-reference` operation on a copy before fitting and epoching. For regression, require an explicit average/named reference in configuration, consistent with MNE's EEG regression instructions. Validate all reference channels and fingerprint the resulting reference state. Epoch rejection thresholds refer to this state; final reference is separate.
- [x] Fit EEG-only SSP using MNE `compute_proj_eog`/`compute_proj_ecg`, explicit real artifact channels, `n_mag=0`, `n_grad=0`, configured `n_eeg`, and declared detection filters/epochs/rejection. Store each candidate vector's identity, channel order, variance/explained summaries, and contributing artifact events. Reject unavailable detection channels and empty training events.
- [x] Review chooses explicit included SSP vector IDs. Save/read with MNE projector I/O. On application, add only those reviewed projectors to a copy of the matching epochs and call `apply_proj` once; prove previously applied projectors/reference are consistent. An empty selected set is an explicit no-correction decision.
- [x] Fit `mne.preprocessing.EOGRegression(picks=good_eeg, picks_artifact=eog_names, proj=False)` on full-rate, unbaselined, referenced training epochs, excluding bad spans and configured extreme training segments. Save/load using MNE EOG-regression HDF5 APIs. No pseudoinverse/regression reimplementation.
- [x] Regression review records `apply: true|false` against the exact model hash; application reuses that model on compatible epochs without a refit. Export coefficients/channel identities/reference and removed-variance QC. Regression is not claimed to perfectly distinguish neural activity correlated with the EOG signal.
- [x] Extend exact-fit invalidation and model persistence tests to all methods. Compare SSP-corrected/regressed signals against independent direct MNE calls. Inject a known shared EOG contribution and verify attenuation plus unchanged auxiliary predictors. Test a mismatched reference, bad list, or channel order as an error.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_artifacts.py tests/preprocessing/test_ica.py -q
```

Expected: each offered artifact strategy has a complete fit → inspect/review → apply path and identical data assumptions at fit/application.

### Task 10: Construct epochs, apply reviewed artifacts, and reject residual artifacts

**Files:** Create `epochs.py`, `test_epochs.py`; extend `pipeline.py` as orchestration is introduced.

- [x] Build both epoch variants through one small internal MNE constructor, using explicit sample-derived fixed duration where applicable. Construction bounds include padding; compute `rejection_tmin`/`rejection_tmax` from the requested analysis bounds and rejection settings, not from padding. Retain input channel order, including bad channels, so later interpolation remains possible; use an explicit channel-name list rather than MNE's default picks that can omit bad channels.

```python
epochs = mne.Epochs(
    raw, events.events, event_id=dict(events.event_id),
    tmin=tmin, tmax=tmax, baseline=None, picks=list(raw.ch_names),
    preload=True, reject=None, flat=None, proj=False, decim=1,
    reject_tmin=rejection_tmin, reject_tmax=rejection_tmax,
    detrend=None, on_missing="raise", reject_by_annotation=True,
    metadata=events.metadata, event_repeated="error",
)
```

- [x] Preserve MNE's complete `drop_log` and `selection`, including IGNORED, NO_DATA/TOO_SHORT, and BAD annotation reasons. Reject malformed events earlier; valid events whose requested epoch extends outside the recording are recorded as dropped, not clipped or padded.
- [x] Require `ReviewedArtifact` exactly when an artifact method was configured. Verify its prepared-data identity; reject a supplied model when `artifact` is disabled or the model method differs from the configured method. Apply on epochs before baseline correction or final reference changes; compare against direct MNE application.
- [x] For fixed-threshold rejection, call `epochs.drop_bad(reject=..., flat=...)` after the selected artifact correction on full-rate data. Bad channels are excluded from threshold decisions. If both high/low PTP thresholds are configured, require flat < reject for the same type. For fixed-threshold rejection, require final limits at least as stringent as configured artifact-training limits for corresponding EEG types; reject incompatible settings. Autoreject learns its own limits: report any difference from training limits, without silently clamping them or claiming the same threshold contract.
- [x] Fail with a clear summary if no epochs remain at construction or after amplitude rejection. Distinguish this from a valid run that drops some epochs.
- [x] Test exact selected event samples, metadata alignment, full-rate spike rejection, annotated rejection, explicit empty ICA review, and events at both recording edges. Verify fixed epochs have exactly `round(duration*sfreq)` samples and no accidental extra shared sample.

Representative independent rejection oracle:

```python
expected = mne.Epochs(
    raw, events, {"stimulus": 1}, tmin=-0.2, tmax=0.8,
    baseline=None, proj=False, preload=True,
    reject={"eeg": 150e-6}, reject_by_annotation=True,
)
np.testing.assert_array_equal(result.epochs.selection, expected.selection)
np.testing.assert_array_equal(result.epochs.events, expected.events)
assert result.epochs.drop_log == expected.drop_log
```

Use this oracle only for the matching test configuration with no ICA, interpolation, reference, or downsampling; other tests build the corresponding independent sequence explicitly.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_epochs.py -q
```

Expected: exact MNE drop/event agreement and explicit failure when all epochs are lost.

### Task 11: Add autoreject and manual epoch review

**Files:** Create `rejection.py`, `test_rejection.py`; extend event ledger, `review.py`, and `io.py`.

- [x] Keep fixed thresholds and autoreject as exclusive automatic rejection modes. Manual exclusion remains independently available. None disables automatic rejection without implying manual review occurred.
- [x] Fit `autoreject.AutoReject` with explicit good EEG picks, feasible `n_interpolate`/`consensus` grids, configured integer CV folds and seed. Check the number of usable epochs/electrodes against the requested grid; do not quietly reduce the folds or interpolation counts. Persist via the library's HDF5 save/read functions, with exact training identity.
- [x] Fit rejection on the configured analysis interval at full sample rate, using an analysis-window crop of the padded corrected epochs. Obtain its RejectLog on that same window. Pass that explicit log into transform of the corresponding padded epochs so channel repairs extend through padding without changing which samples drove rejection. Test that the supported autoreject version permits this call and yields the same retained event IDs; do not silently evaluate a different window.
- [x] Keep globally bad EEG excluded from autoreject's fit/picks. Its transient per-epoch interpolation does not clear globally bad labels or replace the later global interpolation stage.
- [x] Save the fitted thresholds, CV choice, bad-epoch mask, and each channel/epoch label (good, bad, interpolated). Map all outcomes back through input `selection`, not through current row numbers. Export a long-form repair TSV alongside the event ledger.
- [x] Manual review decisions contain original event IDs and checkpoint identity. `epochs.plot` is offered for interactive review; headless files specify the same IDs. A changed epoch definition/automatic rejection result invalidates the old review.
- [x] Test fit and transform separately to allow training-only fitting. Test direct-autoreject equivalence, deterministic seeds, inadequate data errors, bad-channel handling, padding-window consistency, all-rejected failure, and metadata identity after automatic plus manual drops.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_rejection.py tests/preprocessing/test_epochs.py -q
```

Expected: fixed, automated, and manual paths produce auditable epoch/channel decisions, and fitted rejection never leaks across an unrequested fit scope.

### Task 12: Interpolate, reference, decimate, and baseline-correct

**Files:** Extend `epochs.py`, `test_epochs.py`.

- [x] Capture original EEG bad names. For requested interpolation, temporarily select only those names as interpolation targets, call `epochs.interpolate_bads(reset_bads=True, method={"eeg": "spline"})`, then restore unrelated bad labels. Do not accidentally interpolate EOG/ECG/MEG with MNE's defaults. Fail on insufficient geometry; no dropping channels as a substitute.
- [x] If interpolation is disabled, retain bad labels for the existing feature runner to exclude. Report exactly which channels were repaired versus still bad.
- [x] Add explicitly requested missing acquisition-reference electrodes with `mne.add_reference_channels(..., copy=False)` after interpolation/rejection. Apply an explicitly supplied montage again to locate newly added electrodes; preserve individualized geometry through MNE where available. Do not require geometry merely to subtract a named reference, but report missing positions honestly.
- [x] Require at least two good EEG channels for an average reference. For named references require every name is retained EEG and not still bad. Apply `epochs.set_eeg_reference(ref_channels=..., projection=False)` and leave EOG/ECG/stim unchanged. Retain reference electrodes in the saved EEG channel set.
- [x] After Task 6's anti-alias checks, call `epochs.decimate(sampling.factor)` for `sampling.method=decimate`. Preserve `events[:,0]` as original acquisition samples; write `event_sample_sfreq` separately from final `epochs.info["sfreq"]`. Do not claim those event samples index the decimated data.
- [x] Validate baseline on the actual final `epochs.times`, then call `epochs.apply_baseline(cleaning.baseline)` only when configured. Record the actual included sample bounds.
- [x] Test direct-MNE interpolation equivalence, average-reference sum near zero on good EEG, named reference subtraction, bad-channel exclusion, inactive projectors, missing original reference restoration, preservation of auxiliary channels, final sample count, and baseline mean near zero within numerical tolerance.
- [x] Add a regression where a one-sample spike would disappear under decimation. Verify it was rejected before downsampling. Add unsafe low-pass/decimation combinations and require failure before any output.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_epochs.py -q
```

Expected: final signals and metadata agree with the corresponding MNE sequence; no silent reference or sampling changes.

### Task 13: Offer target-rate resampling, padding removal, and detrending

**Files:** Create `sampling.py`, `test_sampling.py`; extend `epochs.py`, `config.py`, and provenance.

- [x] `sampling=None` changes no sample rate. Decimation uses the explicit FIR low-pass guard already specified. Target-rate resampling uses `Epochs.resample(..., method="polyphase")` with fixed recorded window/padding choices and a lower positive output sampling rate. Retain one code path across supported MNE versions.
- [x] Epoch construction includes `epochs.padding` on both sides; the analysis interval remains separately defined. Events without enough recording support for the padded epoch are dropped with the boundary reason. Do not fabricate padding in the pipeline or borrow samples across BAD discontinuities.
- [x] Validate configured epoch padding against the actual polyphase FIR support for the realized input/output lengths and recorded window. Establish the support bound from the documented MNE/SciPy filter construction and test with impulse/edge responses for supported versions; do not trust an arbitrary one-second recommendation or use private MNE helpers. If a version's public/default filter behavior cannot meet this contract, fail that integration check instead of silently reducing padding.
- [x] All residual epoch rejection occurs at the original rate before this stage. Resampling includes MNE's anti-alias filter; do not apply the decimation-only `h_freq <= target/3` condition to polyphase resampling as if it were unfiltered sample dropping.
- [x] Crop padding after resampling/decimation using requested analysis tmin/tmax and record actual returned bounds/sample count. Preserve acquisition event sample coordinates; quantify any output time-grid displacement and require it to be within the documented half-output-sample rounding tolerance. Fail inconsistent grids rather than rewriting events.
- [x] Offer constant/linear detrending on EEG only with `scipy.signal.detrend` through the MNE public data-operation interface, after sampling/cropping and before optional baseline. Retain time axis/channel order and auxiliary data. Keep it off by default; explain its effect on low-frequency/ERP amplitudes.
- [x] Tests compare direct MNE/SciPy calls, 250→200 Hz and 512→250 Hz cases, passband amplitude, stopband alias suppression, padded edge effects, original-event preservation, final-grid baseline, and one-shot/stepwise equivalence. Reject insufficient padding and impossible target rates.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_sampling.py tests/preprocessing/test_epochs.py -q
```

Expected: both offered downsampling methods have explicit anti-aliasing/time-grid contracts, and padding/detrending are individually runnable stages.

### Task 14: Assemble the Python workflow and audit ledger

**Files:** Create/finish `pipeline.py`; extend `provenance.py`, public `__init__.py`, `test_pipeline.py`.

- [x] Implement public orchestration using the contracts in section 5. Keep channel preparation, event resolution, annotations, filters, ICA, and epoch finalization as focused helpers. No result writes, report saves, global RNG changes, or global MNE log changes in numerical functions.
- [x] Return an event ledger with one row per original event: original row index, event sample, event code, label, original sampling rate, retained boolean, final epoch row or null, and every drop reason. Compose MNE `selection` with `EventData.original_row` to retain the authoritative acquisition-event identity, including after crop and repeated rejection stages.
- [x] Build canonical provenance containing resolved settings and stage order; input summary/hash; package versions; original/final rates/passbands; filter coefficients/hashes and notch centers; original/final bad channels; interpolation/reference choices; annotation sources; ICA fit/review IDs/scores/exclusions/rank; event source/mapping; original/retained/dropped counts; and output file hashes after serialization.
- [x] Preserve scientific decisions with data; do not conflate preprocessing parameters with existing feature-column computation hashes. Store the manifest's relative filename and a processing-identity hash in `epochs.info["description"]` so the source FIF links back to the bundle; preserve any original description in the JSON. The processing identity covers input/settings/review identity, not serialized output hashes. Do not embed the manifest's final file hash inside FIF while also hashing FIF in the manifest: that would create a circular hash dependency. Explain that the existing feature runner is a consumer, not a verifier of preprocessing history.
- [x] Assert final nonempty EEG/epochs, finite output, aligned metadata, unchanged acquisition-event fields in the ledger, correctly shifted event samples when delay correction was requested, and consistent final rate/time grid before returning.
- [x] Test the complete non-ICA path and the separate fit→review→final path. Test that a disabled optional stage performs no corresponding numerical operation and that no-stage processing matches direct MNE epoch extraction.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_pipeline.py -q
```

Expected: composable in-memory processing, complete auditable decisions, and immutable caller inputs.

### Task 15: Implement the stage registry and shared sequential executor

**Files:** Create `stages.py`, `execution.py`, `test_stages.py`, `test_execution.py`; connect `pipeline.py`.

- [x] Define all 26 stage IDs from section 3 exactly once, with input/output type, callable, scientific config subset, review prerequisite, and dependency IDs. No reflection-based arbitrary function execution. Stage helpers remain independently callable numerical functions.
- [x] Most stages form a sequence; artifact fitting/review and epoch construction are separate branches from `artifact-reference`. `apply-artifact` joins the reviewed model with epochs. A changed fit does not invalidate untouched epoch construction, although its application descendants become stale. Disabled method stages resolve to recorded identity edges rather than manufacturing fake model objects.
- [x] Build the enabled dependency graph from validated settings. Detect unknown IDs and cycles before executing. `list_steps` uses this same registry to generate documented availability and status; do not maintain a second CLI-only list that can drift.
- [x] Implement step/next/run-until using the same dispatch and explicit artifacts. A single step never silently runs parents. Next selects the first pending enabled stage in topological catalog order. Run stops at required review and returns needs-review with the candidate report/decision path.
- [x] Use structured statuses `pending`, `completed`, `disabled`, `needs-review`, `stale`, `failed`. Record failure context without silently substituting an earlier result. Functions that are disabled must not be invoked.
- [x] Direct `preprocess` iterates these same numerical operations; it accepts explicit validated review/model inputs for in-memory execution. Do not make the direct API depend on a writable output directory.
- [x] Add tests proving each of 26 registered stages has a callable, contract, configuration owner, and serialized output type. Test valid branches, disabled stages, unsupported order, stale input, first pending step, and stopping exactly at the requested target.

```python
def test_single_step_does_not_run_missing_parents(tmp_path):
    workflow = make_synthetic_workflow(tmp_path)
    with pytest.raises(ValueError, match="notch.*requires"):
        run_step(workflow, "notch")
    assert not list(workflow.workspace.glob("**/*_raw.fif"))
```

`make_synthetic_workflow` is a test helper in `tests/preprocessing/conftest.py`: write the deterministic raw fixture, write/parse a complete matching config with notch enabled, and return `open_workflow(config)`. Do not mock the executor in this test.

```bash
MNE_DONTWRITE_HOME=true .venv/bin/python -m pytest tests/preprocessing/test_stages.py tests/preprocessing/test_execution.py -q
```

Expected: one behavior for standalone, sequential, and target-based execution; all supported steps discoverable.

### Task 16: Persist checkpoints and enforce precise invalidation

**Files:** Create `checkpoints.py`, `test_checkpoints.py`; extend `execution.py`, `io.py`, and identity records.

- [x] Implement section 3.2's per-stage directories, canonical native serialization, payload hashes, manifest schema, parent IDs, implementation version, and immutable publication. Add one workspace writer lock using an established small locking dependency (`filelock` in the preprocessing extra), not homegrown process detection. Readers may inspect only completed published artifacts.
- [x] Raw/Epochs checkpoints preserve annotations, first_samp, measurement date, channel geometry, projectors, bads, event mapping, baseline state, and metadata. Test round trips before relying on their hashes. Record JSON copies of semantic identity where native serialization normalizes representational details.
- [x] Keep decision files separate from numeric checkpoints. Their payload records reviewed artifact ID and exact choices; writing a decision changes only affected descendants. Do not guess “newest model” from directory order.
- [x] Recompute each stage identity from its own settings/parent identities. Changed crop/bads/filter parameters invalidate the appropriate downstream nodes; changed final baseline does not refit ICA or autoreject. File path/output styling changes do not become scientific model changes.
- [x] Fresh subprocess resume loads verified parent data, not the original input plus already-applied transforms. Test failure/restart before and after every numerical stage. No stage is marked complete before every output validates.
- [x] Test interrupted directory publication, corrupt FIF/HDF5/JSON, unsupported manifest schema, source-content changes, stale component/epoch reviews, duplicate writers, and explicit reset. No automatic refit on corrupted or stale reviewed inputs.
- [x] Run the same synthetic recording uninterrupted and one stage per fresh process, then compare samples, selected events, metadata, bads, reference, passband, final sampling grid, and canonical scientific provenance. Allow only documented native serialization tolerances; wall-clock time/report timestamps can differ.
- [x] Verify the feature runner never discovers `.preprocessing` payloads. Keep disk use transparent in `status`; deleting checkpoints is not required for this implementation and must never happen implicitly.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_checkpoints.py tests/preprocessing/test_execution.py -q
```

Expected: crash-safe sequential continuation with exact dependency tracking and no double application.

### Task 17: Build QC reports and robust result bundles

**Files:** Finish `report.py`, `io.py`, `test_report.py`, and `test_io.py`.

- [ ] Final bundle: `<name>_epo.fif` (and MNE split files if needed), `<name>_preprocessing.json`, `<name>_events.tsv`, `<name>_report.html`. ICA fitting additionally writes the three artifacts in Task 8 and `<name>_ica-report.html`.
- [x] Generate MNE Report content with source/config summary, bad channels/annotations, before/after PSD on explicitly comparable windows/channels, retained/drop counts by reason and condition, final rate/reference/passband, and interpolation/ICA decisions. Report creation must not mutate or trigger extra rejection on the result.
- [x] The ICA review report must expose **every** fitted component, including source traces/spectra, topographies, and EOG/ECG scores when requested. Do not accept `Report.add_ica`'s first-20-component default as a complete review. Require montage for this enabled workflow; basic non-ICA reports can present data without topographies by explicit design.
- [x] Generate deterministic headless figures using Matplotlib Agg in tests, close created figures, and call `report.save(..., open_browser=False)`. Preserve visible scientific warnings; do not hide failed report sections with broad exception handling.
- [x] Before expensive processing, check all known output destinations and source collisions. Default is no overwrite. `--overwrite` replaces only that named bundle's artifacts, not unrelated files or raw inputs.
- [x] Stage a bundle under a temporary directory inside the output directory; save double-precision epochs with `fmt="double"`, collect any split files, write TSV/report, hash written artifacts, and write the manifest last. Validate by reloading staged epochs and checking sample shape, selection, metadata, reference, and values before publishing.
- [x] Do not claim a set of file replacements is a filesystem-wide atomic transaction. Treat the last-published manifest and its file hashes as the completion marker. If publishing an overwrite fails partway, surface the error and do not leave a manifest falsely indicating a valid new bundle. Test this failure explicitly.
- [x] Do not replace an artifact review while overwriting final export files. Refitting requires `reset --from fit-artifact` followed by `step fit-artifact`; the new fit identity invalidates its old review. Reusing unchanged checkpoints is normal execution, distinct from replacing a final export bundle.
- [x] Round-trip every provenance field and epoch selection; test TSV escaping, invalid output names, existing bundle refusal, input/companion-file collisions, split FIF handling, and interrupted writes. JSON must use `allow_nan=False`.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_io.py tests/preprocessing/test_report.py -q
```

Expected: readable FIF/TSV/JSON/HTML artifacts, all ICA components visible, and incomplete writes visibly incomplete.

### Task 18: Add CLI entry points and connect the existing feature runner

**Files:** Create `preprocessing/cli.py`, `test_cli.py`; modify `runner/cli.py`, `tests/test_cli.py`, and `tests/test_runner_batch.py` only where integration needs coverage.

- [x] Implement exactly the command family and return codes in section 3.1. Register parsers without importing GUI/YAML/PyPREP/autoreject at top-level. Use `step fit-artifact` for each offered artifact model, with the same stage interface.
- [x] Generate concise event/resting starter recipes with all optional stages documented. `steps` prints available methods, config paths, input requirements, whether review is needed, and each stage's output. Disabled methods remain discoverable.
- [x] Implement `check` with concrete errors such as `annotations.muscle.filter_freq exceeds the 125 Hz Nyquist frequency`; never change values automatically to make a recording pass.
- [x] `run`, `step`, and `next` call the same execution functions. Expose `--n-jobs` for applicable MNE calls and final-export `--overwrite`; reset controls intermediate recomputation. Progress includes recording, stage, duration, result counts, warnings, artifact path, and next action. No claim of completion before export validation.
- [x] Tests cover init/check/steps/status, missing parents, disabled steps, first pending stage, requested stopping stage, review-needed exit 3, unchanged reuse, stale decision error, explicit reset, final overwrite refusal, and clean process restart between each stage.
- [x] Test each optional dependency absent: core package/feature CLI still works; requesting the corresponding operation names its installation extra. GUI failure never masquerades as successful review.
- [x] Save synthetic raw FIF, complete the sequential pipeline, then point a normal existing feature recipe at the output root. Assert intermediate checkpoint epochs are never discovered, only exported epochs are read, and event metadata survives into feature rows.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_cli.py tests/test_cli.py tests/test_runner_batch.py -q
```

Expected: short user workflows, precise incomplete/error states, and unchanged feature-runner semantics.

### Task 19: Make visual and headless review equally usable

**Files:** Create/finish `review.py`, `test_review.py`; extend `report.py`, CLI tests, and guide.

- [x] Raw viewer: use MNE Raw browser on a disposable checkpoint copy, with channels/PSD/sensor positions and existing/candidate annotations visible. Capture changes to bad channels and annotations as a decision diff, not edits to the parent FIF.
- [x] Artifact viewer: use MNE component/property/source plots for all ICA components, SSP projector plots for every candidate, and regression coefficient/removal summaries. Translate choices into model-specific decision dataclasses with exact model hash; an empty selection is explicit.
- [x] Epoch viewer: show corrected full-rate epochs with automatic rejection/repair summaries. Capture manually excluded original event IDs. Warn clearly if an attempted decision references rows already absent from its source checkpoint; never reinterpret them as current row positions.
- [x] `review --decisions FILE` validates the same schema and produces the same decision/result as the GUI path. Generate pending templates with null decisions, plus concise field descriptions; a user does not need to invent filenames, hashes, or component ranges.
- [x] Viewer cancellation/crash or headless-display failure must leave review pending and downstream data untouched. A normal completed review saves only valid decisions and prints the next command. No web interface, hidden browser service, or display dependency for normal numerical runs.
- [ ] Unit-test viewer adapters with controlled returned edits, and test headless review end to end without GUI mocks. Add a manual GUI smoke checklist covering raw/ICA/SSP/regression/epochs on the synthetic example; report whether it was actually exercised, separately from headless tests.
- [x] Ensure short everyday help (`next`, `run`, `status`, `review`) and detailed per-stage help. Error messages include field, actual value, required condition, and action; no vague “invalid pipeline” messages.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/preprocessing/test_review.py tests/preprocessing/test_cli.py -q
```

Expected: a user can inspect, decide, save, and resume with native MNE tools or equivalent decision files, without modifying Python internals.

### Task 20: Document the complete workflow and add real-data validation

**Files:** Create the two documentation pages, `examples/preprocessing.yaml`, and `tests/validation/test_preprocessing.py`; modify the README, installation page, documentation indexes, and validation guide.

- [x] Document installation, raw readers, SI units, study-specific filter/reference choices, manual bad-channel/spans review, optional automatic amplitude/break annotations, event sources, ICA fit→review→run, fixed/event epochs, residual rejection, interpolation, downsampling, baseline, outputs, and feature-runner handoff.
- [x] State defaults accurately: no filtering, notch, ICA, interpolation, reference change, baseline, or decimation unless configured. Existing bad annotations are honored and invalid input is rejected. “Clean epochs” is not a claim that all artifacts were automatically detected.
- [x] Document sampling/index conventions, fixed epoch endpoints, original reference restoration, artifact-fit-reference rejection thresholds, per-recording fitting, missing-montage requirements, automated detector assumptions, and each method's failure conditions.
- [x] Explain how to inspect raw traces/PSD and mark bad electrodes/spans using MNE before rerunning. Explain that component review is invalidated by a new fit, and include one example of intentionally keeping all components.
- [x] Give an end-to-end Python example using actual public classes/functions and an end-to-end CLI example. Execute the Python example on the synthetic fixture and parse every full/fragment example through the real loader after applying its documented surrounding configuration. Keep docstrings to concise public contracts; put scientific explanations in the guide.
- [x] Add one opt-in raw EEGBCI recording test using `mne.datasets.eegbci.load_data(1, [3], update_path=False)`, `mne.io.read_raw_edf`, and `mne.datasets.eegbci.standardize`. Use its T1/T2 annotation event mapping, explicit 1–40 Hz filtering, no ICA, final average reference, and tmin=-0.2/tmax=1.0. Independently run the same MNE sequence and compare retained events and samples. Do not route through the existing preprocessed EEGBCI fixture or change its established outputs.
- [x] Give the new test the existing `validates` metadata with a precise numerical-equivalence claim and dataset label, following existing tests. Add separate synthetic artifact-attenuation evidence; real-data equivalence alone does not prove scientific adequacy of cleaning.

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m sphinx -W --keep-going -b html docs /private/tmp/eegfeat-preprocessing-docs
EEGFEAT_DATASETS=1 MNE_DONTWRITE_HOME=true MPLBACKEND=Agg .venv/bin/python -m pytest tests/validation/test_preprocessing.py -ra
```

Expected: documentation builds and opted-in real-data comparison passes. If data cannot be downloaded, record the concrete limitation and keep that validation explicitly unverified.

### Task 21: Package, CI, independent review, and final verification

**Files:** Modify `.github/workflows/ci.yml`, `tests/test_packaging.py`, and `pyproject.toml` as needed for the implemented extra and template.

- [x] Add the preprocessing extra to the CI jobs that run its tests; explicitly import YAML/sklearn there so missing dependencies cannot silently skip the integration suite. Update exact CI-string assertions in packaging tests to reflect the new extras list.
- [ ] Run dedicated automatic-quality/rejection and GUI-adapter jobs with their declared extras, plus a manual GUI smoke checklist. Keep a base-wheel import/CLI smoke test without extras. Add a wheel-with-preprocessing-extra test that runs `preprocess init` outside the source checkout and verifies the packaged YAML exists and parses.
- [ ] Add a Python 3.11/MNE 1.8 job for the preprocessing suite and retain current supported Python endpoint jobs with current MNE. Use one code path; resolve unsupported APIs by choosing supported calls, not try/except compatibility branches. Do not lower or raise the project's declared minimum silently.
- [x] Run the full local suite once after focused tests pass, plus Ruff, Black check, mypy, and strict Sphinx. Investigate new failures at their root; do not relax tolerances or broadly silence warnings to make CI green.
- [x] Review the complete diff against section 3's order and section 7's acceptance matrix. Check no automatic ICA exclusion, hidden preprocessing in feature extraction, guessed metadata, broad exception catches, or new fallback behavior slipped in.
- [x] Record actual commands/outcomes and any unavailable external validation. Commit only coherent implemented changes after verification; no automatic merge or publication is part of this plan.

Final commands (use writable cache directories in restricted environments):

```bash
export MNE_DONTWRITE_HOME=true
export MPLBACKEND=Agg
export MPLCONFIGDIR=/private/tmp/eegfeat-preprocessing-mpl
export XDG_CACHE_HOME=/private/tmp/eegfeat-preprocessing-cache
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests
.venv/bin/python -m black --check src tests
.venv/bin/python -m mypy
.venv/bin/python -m sphinx -W --keep-going -b html docs /private/tmp/eegfeat-preprocessing-docs
.venv/bin/python -m build
```

Expected: all available checks pass. Opt-in dataset tests run separately; normal pytest skips are not evidence that those checks ran.

## 7. Acceptance matrix

| Requirement | Evidence required before completion |
| --- | --- |
| Optional pipeline | Base installation/import and existing CLI tests; no preprocessing call inside feature runner. |
| Clear scientific implementation | Direct MNE calls in focused modules; no signal-processing reimplementation; cited public guide. |
| Input preparation | Channel/type/montage validation and input immutability tests. |
| Selection and timing | Acquisition-relative crop, trigger-delay rounding, bipolar auxiliary derivation, metadata joins, and original event IDs preserved. |
| Bad channels and segments | Existing/manual/amplitude/break/muscle cases, PyPREP/bridge evidence, bad lists preserved, exact annotation timing before/after cropping. |
| Stimulation repair | MNE equivalence, untouched triggers, rejected invalid windows, and repaired-sample ledger. |
| Filtering | Independent MNE equivalence, tone attenuation, stimulus preservation, boundary/length failures. |
| Events | Nonzero first_samp, dated/undated annotations, stim/file sources, duplicate/missing event errors. |
| ICA | Same-fit reconstruction, rank checks, training exclusions, artifact injection, review and stale-fit tests. |
| SSP and regression | Independent MNE agreement, explicit fit/application reference, saved model review, and rejected incompatible data. |
| Epoching/rejection | Exact selection/drop_log, fixed-duration sample count, metadata alignment, all-rejected failure. |
| Autoreject and manual review | Separate fit/transform tests, channel-by-epoch repair log, padding-window consistency, original event identity, GUI/headless decision equivalence. |
| Interpolation/reference | MNE agreement, geometry errors, retained bad flags, auxiliary-channel preservation. |
| Sampling/detrending/baseline | Anti-alias guards, target-rate MNE equivalence, validated padding/edge response, pre-downsampling spike rejection, SciPy detrending, final-grid baseline checks. |
| Reproducibility | Seed, configuration, versions, filters, input/model/output hashes, event ledger. |
| Outputs and QC | FIF/JSON/TSV round trip, all ICA components visible, no false-success manifest after interrupted write. |
| Usable end to end | CLI preprocessing followed by existing feature recipe check/run. |
| Every step available | All 26 catalog entries have callable operations, config ownership, prerequisites, outputs, and help. |
| Sequential ease | Fresh-process step/next/run-until/status/review examples, exact next-action messages, and documented return codes. |
| Safe resumption | Uninterrupted/stepwise numerical and metadata equivalence, no double application, targeted invalidation, corrupt/stale artifact errors, interruption recovery, and writer-lock tests. |
| Optional installations | Each missing extra produces an actionable error only when requested; core import and feature extraction remain available. |
| Supported versions | New preprocessing suite on minimum MNE and current MNE, without compatibility shims. |
| Documentation | Parsed examples, executed API example, strict Sphinx build. |
| Real EEG evidence | Explicitly opted-in raw EEGBCI equivalence test; report unavailable download honestly. |

## 8. Plan execution notes

Implement all 21 tasks in order. The stage catalog and checkpoint contract are requirements from the outset, not an optional follow-up to a monolithic pipeline. Keep intermediate commits small; each commit should include the behavior and its meaningful tests. Do not advertise the preprocessing extra/CLI as complete until the end-to-end task works. Scientific checks take precedence over abstraction or brevity when their removal would hide incorrect data handling.

No separate design document is required. This plan incorporates the user's requested scope and sequential-use revision; implementation should proceed against it when requested, without repeating the design approval process.

### Planning validation performed

This revision was checked for 26 unique ordered stage IDs, 21 consecutively numbered implementation tasks, Python-block syntax, five valid YAML blocks, valid command targets/task references, balanced fences, and whitespace. Scientific workflow and configuration references were reviewed for the expanded methods and sequencing contract. Local MNE probes verified the corrected dated/undated annotation-append behavior with a nonzero first sample, and the exact fixed-epoch sample-count assertions. The interpolation probe was interrupted while importing SciPy; it provides no numerical interpolation evidence. The implementation tests and supported-version CI listed above remain required and have not run because implementation has not started. Only this plan was added to the repository.

### Implementation notes (2026-09-21)

Implemented and verified locally with MNE 1.13.2, PyPREP 0.9.0, autoreject 0.4.3:
`tests/preprocessing` (63 tests), Ruff, Black, mypy, the strict Sphinx build, and
the opt-in EEGBCI equivalence test all pass. Unticked items above are deliberate:

- Fitted models live in the immutable `fit-artifact` checkpoint (`model-ica.fif`,
  `model-proj.fif`, `model-regression.h5`) with their decision file, and the export
  manifest records the fit identity, evidence, and decision. The export bundle does
  not duplicate the model files.
- The `preprocessing` extra requires MNE ≥1.13.2 (`ICA(rng=...)`, `Report.add_projs`
  keyword API); the CI `preprocessing-minimum` job pins that version instead of 1.8.
- Viewer adapters are covered with controlled returned decisions; the interactive Qt
  viewers were not exercised in this environment (no display).
- Real-data validation runs on MNE's ERP CORE Flankers recording (bipolar VEOG, crop,
  notch, band-pass, reviewed ICA, epochs, threshold rejection, average reference,
  decimation, crop, baseline, export) and an EEGBCI eyes-open run (PyPREP, autoreject,
  interpolation), each checkpoint compared with the direct MNE or library call. That
  run exposed and fixed three edge cases synthetic data never hit: FIF calibration
  round-off on data with a DC offset (round-trip checks are now relative), a stale
  reject window after `Epochs.decimate` that `read_epochs` refuses, and crop/baseline
  bounds that fall up to one sample inside a rounded, decimated grid.

