# Cohort Preprocessing and Command-Line Ergonomics

**Date:** 2026-09-21
**Branch:** `feat/optional-preprocessing`
**Status:** approved design, awaiting implementation plan

## 1. Goal

Make `eegfeat preprocess` run a whole cohort from one recipe, let a recipe
state its review policy, and make every command explain itself, without
changing a single numerical result, checkpoint identity, or validation rule.

Today one YAML names one raw file, review gates stop every recording until a
person runs `review`, stop lines are not commands, the pending decision file is
JSON with unexplained fields, and no subcommand or flag has help text. On
2026-09-21 preprocessing 146 recordings needed an external driver script that
generated a YAML per recording and automated the ICA review by hand.

## 2. Invariants

These hold for the whole change and are checked by the tests in section 8.

1. **No configuration field is removed, renamed, loosened, or given a hidden
   default.** Every key that is explicit today stays explicit. Unknown keys
   remain errors. The two additions that supply a value the user did not write
   are `output.name` derived from the file name (section 3.2) and the
   `suggested` review policy (section 4), and both are opt-in.
2. **The engine below the recipe loader does not change behaviour.** Stage
   functions, `execute_numeric`, checkpoint layout, stage identities, decision
   binding, and export verification are untouched except for the two additions
   in section 4 and section 6.4. A recipe that loads today produces
   byte-identical checkpoints and bundles.
3. **A cohort recipe is a set of one-recording configurations.** Everything
   that runs, reviews, resets, or inspects a recording takes the same
   `PreprocessingConfig` it takes now.
4. **Automation is a recorded study choice, never an implicit fallback.** A
   review gate is skipped only because the recipe says `suggested` or
   `disabled`, and the decision file that results is bound to its parent
   checkpoint exactly like a human decision.
5. **Failures are isolated and reported, never hidden.** One recording's error
   does not stop the others, and the exit status says that something failed.

## 3. Recipe

### 3.1 `input`

| Key | Type | Rule |
| --- | --- | --- |
| `path` | string | One recording. Resolved against the recipe directory. Unchanged. |
| `root` | string | A directory searched with `pattern`. Resolved against the recipe directory. |
| `pattern` | string | A `pathlib` glob relative to `root`. Required with `root`. No default. |

Exactly one of `path` and `root` is present. `pattern` without `root` is an
error. Every matched file must be a regular file with an accepted suffix
(`.fif`, `.fif.gz`, `.edf`, `.bdf`, `.vhdr`, `.set`); a match with another
suffix is an error naming the file, not a silent skip. Paths with a component
that starts with `.` are skipped, because macOS writes `._` AppleDouble files
beside every file on an external drive. Matching nothing is an error. The
matches are processed in sorted order.

### 3.2 `output`

| Key | Type | Rule |
| --- | --- | --- |
| `directory` | string | Unchanged. With `root`, the input tree below `root` is mirrored under it. |
| `name` | string | Optional. Allowed only with `input.path`. |

When `name` is absent it is the recording's file name with the accepted suffix
removed, then one trailing `_raw` or `_eeg` removed. The result must satisfy
the existing safe-stem rule (`[A-Za-z0-9][A-Za-z0-9_.-]*`); otherwise the error
says to set `output.name` or rename the file. `name` with `root` is an error,
because every recording would write to the same bundle.

Per recording, the effective `OutputSettings` is
`directory = output.directory / source.parent.relative_to(root)` and
`name` as above. The workspace therefore stays
`<directory>/.preprocessing/<name>` with no change to the checkpoint layout.
Two recordings that resolve to the same bundle are an error naming both files.

### 3.3 Recording labels

Each recording gets a label used by `--recording`, progress lines, and stop
lines. The label is `name` when names are unique across the cohort, otherwise
the bundle path relative to `output.directory` in POSIX form. A one-recording
recipe has one label and never needs `--recording`.

### 3.4 Per-recording files

`epochs.events.path` and `epochs.metadata` name files that belong to one
recording. Both accept two placeholders, substituted per recording before the
existing path resolution:

| Placeholder | Value |
| --- | --- |
| `{name}` | The derived or given `output.name`. |
| `{parent}` | The absolute directory containing the recording. |

Any other `{...}` token is an error naming the key. A relative result still
resolves against the recipe directory, so `{parent}/{name}_events.tsv` reads a
sidecar beside the recording and `metadata/{name}_trials.tsv` reads a flat
sidecar folder. Every other path in the recipe (`channels.montage`,
`stimulation`, artifact settings) is a study-level value and takes no
placeholders. The placeholders also work with `input.path`.

### 3.5 `workflow`

| Key | Values | Default | Change |
| --- | --- | --- | --- |
| `raw_review` | `required`, `suggested`, `disabled` | `required` | adds `suggested` |
| `artifact_review` | `required`, `suggested` | `required` | new key |
| `epoch_review` | `required`, `optional`, `disabled` | `optional` | none |

`artifact_review` is accepted whether or not an `artifact` block exists, like
`epoch_review`; it governs a gate that may be disabled. There is no
`suggested` for `epoch_review` because no detector proposes epochs.

## 4. The `suggested` policy

A review stage whose policy is `suggested` is enabled exactly as `required`
is. When execution reaches the gate and no decision file exists, the engine
writes the suggested decision to the same decision path, bound to the same
parent identity, and continues. If a decision file already exists it is used
as today. Nothing else about the gate changes: a later change to a detector
setting changes the parent identity and invalidates the decision, and `reset`
retires it.

The suggested decisions are:

| Gate | Decision |
| --- | --- |
| `review-raw` | `bads` = channels already marked bad on the reviewed raw, in order, followed by detector candidates not yet listed; `spans` = the annotation candidates. |
| `review-artifact`, ICA | `exclude` = the fit's `suggested_exclude`, with the fit's `fit_id`. |
| `review-artifact`, SSP | `include` = every fitted projector. |
| `review-artifact`, regression | `apply: true`. |

The raw rule fixes an existing defect: `review --suggested` currently takes
only the detector candidates, so it clears bad channels marked on the file or
by `channels.bads`. `save_review(..., suggested=True)` and the policy share one
`suggested_decision` function, which moves from `review.py` to `execution.py`
so the engine can call it without an import cycle.

Provenance already records `raw_review`; it additionally records
`artifact_review`. The Python entry point `preprocess()` keeps constructing
its `WorkflowSettings` from the decisions it is given and uses `required` for
`artifact_review`.

## 5. Loading

`config.py` gains `load_recipe(path) -> dict[str, PreprocessingConfig]`,
label to configuration in processing order. It reads the YAML once, validates
every study-level section once, then resolves `input`, `output`, and the two
placeholder fields per recording. `load_config(path)` remains and returns the
single configuration; when the recipe selects more than one recording it
raises `ValueError` naming the count and `load_recipe`. `PreprocessingConfig`,
`open_workflow`, `run_until`, `run_step`, `run_next`, `list_steps`,
`read_checkpoint`, `reset_from`, and `save_review` keep their signatures.
`load_recipe` is exported from `eegfeat.preprocessing`.

## 6. Command line

### 6.1 Help

Every subcommand, positional argument, and flag has a help string. The
`preprocess` parser has a description. The top-level `eegfeat` description
names both halves: preprocessing raw recordings into epochs, and computing
features from epochs.

### 6.2 Selecting recordings

`--recording LABEL` selects one recording of a cohort recipe. An unknown label
is an error that lists the labels.

| Command | Without `--recording` | With it |
| --- | --- | --- |
| `check` | every recording | that one |
| `run` | every recording | that one |
| `status` | one line per recording | the stage list, as today |
| `review` | each recording awaiting that review, in order; `--decisions` then requires `--recording` unless exactly one awaits | that one |
| `step`, `next`, `inspect`, `reset` | error when the recipe selects more than one | that one |
| `init`, `steps` | not applicable | not accepted |

### 6.3 Cohort `run` and `check`

`run` processes the selected recordings in order. A recording runs to
`--until` or stops at a gate. An exception in one recording is caught, reported
on that recording's line, and the next recording starts. `check` runs the
existing `load`, `prepare`, and `events` validation for each recording and
writes nothing.

Progress reuses `eegfeat.runner.progress`: `TextReporter` by default and
`JsonReporter` with the new `--progress-json` flag, so the events match the
feature runner's. Each stage is a `step` event. A recording that stops at a
gate is reported as done with the message `awaiting review-raw` and its stop
command; a failed recording is reported with its error.

Exit status:

| Code | Meaning |
| --- | --- |
| 0 | every selected recording reached `--until` (or, for `check`, passed) |
| 1 | at least one recording failed |
| 2 | the recipe or a prerequisite is wrong before any recording ran |
| 3 | no recording failed and at least one awaits a review |

### 6.4 Stop lines and next actions

The engine keeps producing `next_action` strings with the `CONFIG`
placeholder. The CLI renders each as a complete command: prefix
`eegfeat preprocess `, replace `CONFIG` with the recipe path as given, and
append ` --recording LABEL` when the recipe selects more than one recording.
Rendering happens in one function used by every command that prints a next
action.

`status` without `--recording` prints, per recording, the label and one of:
`exported`, `awaiting <review-stage>`, `stale at <stage>`, or
`<done> of <enabled> stages`. `--verify` behaves as today for the selected
recording.

When a `run` finishes with every selected recording exported, the last line is
`Next: eegfeat init recipe.toml` followed by the `inputs.root` value to set,
which is `output.directory` as written in the recipe.

### 6.5 Pending decision file

The engine writes `review-<target>.pending.yaml` as YAML with a comment above
each field explaining what it holds and what an empty list means, using the
wording of the current guide. `parent_id` is written unchanged and commented
as not to be edited. The strict loader reads the file as today.

### 6.6 `review` without flags

`review CONFIG <target>` with neither `--decisions` nor `--suggested` first
looks for that recording's pending file. If it exists and no field is `null`,
it is used exactly as `--decisions <that file>`. Otherwise the viewer opens. If
the viewer extra is missing, the error names the extra and the pending file to
fill in. `--decisions` and `--suggested` keep their meaning and remain
mutually exclusive.

## 7. Documentation

`template.yaml` gains commented `root`/`pattern` lines and the two new policy
values. `examples/preprocessing.yaml` becomes a cohort recipe with a
`{parent}/{name}_events.tsv` events file. The README preprocessing section and
`docs/guides/preprocessing.rst` show the cohort path first and the
one-recording path as the special case, list the exit codes and the
`--recording` rule, and drop the instruction to edit a JSON file. Feature-runner
documentation is untouched.

## 8. Tests

Test-driven, in `tests/preprocessing/`.

- `load_recipe`: `path` and `root` exclusive; `pattern` required with `root`;
  `name` forbidden with `root`; derived names strip one `_raw`/`_eeg`; unsafe
  derived name errors; wrong-suffix match errors; hidden paths skipped;
  duplicate bundle errors; placeholders substitute and unknown ones error;
  labels fall back to relative paths on collision; `load_config` refuses a
  cohort.
- `suggested`: the decision file is written with the parent identity; the raw
  decision keeps pre-marked bads; a detector setting change makes it stale;
  `raw_review: suggested` with `disabled` detectors records existing bads and
  no spans; `review --suggested` now keeps pre-marked bads.
- Cohort execution: a two-recording recipe runs both, mirrors the tree, and a
  failure in the first still exports the second with exit 1; a gate yields
  exit 3 with a complete stop command; `--recording` narrows; `--progress-json`
  emits the runner's event names.
- CLI: every subcommand's `--help` mentions each of its flags; `status`
  summary lines; the export handoff line; `step` without `--recording` on a
  cohort is an error.
- Pending file: written YAML loads with the strict reader; a filled file is
  accepted by flagless `review`; a file with a `null` is refused with the
  pending path in the message.
- Existing tests, including the equivalence and one-shot-versus-stepwise
  tests, pass unchanged.

## 9. Out of scope

A single preprocess-then-features command, per-recording overrides inside one
recipe (exceptions go through review decisions or a second recipe with a
narrower pattern), BIDS entity discovery, pooled or subject-wide ICA, raw
concatenation, and any change to the feature runner's recipe.
