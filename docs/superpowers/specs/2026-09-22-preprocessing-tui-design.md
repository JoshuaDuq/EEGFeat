# Preprocessing TUI

An optional Go terminal front end for `eegfeat preprocess`. It owns the
workflow and the review decisions; MNE's viewers remain the place to look at
signals. Nothing in the Python package depends on it.

## Why

The review gates work, but the interactive path is thin: closing the viewer is
the save gesture, the confirm dialog carries its question only in its title
bar, and the ICA dialog lists bare component indices with nothing pre-ticked
although ICLabel's verdict sits in the checkpoint. The evidence a reviewer
needs is text, and a terminal renders text well.

## Shape

Bubble Tea, in the EEG_fMRI_Pipeline TUI's monochrome style: `colors.go` is
carried over verbatim, plus only the layout helpers this app calls. Two
screens.

**Home** — recordings on the left with each one's summary; the selected
recording's stage list on the right, following its cursor when it does not
fit. The same pane renders a live run: stage glyphs flip as `--progress-json`
events arrive. A full-width LOG panel below takes the rows the stage list does
not need (omitted under five) and keeps everything Python wrote during runs —
stderr and log events, newest last, 500 lines, `pgup`/`pgdn` to scroll — so
the output of a run is still readable at the gate it stopped at. `↵` executes
Python's own `next` recommendation.

**Gate** — one checklist for every gate. Rows arrive pre-ticked from the
detectors' verdict and carry their reasons: PyPREP tests and vote counts for
channels, ICLabel class and confidence and detectors for components, event
name and peak-to-peak amplitude for epochs. The raw gate adds a SPANS section
and an `onset duration` input. `v` suspends the TUI and opens
`inspect STAGE` in MNE's viewer.

Keys — home: `↑↓` navigate, `↵` next action, `r` run selected, `a` run all,
`x` reset from stage, `v` viewer, `esc` stop a live run, `q` quit.
Gate: `space` toggle, `s` restore suggested, `a` clear the checklist (spans
stay individually toggleable), `o` sort by score, `v` viewer, `↵` save,
`esc` back. A reset, stopping a run (`esc`, or `q` during one) and leaving a
gate whose ticks differ from the suggestion all ask `y`/`n` first; `ctrl+c`
never asks.

A failure shows its last line — Python's one-line message, or a traceback's
exception — and the log keeps the whole text, command included. The viewer's
stderr is copied for the same reason: it prints to the screen the TUI covers
again on return.

## Python contract

Two read-only projections of state that is already on disk. Writes stay in
Python: the TUI serialises a decision to a temp JSON file and calls
`review --decisions FILE`, so `parent_id`, `fit_id` and null checks keep
running where they are. Go never reads or writes inside `.preprocessing/`.

`status CONFIG [--recording L] --json`

```json
{"recordings": [{"label": "sub-01", "summary": "awaiting review-raw",
  "stages": [{"stage": "load", "state": "completed", "reason": ""}],
  "next": {"kind": "review", "stage": "review-raw", "target": "raw"}}]}
```

`next.kind` is `review`, `run` or `reset`, derived from the pending stage's
state; `null` once exported. A gate whose decision is already saved reports
`pending`, not `needs-review`: only the run is left (this also corrects the
text `status`, which used to keep recommending `review` after one was saved).

`inspect CONFIG STAGE --recording L --json` — review stages only; exclusive
with `--report`. Reads `state.json` and the FIF header; only the epoch gate
loads data, for the peak-to-peak.

```json
{"stage": "review-artifact", "parent": "fit-artifact", "parent_id": "…",
 "fit_id": "…", "method": "ica", "field": "exclude",
 "items": [{"id": 3, "label": "ICA003", "tags": ["muscle", "iclabel"],
            "score": 0.91, "suggested": true}]}
```

`field` names the decision key (`bads`, `exclude`, `include`, `apply`);
`items[].id` is what that key holds. `parent` is the checkpoint the gate
reviews — the one `inspect` opens, since the gate has none of its own; for a
fitted ICA the viewer shows components and sources rather than the raw. The raw gate adds `duration` and
`spans[]` (`onset`, `duration`, `description`, `suggested`). `apply` gates
have one item; the decision is whether it is ticked.

## Go layout

```
tui/                 module github.com/JoshuaDuq/EEGFeatML/tui
  main.go            eegfeat-tui CONFIG [--n-jobs N]
  styles/            colors.go verbatim; ui.go: panel, header, footer, checkbox
  eegfeat/           Client: Status, Inspect, Review, Reset, Run; the JSON types
  app/               model.go, home.go, gate.go, run.go
```

`eegfeat/` imports no Bubble Tea; it finds the binary through `$EEGFEAT` or
`eegfeat` on `PATH` and fails before the screen opens if neither exists.
Exit 3 from `run` is a reached gate, not a failure. After a run ends, status
is re-fetched; events never decide gate state. While idle, status refreshes
every 5 s so a decision saved from the CLI shows up. The poll never blocks
keys, runs one Python process at a time, and each load is numbered so an
answer superseded by a later load is dropped; a failed poll keeps the last
status and clears itself once status reads again. `esc` stops a run with
SIGTERM — checkpoints publish atomically and Python refuses a second writer.
macOS and Linux; no Windows branch.

## Testing

Python: both `--json` outputs, the review/non-review and `--json`/`--report`
refusals, and that `--decisions` accepts a JSON file. Go: `eegfeat/` against
a fake `eegfeat` script that replays fixtures and records argv; `app/`
through `Update` with key messages — row mapping, decision serialisation,
event to state, save and refresh flow.
