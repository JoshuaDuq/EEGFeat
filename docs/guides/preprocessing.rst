Preprocessing
=============

.. raw:: html

   <p class="hero-lede">
     Turn raw recordings into epochs from one YAML recipe. Each stage of each
     recording is a checkpoint. The feature runner reads the exported FIF and
     does not call this package.
   </p>

Install the ``preprocessing`` extra. PyPREP, autoreject, ICLabel, and the
picard ICA solver need ``preprocessing-auto``. The MNE viewers used by ``review`` and ``inspect`` need
``preprocessing-gui``. Accepted inputs are the suffixes
``mne.io.read_raw`` reads here: ``.fif``, ``.fif.gz``, ``.edf``, ``.bdf``,
``.vhdr``, and ``.set``.

A stage runs when its setting is present. ``null``, an empty list, or a
disabled review leaves that stage out. Amplitudes are volts, times are seconds,
and frequencies are hertz.

Quick Start
-----------

``python -m eegfeat preprocess`` is the same command as ``eegfeat preprocess``.

.. code-block:: bash

   python -m pip install -e ".[preprocessing]"
   eegfeat preprocess init preprocessing.yaml --mode events

``--mode resting`` writes ``epochs.kind: fixed`` and ``duration: 2.0`` instead of
an event block. ``init`` will not replace an existing file.

Edit the recipe before ``check``. ``init`` writes placeholders.

- ``input.path`` is ``recording_raw.fif``. Point it at the recording, or
  replace it with ``input.root`` and ``input.pattern`` to select a cohort.
- ``epochs.events`` is ``source: annotations`` and ``event_id: {stimulus: 1}``.
  The names and codes have to be annotations on the recordings. Use
  ``source: stim`` or ``source: file`` when they are not.
- ``workflow`` is ``raw_review: required``. Keep it to review each recording
  yourself, or set ``suggested`` to save the detectors' verdict and continue.
- ``channels.projections`` is ``error``. A FIF with inactive projectors stops
  ``check`` until this is ``apply`` or ``discard-inactive``.

.. code-block:: bash

   eegfeat preprocess check preprocessing.yaml
   eegfeat preprocess run preprocessing.yaml

``check`` reads every recording and runs ``load``, ``prepare``, and ``events``.
It writes nothing. ``run`` processes the recordings in order. A failure in one
is reported and the next one starts. With ``raw_review: required``, each
recording stops at the raw gate; the run prints the command that continues it
and exits 3:

.. code-block:: text

   [1/2] sub-01
         ✓ awaiting review-raw · eegfeat preprocess review preprocessing.yaml --recording sub-01 raw

With a display, install ``eegfeat[preprocessing-gui]`` and run that printed
command. It opens the Qt browser. The decision is saved when the dialog is
accepted.

Without a display, fill in the pending file the run wrote,
``<bundle directory>/.preprocessing/<name>/decisions/review-raw.pending.yaml``.
Each field is explained in a comment above it; replace every ``null`` and
leave ``parent_id`` as written. Then run the same ``review`` command: it reads
the filled file. Without ``--recording``, ``review`` walks every recording
that awaits that gate.

.. code-block:: bash

   eegfeat preprocess review preprocessing.yaml raw
   eegfeat preprocess run preprocessing.yaml

That second ``run`` exports. The init recipe has ``artifact: null`` and
``epoch_review: optional``, so there is no second gate. ``optional`` and
``disabled`` both skip epoch review. Only ``epoch_review: required`` stops at
``review-epochs``. An ``artifact`` block adds ``review-artifact``, governed by
``artifact_review``, with the same exit code and the same pending-file
pattern. :file:`examples/preprocessing.yaml` is a cohort recipe with ICA and
both gates set to ``suggested``, so it runs to export unattended.

When every recording is exported, ``run`` prints the ``eegfeat init`` command
and the ``inputs.root`` to set. ``status`` prints one line per recording,
``exported``, ``awaiting <stage>``, ``stale at <stage>``, or how many stages
are done, followed by the next command to run. ``--recording LABEL`` narrows any command to one recording, and is
required for ``step``, ``next``, ``inspect``, and ``reset`` on a cohort.

A later ``run`` reuses a checkpoint whose recipe and parents still match,
checking only its identity; a payload is read and hashed when a stage
consumes it, and ``status --verify`` re-reads every completed checkpoint.
The source recording is read and hashed once per ``run``. A pending review
file the run already wrote is kept, with any edits in it.
``next`` runs one pending stage. ``step STAGE`` runs that stage and requires
its parents. ``reset --from STAGE`` retires that stage and every stage that
depends on it. Payloads stay on disk.

.. code-block:: text

   eegfeat preprocess init CONFIG [--mode events|resting]
   eegfeat preprocess check CONFIG [--recording LABEL]
   eegfeat preprocess steps CONFIG
   eegfeat preprocess status CONFIG [--recording LABEL] [--verify]
   eegfeat preprocess step CONFIG STAGE --recording LABEL [--n-jobs N] [--overwrite]
   eegfeat preprocess next CONFIG --recording LABEL [--n-jobs N] [--overwrite]
   eegfeat preprocess run CONFIG [--until STAGE] [--recording LABEL] [--n-jobs N] [--overwrite] [--progress-json]
   eegfeat preprocess inspect CONFIG STAGE --recording LABEL [--report]
   eegfeat preprocess review CONFIG raw|artifact|epochs [--recording LABEL] [--decisions FILE | --suggested]
   eegfeat preprocess reset CONFIG --from STAGE --recording LABEL

``--recording`` is optional when the recipe selects one recording. Every
command and flag has ``--help``.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Exit
     - Meaning
   * - 0
     - Every selected recording reached the requested stage, or ``check`` passed.
   * - 1
     - At least one recording failed. The others still ran. Also an I/O failure.
   * - 2
     - The recipe or a prerequisite is wrong before any recording ran.
   * - 3
     - No recording failed and at least one awaits a review.

``--n-jobs`` is passed to filtering and resampling. ``--overwrite`` republishes
an export that already matches the recipe; a deleted bundle is republished
without it. A stale checkpoint is removed with ``reset``, not with
``--overwrite``. ``--progress-json`` writes one JSON event
per line in the feature runner's format, with each recording as a subject.

From Python, :func:`eegfeat.preprocessing.load_recipe` returns one
configuration per recording, keyed by label, and
:func:`eegfeat.preprocessing.load_config` returns the single configuration of
a one-recording recipe. The rest of the API is per recording.

.. code-block:: python

   from eegfeat.preprocessing import load_recipe, open_workflow, run_until

   for label, config in load_recipe("preprocessing.yaml").items():
       outcome = run_until(open_workflow(config), "review-raw")
       print(label, outcome.state, outcome.next_action)

The Recipe
----------

A recipe is YAML. Relative paths resolve against the recipe file. Unknown keys
are errors. Duplicate keys are errors.

.. code-block:: yaml

   input:
     root: sourcedata
     pattern: "sub-*/eeg/*_task-example_eeg.vhdr"
   output:
     directory: preprocessed
   workflow:
     raw_review: required
     artifact_review: suggested
     epoch_review: optional
   filter:
     l_freq: 0.1
     h_freq: 40.0
     notch_freqs: [60.0]
   epochs:
     kind: events
     events: {source: annotations, event_id: {stimulus: 1}}
     tmin: -0.5
     tmax: 1.5

``init --mode resting`` writes ``epochs.kind: fixed`` with ``duration: 2.0``
instead of the event block. A filled study file is
:file:`examples/preprocessing.yaml`.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Section
     - Keys
   * - ``input``
     - ``path``, one recording. Or ``root`` and ``pattern``, a directory and a
       glob below it; ``pattern`` has no default. Exactly one of ``path`` and
       ``root``. Every match must be a recording (``.fif``, ``.fif.gz``,
       ``.edf``, ``.bdf``, ``.vhdr``, ``.set``); another suffix is an error.
       Hidden files and directories are skipped.
   * - ``output``
     - ``directory``, and with ``path`` an optional ``name``. Without
       ``name``, the export is named after the file with its suffix and one
       trailing ``_raw`` or ``_eeg`` removed. With ``root``, ``name`` is not
       allowed and the tree below ``root`` is mirrored under ``directory``.
       Two recordings that would share a bundle are an error.
   * - ``workflow``
     - ``raw_review`` is ``required`` (default), ``suggested``, or
       ``disabled``. ``artifact_review`` is ``required`` (default) or
       ``suggested``. ``epoch_review`` is ``required``, ``optional``, or
       ``disabled``. ``suggested`` saves the detectors' verdict as the
       decision when the gate is reached without one, bound to the same
       parent checkpoint as a saved review.
   * - ``channels``
     - ``rename``, ``types``, ``drop``, ``bads``, ``montage`` (a standard name,
       or ``{path: ...}`` to a digitized FIF or any electrode file
       ``mne.channels.read_custom_montage`` reads, positions taken as
       written), ``interpolate_bads``, ``bipolar``
       (``name``, ``anode``, ``cathode``, ``type`` of ``eog`` or ``ecg``),
       ``projections`` (``error``, ``apply``, ``discard-inactive``).
   * - ``crop``
     - ``tmin``, ``tmax``, seconds from the start of the file. ``null`` skips it.
   * - ``annotations``
     - ``bad_spans`` (``onset``, ``duration``, ``description`` starting with
       ``BAD``), ``amplitude``, ``breaks``, ``muscle``. A null detector is skipped.
   * - ``bad_channels``
     - PyPREP. ``method: pyprep``. ``methods`` among ``flat``, ``deviation``,
       ``correlation``, ``high_frequency``, ``snr``. ``snr`` requires the last
       two. ``ransac``, ``random_state``, ``repeats`` (RANSAC draws that vote;
       a channel is a candidate when a strict majority flags it), and
       ``notch_freqs`` (a notch on the diagnostic copy only, as PREP does
       before its deviation test).
   * - ``bridges``
     - ``true`` records bridged pairs and their median electrical distance as
       raw-review evidence. ``false`` skips it.
   * - ``stimulation``
     - ``event_ids``, ``channels``, ``tmin``, ``tmax``, ``mode``
       (``linear``, ``window``, ``constant``). ``constant`` requires ``baseline``.
   * - ``filter``
     - ``l_freq``, ``h_freq``, ``notch_freqs``. A null cutoff or an empty notch
       list skips that stage. ``l_freq`` must be below ``h_freq``.
   * - ``artifact``
     - ``method`` of ``ica``, ``ssp``, or ``regression``, the matching settings
       block, and ``reference`` (``average``, a list of EEG names, or ``null``).
       Regression requires a reference. ``null`` skips fitting, review, and apply.
   * - ``epochs``
     - ``kind: events`` with ``events``, ``tmin``, ``tmax``. Or ``kind: fixed``
       with ``duration``, ``overlap``, ``start``, ``stop``. Both accept
       ``padding``, ``baseline``, and ``detrend`` (``constant`` or ``linear``).
       Event epochs also accept ``metadata``, a TSV with one row per event.
       ``metadata`` and ``events.path`` accept ``{name}`` (the export name)
       and ``{parent}`` (the recording's directory), resolved per recording.
   * - ``epochs.events``
     - ``source`` of ``annotations``, ``stim``, or ``file``. ``event_id`` maps
       names to codes. ``stim`` also takes ``stim_channel``, ``shortest_event``,
       and ``min_duration``. ``file`` takes ``path``. ``delay`` subtracts
       ``round(delay * sfreq)`` samples from every event, so a positive value
       moves events earlier.
   * - ``rejection``
     - ``method: thresholds`` with ``reject`` and ``flat`` in volts, and an
       optional ``tmin`` and ``tmax`` inside the analysis window. Or
       ``method: autoreject`` with ``n_interpolate``, ``consensus``, ``cv``,
       and ``random_state``.
   * - ``reference``
     - Final EEG reference. ``channels: average`` or a list of good EEG names.
       ``add_channels`` inserts missing reference electrodes first. ``null``
       leaves the reference unchanged.
   * - ``sampling``
     - ``method: decimate`` with integer ``factor``, which requires
       ``filter.h_freq``. Or ``method: resample`` with ``sfreq`` and ``padding``.
       The padding must equal ``epochs.padding``.

``annotations.amplitude`` takes ``peak`` and ``flat`` in volts for ``eeg`` only,
plus ``bad_percent`` and ``min_duration``. ``annotations.breaks`` takes
``min_break_duration``, ``t_start_after_previous``, and ``t_stop_before_next``,
and requires event epochs. ``annotations.muscle`` takes ``filter_freq``,
``threshold``, and ``min_length_good``.

ICA settings are ``method`` (``fastica``, the default, ``infomax`` with the
extended update, or ``picard`` with ``ortho: false`` and ``extended: true``),
``l_freq`` (default 1 Hz, on a copy), ``n_components``, ``random_state``,
``max_iter``, ``reject``, ``flat``, ``tstep``, ``eog_channels``,
``ecg_channel``, and ``iclabel``. ``iclabel`` takes ``threshold`` (default
0.8) and ``keep`` (default ``[brain, other]``); it requires ``infomax`` or
``picard``, ``artifact.reference: average``, ``l_freq`` of at least 1 Hz, and
a low-pass at or below 100 Hz, the band ICLabel was trained on, and runs
ICLabel on the training copy. The fit checkpoint records every detector's verdict as
evidence: ``scores`` per channel, ``iclabel`` labels and class probabilities,
``suggested`` per detector, and ``suggested_exclude``, the union of the
components MNE's EOG and ECG detectors flag and the components whose winning
ICLabel class is outside ``keep`` at or above ``threshold``. The fit excludes
nothing. SSP settings are ``n_eeg``,
``eog_channels`` or ``ecg_channel``, ``l_freq``, ``h_freq``, ``tmin``, ``tmax``,
and ``reject``. Regression settings are ``eog_channels``, ``tstep``, ``reject``,
and ``flat``.

Stages
------

Stages run in this order. A disabled stage is omitted and its children read the
previous enabled parent. ``apply-artifact`` waits for both ``epoch`` and
``review-artifact``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Stage
     - What it does
   * - ``load``
     - Read the file and reject non-finite data.
   * - ``prepare``
     - Rename and type channels, derive bipolar EOG or ECG, drop channels, set
       the montage, merge bad labels, and apply the projector policy.
   * - ``events``
     - Build the event table on the acquisition grid.
   * - ``crop-raw``
     - Keep ``[tmin, tmax]``. Event codes are unchanged.
   * - ``annotate``
     - Add the manual BAD spans, then amplitude, break, and muscle candidates.
   * - ``detect-bads``
     - PyPREP channel candidates and, when enabled, bridged pairs.
   * - ``review-raw``
     - Apply the saved bad channels and BAD spans.
   * - ``repair-stim``
     - ``mne.preprocessing.fix_stim_artifact`` on the declared windows.
   * - ``notch``, ``filter``
     - Zero-phase Hamming FIR (``firwin``) on physiology channels. BAD and edge
       annotations are skipped. A clean segment shorter than the filter is an error.
   * - ``artifact-reference``
     - Reference used for the artifact fit.
   * - ``fit-artifact``, ``review-artifact``
     - Fit ICA, EEG SSP, or EOG regression. Review stores the components to
       exclude, the projectors to apply, or whether to apply the regression.
   * - ``epoch``
     - ``mne.Epochs`` with ``baseline=None``, ``proj=False``, and ``decim=1``.
       BAD annotations drop epochs. Padding lies outside the analysis window.
       From here on a checkpoint holds epochs, not the continuous data.
   * - ``apply-artifact``
     - Apply the reviewed operator. The fit is not repeated.
   * - ``fit-rejection``, ``reject``
     - Peak-to-peak and flat thresholds, or autoreject, on the analysis window.
   * - ``review-epochs``
     - Drop epochs by original event row.
   * - ``interpolate``
     - Spline interpolation of bad EEG. Other bad labels stay.
   * - ``reference``
     - Average or named EEG reference.
   * - ``resample``
     - Integer decimation, after the low-pass guard, or polyphase resampling of
       the padded epochs. The output grid must contain the epoch origin.
   * - ``crop-epochs``, ``detrend``, ``baseline``
     - Remove padding, then optional SciPy detrending of EEG, then an optional
       baseline on the final grid.
   * - ``report``, ``export``
     - HTML report and the files below.

Event samples stay on the acquisition grid. The events table has the original
sample, the delay-corrected sample, and ``event_sample_sfreq``.
``final_sfreq`` in the manifest is the epoch rate after resampling. A
fixed-length epoch of ``duration`` seconds has ``round(duration * sfreq)``
samples, so its last time is ``(n_samples - 1) / sfreq``.

Review
------

``run`` returns at the first enabled review that has no saved decision and
no ``suggested`` policy. The pending file for that gate is
``<bundle directory>/.preprocessing/<name>/decisions/review-<target>.pending.yaml``,
YAML with a comment above each field. Leave ``parent_id``. Replace every
``null``, then run ``review`` for that gate: a filled pending file is used
before the viewer is opened. ``--decisions FILE`` reads another file instead.
The Qt viewer (``preprocessing-gui``) writes the same decision when the dialog
is accepted. The quick start above is the raw gate. The other gates use the
same steps and the fields below.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Target
     - Decision
   * - ``raw``
     - ``bads`` replaces ``info["bads"]``. ``[]`` clears it. ``spans`` are
       appended. ``[]`` appends none. Each span has ``onset``, ``duration``,
       and a ``description`` that starts with ``BAD``. ``onset`` is seconds
       from the start of the recording, like ``annotations.bad_spans``, also
       after ``crop-raw``. The viewer opens with the suggested bad channels and
       spans already marked and saves the spans added to them.
   * - ``artifact``, ICA
     - ``fit_id``. ``exclude``, component indices. ``[]`` excludes none.
   * - ``artifact``, SSP
     - ``fit_id``. ``include``, projector indices to apply.
   * - ``artifact``, regression
     - ``fit_id``. ``apply``, true or false.
   * - ``epochs``
     - ``exclude``, original event rows to drop.

``review --suggested``, and a ``suggested`` policy in ``workflow``, save the
detectors' own verdict as the decision without a viewer or a file: for
``raw`` the channels already marked bad plus the amplitude and PyPREP
candidates, and the detected BAD spans; for ``artifact`` the ICA
``suggested_exclude`` list, every SSP projector, or applying the regression.
There is no suggestion for ``epochs``. The saved decision is the same file a
viewer or ``--decisions`` would write, so provenance records the choice
either way, and a changed detector setting invalidates it like any decision.

A decision is kept. Replacing it requires ``reset --from`` that review stage.
A new fit requires a new decision. ``parent_id`` has to match the checkpoint
the decision was written for.

``inspect STAGE`` opens that checkpoint in the browser. ``--report`` writes an
HTML file beside the workspace and opens it: the continuous data before
epoching, the fitted operator at ``fit-artifact`` and ``review-artifact``, the
epochs after.

Terminal Front End
~~~~~~~~~~~~~~~~~~

``tui/`` holds an optional Go program that drives the same commands from one
screen: every recording and where it stands, a live ``run`` with its log kept
below the stage list, and each review gate as a checklist that opens pre-ticked with the detectors' verdict and
their reasons (PyPREP tests per channel, ICLabel class and confidence per
component, peak-to-peak amplitude per epoch). It writes decisions through
``review --decisions`` and reads state through ``status --json`` and
``inspect STAGE --json``, so nothing it does bypasses the checks above, and
the Python package does not depend on it. ``v`` opens the reviewed checkpoint
in the MNE viewer when ``preprocessing-gui`` is installed.

Lists scroll to keep the selected row visible. ``Home`` / ``End`` jump to the
first / last recording, stage, or review row; ``Page Up`` / ``Page Down`` page
through review rows (and scroll the run log on the home screen). Sorting a
review with ``o`` keeps the same item selected. While loading, starting,
saving, or resetting, the header shows the operation and further actions
wait for it to finish.

.. code-block:: bash

   cd tui && go build -o eegfeat-tui .     # Go 1.23+
   ./eegfeat-tui preprocessing.yaml        # finds eegfeat on PATH, or set EEGFEAT

``--json`` on ``status`` and ``inspect`` is a documented contract for any
front end: ``status --json`` lists each recording's stages and its ``next``
action (``review``, ``run`` or ``reset``); ``inspect review-raw --json``
returns the gate's ``items`` with ``suggested`` flags and ``tags``, its
``parent`` checkpoint, and the ``parent_id`` a decision must echo.

Outputs
-------

Export writes into ``output.directory``. The manifest is published last. A
directory without ``<name>_preprocessing.json`` has no finished export.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - File
     - Contents
   * - ``<name>_epo.fif``
     - Double-precision epochs. ``info["description"]`` names the manifest.
   * - ``<name>_events.tsv``
     - One row per original event, with ``retained``, ``epoch_row``, and
       ``drop_reason``.
   * - ``<name>_repairs.tsv``
     - Present when autoreject ran. Per-epoch channel repairs.
   * - ``<name>_report.html``
     - MNE report.
   * - ``<name>_preprocessing.json``
     - Provenance, package versions, stage order, settings, and file hashes.

Checkpoints live in ``<bundle directory>/.preprocessing/<name>/``, where the
bundle directory is ``output.directory`` plus the recording's path below
``input.root``. Hidden files the OS adds there, such as ``.DS_Store``, are
ignored. The feature runner skips that hidden directory. Point its
``inputs.root`` at ``output.directory`` and its pattern at ``**/*_epo.fif``;
``run`` prints that handoff when every recording is exported.

.. literalinclude:: ../../examples/preprocessing.yaml
   :language: yaml
   :caption: examples/preprocessing.yaml

Notes
-----

- Scalp EEG. There is no ASR, current-source density, source model, Maxwell
  filter, or EEG-fMRI gradient correction.
- ICA, SSP, EOG regression, and autoreject are fit on the recording being
  processed. A fit that must stay inside a training split uses the numerical
  functions in :mod:`eegfeat.preprocessing` directly.
- Muscle and high-frequency detectors need the unfiltered recording to cover
  the band they measure.
- A notch removes a line and does not change the highpass or lowpass stored on
  the epochs. The feature runner's passband check does not see that hole.
- ``check`` runs ``load``, ``prepare``, and ``events`` in memory. Later stages
  are checked when they run.
