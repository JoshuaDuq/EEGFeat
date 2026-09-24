"""Load public MNE datasets into the epochs the validation suite runs on.

Each loader follows the corresponding MNE tutorial, so the epochs here are the
ones the literature on these datasets describes. Nothing is preprocessed beyond
what the tutorials do: the point is to see the known effects come out of eegfeat
on data it has never been tuned on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci, erp_core, sleep_physionet, ssvep

# PhysioNet motor task: runs 3, 7 and 11 are real left- and right-fist movement,
# cued at time zero and held for about four seconds, with rest between trials.
EEGBCI_RUNS = (3, 7, 11)
EEGBCI_KINDS = {"T0": "rest", "T1": "left", "T2": "right"}
EEGBCI_EVENTS = {"rest": 1, "left": 2, "right": 3}
EEGBCI_TMIN, EEGBCI_TMAX = -1.0, 4.0

SSVEP_EVENTS = {"12hz": 255, "15hz": 155}
SSVEP_TMIN, SSVEP_TMAX = -1.0, 20.0
SSVEP_RESAMPLE_HZ = 250.0

# Stages 3 and 4 are one class, as in the AASM scoring MNE's tutorial adopts.
SLEEP_EVENTS = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
SLEEP_STAGES = {1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "R"}
SLEEP_EPOCH_SEC = 30.0


@dataclass(frozen=True)
class Recording:
    """Epochs from one recording, with the metadata the checks condition on."""

    name: str
    epochs: Any

    @property
    def metadata(self) -> pd.DataFrame:
        frame = self.epochs.metadata
        assert frame is not None
        return frame


def load_eegbci(subject: int, *, tmax: float = EEGBCI_TMAX) -> Recording:
    """One subject's three motor-execution runs as rest, left-hand and right-hand epochs.

    ``tmax`` extends the epoch past the four-second cue when a check needs to see
    what follows the movement, such as the post-movement beta rebound.
    """
    paths = eegbci.load_data(subject, list(EEGBCI_RUNS), update_path=False, verbose="error")
    raws = []
    for run, path in zip(EEGBCI_RUNS, paths, strict=True):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        # Each run is annotated separately; tagging the descriptions keeps the run
        # identity through concatenation so folds can stay run-disjoint.
        raw.annotations.description = np.array(
            [f"{d}/run-{run:02d}" for d in raw.annotations.description]
        )
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_eeg_reference("average", projection=False, verbose="error")

    kinds = list(EEGBCI_KINDS)
    event_id = {
        f"{kind}/run-{run:02d}": len(EEGBCI_RUNS) * kind_index + run_index
        for kind_index, kind in enumerate(kinds)
        for run_index, run in enumerate(EEGBCI_RUNS)
    }
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="error")
    kind_index, run_index = np.divmod(events[:, 2], len(EEGBCI_RUNS))
    condition = np.array([EEGBCI_KINDS[kinds[k]] for k in kind_index])
    events[:, 2] = [EEGBCI_EVENTS[name] for name in condition]
    metadata = pd.DataFrame(
        {
            "subject": f"S{subject:03d}",
            "run": [f"run-{EEGBCI_RUNS[i]:02d}" for i in run_index],
            "condition": condition,
            "moving": (condition != "rest").astype(int),
            "right_hand": (condition == "right").astype(int),
        }
    )
    epochs = mne.Epochs(
        raw,
        events,
        EEGBCI_EVENTS,
        tmin=EEGBCI_TMIN,
        tmax=tmax,
        baseline=None,
        metadata=metadata,
        preload=True,
        verbose="error",
    )
    return Recording(f"S{subject:03d}", epochs)


def load_ssvep() -> Recording:
    """The single SSVEP subject: twenty 20-second trials of 12 Hz or 15 Hz flicker."""
    root = ssvep.data_path(update_path=False, verbose="error")
    vhdr = root / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    raw.info["line_freq"] = 50.0
    raw.set_eeg_reference("average", projection=False, verbose="error")
    raw.filter(l_freq=0.1, h_freq=None, verbose="error")
    events, _ = mne.events_from_annotations(raw, verbose="error")
    # Resampled after event extraction so the triggers keep their sample positions.
    raw, events = raw.resample(SSVEP_RESAMPLE_HZ, events=events, verbose="error")
    epochs = mne.Epochs(
        raw,
        events,
        SSVEP_EVENTS,
        tmin=SSVEP_TMIN,
        tmax=SSVEP_TMAX,
        baseline=None,
        preload=True,
        verbose="error",
    )
    condition = np.where(epochs.events[:, 2] == SSVEP_EVENTS["12hz"], "12hz", "15hz")
    epochs.metadata = pd.DataFrame(
        {
            "subject": "sub-02",
            "condition": condition,
            "flicker_hz": np.where(condition == "12hz", 12.0, 15.0),
        }
    )
    return Recording("sub-02", epochs)


def load_sleep(subject: int) -> Recording:
    """One Sleep-EDF subject's first night as 30-second staged epochs."""
    [(psg, hypnogram)] = sleep_physionet.age.fetch_data(
        subjects=[subject], recording=[1], verbose="error"
    )
    raw = mne.io.read_raw_edf(
        psg, stim_channel="Event marker", infer_types=True, preload=True, verbose="error"
    )
    annotations = mne.read_annotations(hypnogram)
    # Sleep-EDF nights are bracketed by hours of wake; MNE's tutorial keeps 30
    # minutes on either side of sleep so the wake class is not the whole night.
    annotations.crop(
        annotations[1]["onset"] - 30 * 60, annotations[-2]["onset"] + 30 * 60, verbose="error"
    )
    raw.set_annotations(annotations, emit_warning=False)
    events, _ = mne.events_from_annotations(
        raw, event_id=SLEEP_EVENTS, chunk_duration=SLEEP_EPOCH_SEC, verbose="error"
    )
    stage = np.array([SLEEP_STAGES[int(code)] for code in events[:, 2]])
    metadata = pd.DataFrame({"subject": f"SC4{subject:02d}1", "stage": stage})
    epochs = mne.Epochs(
        raw,
        events,
        {name: code for code, name in SLEEP_STAGES.items()},
        tmin=0.0,
        tmax=SLEEP_EPOCH_SEC - 1.0 / raw.info["sfreq"],
        baseline=None,
        picks="eeg",
        metadata=metadata,
        preload=True,
        verbose="error",
    )
    return Recording(f"SC4{subject:02d}1", epochs)


# ERP CORE Flankers task (Kappenman et al., 2021): one subject, arrows with
# congruent or incongruent flankers, a left or right button press per trial.
ERP_CORE_FILE = "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
ERP_EOG = {"HEOG_left": "eog", "HEOG_right": "eog", "VEOG_lower": "eog"}
ERP_FILTER_HZ = (0.1, 30.0)


def load_erp_core(locked: Literal["stimulus", "response"]) -> Recording:
    """The Flankers task epoched on the stimulus or on the button press.

    Every epoch carries ``correct``: whether the response matched the target's
    side. Stimulus epochs are baseline-corrected on the 200 ms before onset;
    response epochs on -400 to -200 ms, as the ERP CORE pipeline does.
    """
    root = erp_core.data_path(update_path=False, verbose="error")
    raw = mne.io.read_raw_fif(root / ERP_CORE_FILE, preload=True, verbose="error")
    raw.set_channel_types(ERP_EOG)
    raw.filter(*ERP_FILTER_HZ, picks="eeg", verbose="error")
    events, event_id = mne.events_from_annotations(raw, verbose="error")
    names = np.array([{v: k for k, v in event_id.items()}[code] for code in events[:, 2]])
    is_stimulus = np.char.startswith(names, "stimulus")

    # Each stimulus is followed by its response; pair them by order.
    stimulus_rows = np.flatnonzero(is_stimulus)
    response_rows = np.flatnonzero(~is_stimulus)
    pairs = []
    for stimulus in stimulus_rows:
        following = response_rows[response_rows > stimulus]
        if following.size:
            pairs.append((stimulus, following[0]))
    pairs_array = np.array(pairs)
    target = np.where(np.char.find(names[pairs_array[:, 0]], "target_left") >= 0, "left", "right")
    pressed = np.where(np.char.endswith(names[pairs_array[:, 1]], "left"), "left", "right")
    metadata = pd.DataFrame(
        {
            "subject": "sub-001",
            "target": target,
            "pressed": pressed,
            # A path segment, not a substring: "incompatible" contains "compatible".
            "congruent": np.char.find(names[pairs_array[:, 0]], "/compatible/") >= 0,
            "correct": (target == pressed).astype(int),
        }
    )
    if locked == "stimulus":
        selected, tmin, tmax, baseline = pairs_array[:, 0], -0.2, 0.8, (-0.2, 0.0)
    else:
        selected, tmin, tmax, baseline = pairs_array[:, 1], -0.6, 0.4, (-0.4, -0.2)
    chosen = events[selected].copy()
    chosen[:, 2] = 1
    epochs = mne.Epochs(
        raw,
        chosen,
        {locked: 1},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks="eeg",
        metadata=metadata,
        preload=True,
        verbose="error",
    )
    return Recording(f"sub-001_{locked}", epochs)
