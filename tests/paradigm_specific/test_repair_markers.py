import importlib.util
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eegfeat.preprocessing.provenance import file_hash

SCRIPT = (
    Path(__file__).resolve().parents[2] / "paradigm_specific" / "thermal_pain" / "repair_markers.py"
)
spec = importlib.util.spec_from_file_location("repair_markers", SCRIPT)
assert spec is not None and spec.loader is not None
repair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = repair
spec.loader.exec_module(repair)

HEAD = (
    "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
    "[Common Infos]\nDataFile={data}\n\n[Marker Infos]\n"
)
NAMED = [
    "Volume,V  1,1,1,0",
    "Volume,V  1,901,1,0",
    "Trig_therm,T  1,300,1,0",
    "Bad Interval,Bad_Gradient,700,50,0",
]
GENERIC = ["Stimulus,S  2,1,1,0", "Stimulus,S  1,300,1,0", "Stimulus,S  2,901,1,0"]


def vmrk(path, data, markers):
    path.write_text(
        HEAD.format(data=data) + "\n".join(f"Mk{i}={m}" for i, m in enumerate(markers, 1)) + "\n"
    )


def test_splice_keeps_the_derivatives_header_and_takes_the_named_markers(tmp_path):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    vmrk(fixed, "fixed.eeg", NAMED)
    vmrk(old, "sub-01_desc-bcgnet_eeg.eeg", GENERIC)
    repair.splice_markers(fixed, old)
    text = old.read_text()
    assert "DataFile=sub-01_desc-bcgnet_eeg.eeg" in text
    assert [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("Mk")] == NAMED


def test_splice_refuses_when_the_generic_positions_do_not_match(tmp_path):
    fixed, old = tmp_path / "fixed.vmrk", tmp_path / "old.vmrk"
    vmrk(fixed, "fixed.eeg", NAMED)
    vmrk(old, "old.eeg", ["Stimulus,S  2,5,1,0", "Stimulus,S  1,300,1,0", "Stimulus,S  2,901,1,0"])
    with pytest.raises(ValueError, match="positions"):
        repair.splice_markers(fixed, old)
    assert "Stimulus,S  2,5,1,0" in old.read_text()


def bundle(tmp_path):
    stem = tmp_path / "sub-01_task-thermalactive_run-1"
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    events = np.array([[50, 0, 1], [150, 0, 1]])
    epochs = mne.EpochsArray(
        np.zeros((2, 1, 10)), info, events=events, event_id={"Stimulus/S  1": 1}
    )
    epochs.save(f"{stem}_epo.fif", verbose=False)
    pd.DataFrame({"label": ["Stimulus/S  1", "Stimulus/S  1"], "retained": [True, False]}).to_csv(
        f"{stem}_events.tsv", sep="\t", index=False
    )
    Path(f"{stem}_recipe.yaml").write_text(
        'epochs:\n  events: {source: annotations, event_id: {"Stimulus/S  1": 1}}\n'
    )
    manifest = {
        "schema": 1,
        "files": {
            f"{stem.name}_epo.fif": file_hash(Path(f"{stem}_epo.fif")),
            f"{stem.name}_events.tsv": file_hash(Path(f"{stem}_events.tsv")),
        },
        "provenance": {"events": {"event_id": {"Stimulus/S  1": 1}}},
    }
    Path(f"{stem}_preprocessing.json").write_text(json.dumps(manifest))
    return stem


def test_bundle_repair_renames_everywhere_and_rehashes(tmp_path):
    stem = bundle(tmp_path)
    repair.repair_bundle(Path(f"{stem}_preprocessing.json"))
    assert mne.read_epochs(f"{stem}_epo.fif", verbose=False).event_id == {"Trig_therm/T  1": 1}
    assert pd.read_csv(f"{stem}_events.tsv", sep="\t")["label"].tolist() == ["Trig_therm/T  1"] * 2
    assert '"Trig_therm/T  1": 1' in Path(f"{stem}_recipe.yaml").read_text()
    manifest = json.loads(Path(f"{stem}_preprocessing.json").read_text())
    assert manifest["provenance"]["events"]["event_id"] == {"Trig_therm/T  1": 1}
    for name, digest in manifest["files"].items():
        assert file_hash(tmp_path / name) == digest


def test_only_live_bundles_are_repaired(tmp_path):
    for folder in ("sub-01/eeg", "_superseded_v1/sub-01/eeg"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "sub-01_task-thermalactive_run-1_preprocessing.json").write_text(
            '{"provenance": {"event_id": {"Stimulus/S  1": 1}}}'
        )
    (tmp_path / "sub-01/eeg/._sub-01_task-thermalactive_run-1_preprocessing.json").write_bytes(
        b"\x00\x05\x16\x07"
    )
    assert [p.parent.parent.name for p in repair.live_bundles(tmp_path)] == ["sub-01"]
