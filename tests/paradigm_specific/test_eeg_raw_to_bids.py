import importlib.util
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

pytest.importorskip("mne_bids")

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "paradigm_specific"
    / "thermal_pain"
    / "eeg_raw_to_bids.py"
)
spec = importlib.util.spec_from_file_location("eeg_raw_to_bids", SCRIPT)
assert spec is not None and spec.loader is not None
convert = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = convert
spec.loader.exec_module(convert)


def mixed_raw():
    info = mne.create_info(["Fz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 2_000)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.9, 2.8, 4.0, 5.5, 9.0],
            duration=[0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            description=[
                "Volume/V  1",
                "Volume/V  1",
                "Volume/V  1",
                "Trig_therm/T  1",
                "Bad Interval/Bad_Gradient",
                "Comment/QC_ResidualHigh",
            ],
        )
    )
    return raw


def written(tmp_path, raw, **kwargs):
    source_eeg = tmp_path / "corrected" / "sub-0001" / "eeg"
    source_eeg.mkdir(parents=True)
    raw.save(
        source_eeg / "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_raw.fif",
        verbose=False,
    )
    convert.run_raw_to_bids(
        tmp_path / "corrected",
        tmp_path / "bids",
        "thermalactive",
        montage="",
        source_format="native-fif",
        **kwargs,
    )
    vhdr = tmp_path / "bids" / "sub-0001" / "eeg" / "sub-0001_task-thermalactive_run-1_eeg.vhdr"
    return mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)


def test_markers_keep_their_types_and_names(tmp_path):
    raw = mixed_raw()
    out = written(tmp_path, raw, keep_all_annotations=True)
    assert out.annotations.description.tolist() == raw.annotations.description.tolist()
    np.testing.assert_allclose(out.annotations.onset, raw.annotations.onset, atol=0.01)
    np.testing.assert_allclose(out.annotations.duration, raw.annotations.duration, atol=0.01)


def test_markers_stay_anchored_after_trimming(tmp_path):
    out = written(
        tmp_path, mixed_raw(), keep_all_annotations=True, trim_to_first_event_prefix="Trig_therm"
    )
    assert out.annotations.description.tolist() == [
        "Trig_therm/T  1",
        "Bad Interval/Bad_Gradient",
        "Comment/QC_ResidualHigh",
    ]
    np.testing.assert_allclose(out.annotations.onset, [0.0, 1.5, 5.0], atol=0.01)


def test_bad_intervals_survive_the_default_filter(tmp_path):
    out = written(tmp_path, mixed_raw())
    assert out.annotations.description.tolist() == ["Volume/V  1"] * 3 + [
        "Trig_therm/T  1",
        "Bad Interval/Bad_Gradient",
    ]


def test_a_bare_description_is_filed_as_a_comment():
    assert convert.brainvision_marker("BAD_manual") == ("Comment", "BAD_manual")
    assert convert.brainvision_marker("Trig_therm/T  1") == ("Trig_therm", "T  1")


def test_thermode_canonicalization_rewrites_exactly_eleven_legacy_markers():
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 2_000)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations([float(i) for i in range(11)], [0.0] * 11, ["Stim_on/S  1"] * 11)
    )
    assert convert.canonicalize_thermode_markers(raw)
    assert set(raw.annotations.description) == {"Trig_therm/T  1"}
    raw.set_annotations(mne.Annotations([0.0, 1.0], [0.0, 0.0], ["Stim_on/S  1"] * 2))
    with pytest.raises(ValueError, match="expected 11"):
        convert.canonicalize_thermode_markers(raw)


def test_source_glob_selects_only_the_paradigms_recordings(tmp_path):
    layout = tmp_path / "sub-0001" / "eeg" / "fastr"
    layout.mkdir(parents=True)
    for name in ("BaselineEEG_sub0001_fastr.vhdr", "ThermalPainEEGFMRI_run1_sub0001_fastr.vhdr"):
        (layout / name).write_text("")
    found = convert.find_source_files(
        tmp_path, "brainvision", "thermalactive", "fastr", "ThermalPain*_run*.vhdr"
    )
    assert [p.name for p in found] == ["ThermalPainEEGFMRI_run1_sub0001_fastr.vhdr"]
