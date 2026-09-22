import mne
import numpy as np
import pytest

from eegfeat.preprocessing.config import FixedEpochSettings, OutputSettings, ProcessingSettings
from eegfeat.preprocessing.pipeline import preprocess


def test_export_bundle(raw, tmp_path):
    from eegfeat.preprocessing.io import validate_bundle, write_result

    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(2)))
    output = OutputSettings(tmp_path, "subject")
    manifest = write_result(result, output)
    validate_bundle(manifest)
    restored = mne.read_epochs(tmp_path / "subject_epo.fif", preload=True, proj=False)
    np.testing.assert_allclose(restored.get_data(), result.epochs.get_data(), rtol=0, atol=1e-18)
    with pytest.raises(FileExistsError):
        write_result(result, output)


def test_export_keeps_float_metadata(raw, tmp_path):
    import pandas as pd

    from eegfeat.preprocessing.config import EventEpochSettings, EventSettings
    from eegfeat.preprocessing.io import write_result

    # MNE stores epoch metadata as JSON at 10 significant digits; ordinary reaction
    # times have more, and that documented rounding must not fail the export.
    trials = tmp_path / "trials.tsv"
    pd.DataFrame({"rt": np.linspace(0.2, 0.9, 5) + 1e-11 * np.pi}).to_csv(
        trials, sep="\t", index=False
    )
    events = EventSettings("stim", {"stimulus": 1}, stim_channel="STI", shortest_event=1)
    result = preprocess(
        raw, ProcessingSettings(EventEpochSettings(events, -0.2, 0.8, metadata=trials))
    )
    write_result(result, OutputSettings(tmp_path / "out", "subject"))
    restored = mne.read_epochs(tmp_path / "out" / "subject_epo.fif", preload=True, proj=False)
    np.testing.assert_allclose(restored.metadata["rt"], result.epochs.metadata["rt"], rtol=1e-9)
