import numpy as np

from eegfeat.preprocessing import preprocess
from eegfeat.preprocessing.config import FixedEpochSettings, ICASettings, ProcessingSettings
from eegfeat.preprocessing.ica import fit_ica
from eegfeat.preprocessing.report import build_artifact_report, build_report


def test_report_is_headless_and_leaves_result_unchanged(raw, tmp_path):
    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(2)))
    before = result.epochs.get_data().copy()
    path = tmp_path / "report.html"
    build_report(result).save(path, open_browser=False)
    html = path.read_text()
    assert "Provenance" in html and "Event and rejection ledger" in html and "original_row" in html
    np.testing.assert_array_equal(result.epochs.get_data(), before)


def test_artifact_report_shows_every_component(mixture, tmp_path):
    model = fit_ica(mixture, ICASettings(n_components=4))
    path = tmp_path / "ica.html"
    build_artifact_report(model, mixture).save(path, open_browser=False)
    html = path.read_text()
    assert all(f"component {index}." in html for index in range(4))
    assert "rank" in html
