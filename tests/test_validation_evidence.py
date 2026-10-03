import json

import pytest
from tests.validation.report import Row, write


def test_validation_runs_are_immutable_and_do_not_merge_old_claims(tmp_path):
    old = Row("old::claim", ("variance",), "formula", "", "old", "exact", "1", True, "old")
    new = Row("new::claim", ("variance",), "formula", "", "new", "exact", "2", True, "new")
    write([old], tmp_path)
    snapshots = list((tmp_path / "runs").glob("*/results.json"))
    assert len(snapshots) == 1
    original = snapshots[0].read_bytes()
    write([new], tmp_path)
    current = json.loads((tmp_path / "results.json").read_text())
    assert [row["nodeid"] for row in current["rows"]] == ["new::claim"]
    assert len(list((tmp_path / "runs").glob("*/results.json"))) == 2
    assert snapshots[0].read_bytes() == original
    assert len(current["code_sha256"]) == 64
    assert current["run_id"]


def test_validation_refuses_evidence_from_a_changed_implementation(tmp_path):
    row = Row("claim", ("variance",), "formula", "", "claim", "exact", "1", True, "now")
    with pytest.raises(ValueError, match="implementation changed"):
        write([row], tmp_path, code_sha256="0" * 64)
    assert not (tmp_path / "results.json").exists()
