from eegtable.preprocessing.pipeline import OPERATIONS
from eegtable.preprocessing.stages import STAGES


def test_complete_acyclic_callable_catalog():
    seen = set()
    for stage in STAGES:
        assert stage.name not in seen
        assert set(stage.parents) <= seen
        assert callable(OPERATIONS[stage.name])
        assert stage.output
        seen.add(stage.name)
    assert len(STAGES) == 26 == len(OPERATIONS)


def test_artifact_reference_identity_owns_restored_electrodes():
    from eegtable.preprocessing.config import (
        ArtifactSettings,
        FixedEpochSettings,
        ICASettings,
        ProcessingSettings,
        ReferenceSettings,
    )
    from eegtable.preprocessing.stages import get_stage, stage_settings

    settings = ProcessingSettings(
        FixedEpochSettings(2),
        artifact=ArtifactSettings("ica", ICASettings(), "average"),
        reference=ReferenceSettings("average", ("Cz",)),
    )
    fields = stage_settings(get_stage("artifact-reference"), settings)
    assert fields["reference.add_channels"] == ["Cz"]
