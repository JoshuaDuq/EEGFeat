from eegfeat.preprocessing.pipeline import OPERATIONS
from eegfeat.preprocessing.stages import STAGES


def test_complete_acyclic_callable_catalog():
    seen = set()
    for stage in STAGES:
        assert stage.name not in seen
        assert set(stage.parents) <= seen
        assert callable(OPERATIONS[stage.name])
        assert stage.output
        seen.add(stage.name)
    assert len(STAGES) == 26 == len(OPERATIONS)
