from importlib.util import find_spec
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


def test_knee_extra_accepts_the_published_specparam_release_candidate() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text()
    requirement_text = text.split('knee = ["', 1)[1].split('"]', 1)[0]
    requirement = Requirement(requirement_text)

    assert Version("2.0.0rc7") in requirement.specifier


@pytest.mark.skipif(find_spec("sklearn") is not None, reason="scikit-learn is installed")
def test_the_model_subpackage_names_its_extra_when_scikit_learn_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'eegfeat\[model\]'"):
        import eegfeat.model  # noqa: F401
