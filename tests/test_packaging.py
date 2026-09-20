import tomllib
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_packaging_only_advertises_implemented_features() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    optional_dependencies = configuration["project"]["optional-dependencies"]

    assert "knee" not in optional_dependencies
    assert "specparam" not in (ROOT / "pyproject.toml").read_text()
    assert "specparam" not in (ROOT / "README.md").read_text().lower()
    assert "specparam" not in (ROOT / "docs" / "install.rst").read_text().lower()


def test_package_description_represents_the_full_feature_scope() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert configuration["project"]["description"] == (
        "Labelled EEG feature extraction and modeling for MNE objects"
    )


def test_ci_covers_supported_endpoints_optional_integrations_and_the_wheel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert 'python-version: ["3.11", "3.14"]' in workflow
    assert ".[dev,model,connectivity,microstates]" in workflow
    assert "python -m build" in workflow
    assert "pip install dist/*.whl" in workflow
    assert 'python -c "import eegfeat"' in workflow
    assert 'MNE_DONTWRITE_HOME: "true"' in workflow


@pytest.mark.skipif(find_spec("sklearn") is not None, reason="scikit-learn is installed")
def test_the_model_subpackage_names_its_extra_when_scikit_learn_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'eegfeat\[model\]'"):
        import eegfeat.model  # noqa: F401
