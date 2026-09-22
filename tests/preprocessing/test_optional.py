import os
import subprocess
import sys


def test_core_cli_does_not_import_optional_dependencies():
    code = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('yaml', 'sklearn', 'pyprep', 'autoreject', 'PyQt6'):
        raise AssertionError('unexpected optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import eegfeat
from eegfeat.runner.cli import main
try:
    main(['--version'])
except SystemExit as error:
    assert error.code == 0
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "MNE_DONTWRITE_HOME": "true"},
        capture_output=True,
    )


def test_missing_extra_names_its_installation():
    code = """
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] == 'yaml':
        raise ModuleNotFoundError(name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from eegfeat.preprocessing import load_config
try:
    load_config('missing.yaml')
except ModuleNotFoundError as error:
    assert "pip install 'eegfeat[preprocessing]'" in str(error)
else:
    raise AssertionError('YAML import was not required')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "MNE_DONTWRITE_HOME": "true"},
        capture_output=True,
    )
