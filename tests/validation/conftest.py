"""Opt-in validation against public MNE datasets.

These tests download real recordings from PhysioNet and the MNE servers and
take minutes to run, so they are skipped unless ``EEGFEAT_DATASETS=1`` is set.
Data lands wherever MNE keeps its datasets (``MNE_DATA``, default ``~/mne_data``)
and is reused on later runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from importlib.util import find_spec

import pytest

from validation.loaders import Recording, load_eegbci, load_sleep, load_ssvep

ENABLE = "EEGFEAT_DATASETS"

EEGBCI_SUBJECTS = tuple(range(1, 21))
SLEEP_SUBJECTS = (0, 1)

# Optional extras: eegfeat.model and the microstates need scikit-learn, the
# spectral connectivity estimators need mne-connectivity.
collect_ignore = []
if find_spec("sklearn") is None:
    collect_ignore += [
        "test_decoding.py",
        "test_microstates.py",
        "test_csp.py",
        "test_regression.py",
        "test_model_extras.py",
    ]
if find_spec("mne_connectivity") is None:
    collect_ignore += ["test_connectivity.py"]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if os.environ.get(ENABLE) == "1":
        return
    skip = pytest.mark.skip(reason=f"set {ENABLE}=1 to download and run the dataset checks")
    here = os.path.dirname(__file__)
    for item in items:
        if os.path.commonpath([str(item.path), here]) == here:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def quiet_mne() -> Iterator[None]:
    import mne

    previous = mne.set_log_level("ERROR", return_old_level=True)
    yield
    mne.set_log_level(previous)


@pytest.fixture(scope="session")
def eegbci_recordings() -> list[Recording]:
    return [load_eegbci(subject) for subject in EEGBCI_SUBJECTS]


@pytest.fixture(scope="session")
def sleep_recordings() -> list[Recording]:
    return [load_sleep(subject) for subject in SLEEP_SUBJECTS]


@pytest.fixture(scope="session")
def ssvep_recording() -> Recording:
    return load_ssvep()
