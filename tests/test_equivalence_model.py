from __future__ import annotations

import pathlib

import numpy as np
import pytest

from eegfeat.model.splits import loso_folds

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "model_reference.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="model fixtures not generated")


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    return dict(np.load(FIXTURE, allow_pickle=False))


def _unflatten(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def test_loso_folds_match_the_reference_pipeline(reference: dict[str, np.ndarray]) -> None:
    train = _unflatten(reference["loso_train"], reference["loso_train_offsets"])
    test = _unflatten(reference["loso_test"], reference["loso_test_offsets"])
    folds = loso_folds(reference["groups"].astype(object))
    assert len(folds) == len(train)
    for fold, expected_train, expected_test in zip(folds, train, test, strict=True):
        np.testing.assert_array_equal(fold.train, expected_train)
        np.testing.assert_array_equal(fold.test, expected_test)
