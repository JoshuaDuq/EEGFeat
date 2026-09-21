"""Complexity measures on Sleep-EDF.

Slow-wave sleep is the most regular EEG a healthy adult produces, so sample
entropy and Higuchi's fractal dimension both fall from wake to N3. Sample entropy
is quadratic in the window length, so a stratified subsample of epochs is used.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import eegfeat as ef
from validation.loaders import Recording

WHOLE_EPOCH = ef.Window("epoch", 0.0, 30.0)
PER_STAGE = 40

DATASET = "sleep"


@pytest.fixture(scope="module")
def subsample(sleep_recordings: list[Recording]) -> tuple[ef.Signal, np.ndarray]:
    recording = sleep_recordings[0]
    stage = recording.metadata["stage"].to_numpy()
    rng = np.random.default_rng(0)
    keep = np.zeros(stage.size, dtype=bool)
    for name in ("W", "N3"):
        keep[rng.choice(np.flatnonzero(stage == name), size=PER_STAGE, replace=False)] = True
    signal = ef.Signal.from_epochs(recording.epochs[keep], recording=recording.name)
    return signal, stage[keep]


@pytest.mark.validates(
    "sample_entropy",
    "higuchi_fractal_dimension",
    kind="physiology",
    claim="Sample entropy and Higuchi dimension are lower in N3 than in wake",
    criterion="wake-vs-N3 AUC below 0.2 on both derivations",
)
@pytest.mark.parametrize("measure", [ef.sample_entropy, ef.higuchi_fractal_dimension])
def test_deep_sleep_is_more_regular_than_wake(
    subsample: tuple[ef.Signal, np.ndarray], measure: object, record: Callable[[str], None]
) -> None:
    signal, stage = subsample
    table = measure([signal], windows=[WHOLE_EPOCH], include_global=False)  # type: ignore[operator]
    deep = (stage == "N3").astype(int)
    aucs = [roc_auc_score(deep, table.values[:, column]) for column in range(len(table.meta))]
    record(
        f"{measure.__name__}: AUC "  # type: ignore[attr-defined]
        + ", ".join(f"{meta.space} {auc:.2f}" for meta, auc in zip(table.meta, aucs, strict=True))
    )
    for column, meta in enumerate(table.meta):
        assert np.isfinite(table.values[:, column]).all()
        assert roc_auc_score(deep, table.values[:, column]) < 0.2, meta.space


@pytest.mark.validates(
    "multiscale_entropy",
    "sample_entropy",
    kind="formula",
    claim="Multiscale entropy at scale one is sample entropy",
    criterion="relative error below 1e-9",
)
def test_multiscale_entropy_at_scale_one_is_sample_entropy(
    subsample: tuple[ef.Signal, np.ndarray],
) -> None:
    signal, _ = subsample
    few = ef.Signal.from_arrays(
        data=signal.data[:8],
        times=signal.times,
        ch_names=signal.ch_names,
        sfreq=signal.sfreq,
        row_ids=signal.row_ids[:8],
    )
    plain = ef.sample_entropy([few], windows=[WHOLE_EPOCH], include_global=False)
    scaled = ef.multiscale_entropy(
        [few], windows=[WHOLE_EPOCH], scales=(1, 2), include_global=False
    )
    np.testing.assert_allclose(scaled.select(measure="mse01").values, plain.values, rtol=1e-9)
