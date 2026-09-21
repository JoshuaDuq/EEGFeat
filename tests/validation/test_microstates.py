"""Microstates on resting motor-task epochs.

Four templates account for most of the resting topography's variance and last a
few tens of milliseconds each, as the microstate literature describes. The
per-state measures are also tied together by identities that must hold exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import eegfeat as ef
from validation.loaders import Recording

WINDOW = ef.Window("window", 0.5, 3.5)

DATASET = "eegbci"


@pytest.fixture(scope="module")
def segmentation(eegbci_recordings: list[Recording]) -> ef.MicrostateSegmentation:
    recording = eegbci_recordings[0]
    keep = recording.metadata["moving"].to_numpy() == 0
    signal = ef.Signal.from_epochs(recording.epochs[keep], recording=recording.name)
    return ef.segment(signal, n_states=4)


@pytest.mark.validates(
    "segment",
    kind="physiology",
    claim="Four resting microstate templates explain most of the topographic variance",
    criterion="global explained variance above 0.5; no two templates correlate above 0.9",
)
def test_four_templates_explain_most_of_the_variance(
    segmentation: ef.MicrostateSegmentation, record: Callable[[str], None]
) -> None:
    record(f"global explained variance {segmentation.global_explained_variance:.2f}")
    assert segmentation.global_explained_variance > 0.5
    correlation = np.abs(np.corrcoef(segmentation.templates))
    np.fill_diagonal(correlation, 0.0)
    assert correlation.max() < 0.9, "two templates describe the same map"


@pytest.mark.validates(
    "microstate_duration",
    kind="physiology",
    claim="Microstates last tens of milliseconds",
    criterion="median duration of every state between 30 and 150 ms",
)
def test_states_last_tens_of_milliseconds(
    segmentation: ef.MicrostateSegmentation, record: Callable[[str], None]
) -> None:
    duration = ef.microstate_duration(segmentation, windows=[WINDOW])
    medians = np.nanmedian(duration.values, axis=0)
    record(f"median durations {medians.min():.0f} to {medians.max():.0f} ms")
    assert np.all((medians > 30.0) & (medians < 150.0)), medians


@pytest.mark.validates(
    "microstate_coverage",
    "microstate_duration",
    "microstate_occurrence",
    "microstate_transitions",
    kind="formula",
    claim="Coverage sums to one and equals duration times occurrence; transitions sum to one",
    criterion="relative error below 1e-9",
)
def test_measures_obey_their_identities(segmentation: ef.MicrostateSegmentation) -> None:
    coverage = ef.microstate_coverage(segmentation, windows=[WINDOW]).values
    duration = ef.microstate_duration(segmentation, windows=[WINDOW]).values
    occurrence = ef.microstate_occurrence(segmentation, windows=[WINDOW]).values
    transitions = ef.microstate_transitions(segmentation, windows=[WINDOW])

    np.testing.assert_allclose(coverage.sum(axis=1), 1.0)
    # Coverage is the mean visit length (ms) times how often the state is visited (1/s).
    np.testing.assert_allclose(coverage, duration / 1000.0 * occurrence, rtol=1e-9)
    # Each source state's outgoing probabilities sum to one.
    for source in segmentation.labels:
        from_source = [
            i for i, meta in enumerate(transitions.meta) if meta.space.startswith(f"{source}-to-")
        ]
        assert len(from_source) == segmentation.n_states - 1
        np.testing.assert_allclose(np.nansum(transitions.values[:, from_source], axis=1), 1.0)
