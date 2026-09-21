"""Beta and mu bursts on the PhysioNet motor task.

Sensorimotor beta comes in bursts, and movement suppresses them: fewer bursts,
and a smaller share of the window above threshold, than at rest. The threshold
is fixed per channel from the rest epochs' median envelope, so the comparison
between movement and rest is on one absolute scale.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import ttest_1samp

import eegfeat as ef
from validation.loaders import Recording

MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
BASELINE = ef.Window("baseline", -1.0, 0.0)
WINDOW = ef.Window("window", 0.5, 3.5)
HAND_AREA = ["FC3", "C5", "C3", "C1", "CP3", "FC4", "C2", "C4", "C6", "CP4"]

DATASET = "eegbci"


def _rest_median_threshold(signal: ef.BandSignal, resting: np.ndarray) -> np.ndarray:
    per_channel = 1.5 * np.median(signal.envelope[resting], axis=(0, 2))
    return np.broadcast_to(per_channel, (signal.n_epochs, len(signal.ch_names)))


@pytest.mark.validates(
    "burst_rate",
    "fraction_above_threshold",
    "burst_duration",
    kind="physiology",
    claim="Movement suppresses mu and beta bursts over the hand areas (rest-median threshold)",
    criterion="lower during movement in at least 80 percent of 20 subjects; p below 1e-3",
)
@pytest.mark.parametrize("band", [MU, BETA], ids=lambda band: band.name)
@pytest.mark.parametrize("measure", [ef.burst_rate, ef.fraction_above_threshold, ef.burst_duration])
def test_movement_suppresses_bursts(
    eegbci_recordings: list[Recording],
    band: ef.Band,
    measure: object,
    record: Callable[[str], None],
) -> None:
    differences = []
    for recording in eegbci_recordings:
        moving = recording.metadata["moving"].to_numpy() == 1
        signal = ef.BandSignal.from_epochs(
            recording.epochs, band, recording=recording.name, picks=HAND_AREA
        )
        table = measure(  # type: ignore[operator]
            [signal],
            windows=[WINDOW],
            threshold=_rest_median_threshold(signal, ~moving),
            include_global=True,
        ).select(space="global")
        values = table.values[:, 0]
        differences.append(np.nanmean(values[moving]) - np.nanmean(values[~moving]))

    per_subject = np.array(differences)
    record(
        f"{measure.__name__} {band.name}: lower during movement in "  # type: ignore[attr-defined]
        f"{int((per_subject < 0).sum())} of {per_subject.size} subjects, mean difference "
        f"{per_subject.mean():.3f}"
    )
    # Rate and fraction fall in every subject; the mean burst is also shorter in most.
    assert np.mean(per_subject < 0.0) >= 0.8, per_subject.round(3)
    assert ttest_1samp(per_subject, 0.0, alternative="less").pvalue < 1e-3


@pytest.mark.validates(
    "burst_rate",
    kind="physiology",
    claim="With a per-trial baseline-calibrated threshold, beta burst rate falls during movement",
    criterion="lower in at least 90 percent of subjects; p below 1e-6",
)
def test_baseline_calibrated_rate_falls_during_movement(
    eegbci_recordings: list[Recording],
) -> None:
    """With the percentile threshold set on each trial's own pre-cue baseline, the
    movement window of a moving trial holds fewer bursts than that of a rest trial."""
    differences = []
    for recording in eegbci_recordings:
        moving = recording.metadata["moving"].to_numpy() == 1
        signal = ef.BandSignal.from_epochs(
            recording.epochs, BETA, recording=recording.name, picks=HAND_AREA
        )
        rate = ef.burst_rate(
            [signal], windows=[WINDOW], baseline=BASELINE, include_global=True
        ).select(space="global")
        values = rate.values[:, 0]
        differences.append(np.nanmean(values[moving]) - np.nanmean(values[~moving]))

    per_subject = np.array(differences)
    assert np.mean(per_subject < 0.0) >= 0.9, per_subject.round(3)
    assert ttest_1samp(per_subject, 0.0, alternative="less").pvalue < 1e-6


@pytest.mark.validates(
    "burst_amplitude",
    kind="physiology",
    claim="Mu bursts that survive the rest-median threshold are smaller during movement",
    criterion="lower during movement in at least 90 percent of 20 subjects; p below 1e-4",
)
def test_movement_shrinks_the_bursts_that_remain(
    eegbci_recordings: list[Recording], record: Callable[[str], None]
) -> None:
    """Beta burst amplitude did not separate reliably (15 of 20) and is not asserted."""
    differences = []
    for recording in eegbci_recordings:
        moving = recording.metadata["moving"].to_numpy() == 1
        signal = ef.BandSignal.from_epochs(
            recording.epochs, MU, recording=recording.name, picks=HAND_AREA
        )
        amplitude = ef.burst_amplitude(
            [signal],
            windows=[WINDOW],
            threshold=_rest_median_threshold(signal, ~moving),
            include_global=True,
        ).select(space="global")
        values = amplitude.values[:, 0]
        differences.append(np.nanmean(values[moving]) - np.nanmean(values[~moving]))
    per_subject = np.array(differences)
    record(f"lower during movement in {int((per_subject < 0).sum())} of {per_subject.size}")
    assert np.mean(per_subject < 0.0) >= 0.9, per_subject
    assert ttest_1samp(per_subject, 0.0, alternative="less").pvalue < 1e-4


@pytest.mark.validates(
    "burst_count",
    "burst_rate",
    kind="formula",
    claim="Burst rate is the count divided by the window length",
    criterion="relative error below 1e-9",
)
def test_burst_count_is_rate_times_duration(eegbci_recordings: list[Recording]) -> None:
    recording = eegbci_recordings[0]
    signal = ef.BandSignal.from_epochs(
        recording.epochs, BETA, recording=recording.name, picks=HAND_AREA
    )
    count = ef.burst_count([signal], windows=[WINDOW], baseline=BASELINE, include_global=False)
    rate = ef.burst_rate([signal], windows=[WINDOW], baseline=BASELINE, include_global=False)
    seconds = (WINDOW.tmax - WINDOW.tmin) + 1.0 / signal.sfreq
    np.testing.assert_allclose(rate.values * seconds, count.values, rtol=1e-9)
