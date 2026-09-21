"""ERDS shape measures on the motor task, with six-second epochs.

Beyond the mean, the ERDS trace has a magnitude and duration on each side of
zero, a slope, and latencies. Movement deepens and lengthens the desynchronized
part of the trace and shortens the synchronized part; those are asserted across
subjects. The measures are also tied to each other and to the trace itself by
identities that must hold exactly.

Two things did not show up and are not asserted. The post-movement beta rebound
was not visible at the group level in the 4.5 to 5.8 s window. And the onset
latency fires on rest trials, where nothing happens, on essentially every trial
at a fixed 100 ms persistence; the default persistence of six cycles of the
band's low edge was chosen from that null, and its false-onset rate is asserted.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import ttest_1samp

import eegfeat as ef
from validation.loaders import Recording, load_eegbci

SUBJECTS = tuple(range(1, 21))
MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
BASELINE = ef.Window("baseline", -1.0, 0.0)
MOVEMENT = ef.Window("movement", 0.5, 3.5)
WHOLE = ef.Window("epoch", 0.0, 6.0)
HAND = {"hand": ["FC3", "C5", "C3", "C1", "CP3", "FC4", "C2", "C4", "C6", "CP4"]}

DATASET = "eegbci"


@pytest.fixture(scope="module")
def long_recordings() -> list[Recording]:
    return [load_eegbci(subject, tmax=6.0) for subject in SUBJECTS]


@pytest.fixture(scope="module")
def hand_signals(long_recordings: list[Recording]) -> dict[str, list[ef.BandSignal]]:
    return {
        band.name: [
            ef.BandSignal.from_epochs(r.epochs, band, recording=r.name, picks=HAND["hand"])
            for r in long_recordings
        ]
        for band in (MU, BETA)
    }


def _per_subject_difference(
    long_recordings: list[Recording],
    signals: list[ef.BandSignal],
    measure: object,
    window: ef.Window,
) -> np.ndarray:
    """Mean over movement trials minus mean over rest trials, per subject, in dB."""
    out = []
    for recording, signal in zip(long_recordings, signals, strict=True):
        moving = recording.metadata["moving"].to_numpy() == 1
        values = measure(  # type: ignore[operator]
            [signal],
            baseline=BASELINE,
            windows=[window],
            groups=HAND,
            include_global=False,
            normalize="db",
        ).values[:, 0]
        out.append(np.nanmean(values[moving]) - np.nanmean(values[~moving]))
    return np.array(out)


@pytest.mark.validates(
    "erd_magnitude",
    "erd_duration",
    "ers_magnitude",
    kind="physiology",
    claim="Movement deepens and lengthens desynchronization and shortens synchronization",
    criterion="ERD larger and ERS smaller during movement in 90 percent of subjects; p below 1e-5",
)
@pytest.mark.parametrize("band", [MU, BETA], ids=lambda band: band.name)
def test_movement_deepens_and_lengthens_desynchronization(
    long_recordings: list[Recording], hand_signals: dict[str, list[ef.BandSignal]], band: ef.Band
) -> None:
    signals = hand_signals[band.name]
    for measure in (ef.erd_magnitude, ef.erd_duration):
        difference = _per_subject_difference(long_recordings, signals, measure, MOVEMENT)
        assert np.mean(difference > 0.0) >= 0.9, (measure.__name__, difference.round(3))
        assert ttest_1samp(difference, 0.0, alternative="greater").pvalue < 1e-5
    difference = _per_subject_difference(long_recordings, signals, ef.ers_magnitude, MOVEMENT)
    assert np.mean(difference < 0.0) >= 0.9, difference.round(3)
    assert ttest_1samp(difference, 0.0, alternative="less").pvalue < 1e-5


@pytest.mark.validates(
    "erds_mean",
    "erd_magnitude",
    "erd_duration",
    "ers_magnitude",
    "ers_duration",
    "erds_slope",
    "erds_peak_latency",
    "erds_rebound_latency",
    kind="formula",
    claim="The ERDS measures are the documented trace's mean, balance, slope and argmaxes",
    criterion="relative error below 1e-9; durations tile the window",
)
def test_measures_are_tied_to_each_other_and_to_the_trace(
    hand_signals: dict[str, list[ef.BandSignal]],
) -> None:
    signal = hand_signals["beta"][0]
    kwargs = {
        "baseline": BASELINE,
        "windows": [MOVEMENT],
        "include_global": False,
        "normalize": "db",
    }
    mean = ef.erds_mean([signal], **kwargs).values
    erd_magnitude = ef.erd_magnitude([signal], **kwargs).values
    erd_duration = ef.erd_duration([signal], **kwargs).values
    ers_magnitude = ef.ers_magnitude([signal], **kwargs).values
    ers_duration = ef.ers_duration([signal], **kwargs).values
    slope = ef.erds_slope([signal], **kwargs).values
    peak = ef.erds_peak_latency([signal], **kwargs).values
    rebound = ef.erds_rebound_latency([signal], **kwargs).values

    # The trace itself, from the documented definition.
    power = signal.power
    in_baseline = (signal.times >= BASELINE.tmin) & (signal.times <= BASELINE.tmax)
    in_window = (signal.times >= MOVEMENT.tmin) & (signal.times <= MOVEMENT.tmax)
    reference = power[:, :, in_baseline].mean(axis=2, keepdims=True)
    trace = 10.0 * np.log10(power[:, :, in_window] / reference)
    times = signal.times[in_window]

    np.testing.assert_allclose(mean, trace.mean(axis=2), rtol=1e-9)
    # Every sample is either below or above zero, so the two durations tile the window.
    np.testing.assert_allclose(erd_duration + ers_duration, in_window.sum() / signal.sfreq)
    # The mean is the signed, duration-weighted balance of the two magnitudes.
    balance = (ers_magnitude * ers_duration - erd_magnitude * erd_duration) / (
        erd_duration + ers_duration
    )
    np.testing.assert_allclose(mean, balance, rtol=1e-9)
    fitted = np.array([[np.polyfit(times, cell, 1)[0] for cell in epoch] for epoch in trace])
    np.testing.assert_allclose(slope, fitted, rtol=1e-6)
    peak_index = np.argmax(np.abs(trace), axis=2)
    np.testing.assert_array_equal(peak, times[peak_index])
    # The rebound is the largest value strictly after the peak; NaN when the peak is last.
    after = np.arange(times.size)[None, None, :] > peak_index[..., None]
    expected_rebound = np.where(
        after.any(axis=2),
        times[np.argmax(np.where(after, trace, -np.inf), axis=2)],
        np.nan,
    )
    np.testing.assert_array_equal(rebound, expected_rebound)


@pytest.mark.validates(
    "erds_onset_latency",
    kind="behaviour",
    claim="The default six-cycle persistence keeps the onset's false-positive rate low",
    criterion="above 0.9 at a fixed 100 ms; below 0.15 at the default; lower still at 1500 ms",
)
def test_onset_default_persistence_holds_the_null_rate_down(
    long_recordings: list[Recording],
    hand_signals: dict[str, list[ef.BandSignal]],
    record: Callable[[str], None],
) -> None:
    """On rest trials an onset is a false positive.

    A fixed 100 ms persistence fires on nearly every rest trial in beta. The
    default, six cycles of the band's low edge, was set so that this rate lands
    near 5 percent; longer requirements lower it further.
    """

    def rest_rate(**persistence: float) -> float:
        fired = []
        for recording, signal in zip(long_recordings, hand_signals["beta"], strict=True):
            resting = recording.metadata["moving"].to_numpy() == 0
            onset = ef.erds_onset_latency(
                [signal],
                baseline=BASELINE,
                windows=[WHOLE],
                groups=HAND,
                include_global=False,
                **persistence,
            ).values[resting, 0]
            fired.append(np.isfinite(onset).mean())
        return float(np.mean(fired))

    fixed = rest_rate(min_duration_ms=100.0)
    default = rest_rate()
    longer = rest_rate(min_duration_ms=1500.0)
    record(
        f"beta rest trials with an onset: {fixed:.2f} at 100 ms, {default:.2f} at the six-cycle "
        f"default, {longer:.2f} at 1500 ms"
    )
    assert fixed > 0.9, fixed
    assert default < 0.15, default
    assert fixed > default > longer, (fixed, default, longer)
