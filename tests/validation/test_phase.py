"""Phase measures: inter-trial coherence on SSVEP, coupling on Sleep-EDF.

A flicker locked to the trigger drives a response whose phase repeats from trial
to trial, so phase locking across trials at the flicker frequency is high for the
driven condition and near its chance level for the other. Slow-oscillation to
spindle coupling is the textbook phase-amplitude effect in sleep; on these two
derivations the raw mean vector length did not resolve it against a surrogate,
which is recorded here as a limit rather than asserted away.
"""

from __future__ import annotations

import numpy as np
import pytest

import eegfeat as ef
from validation.loaders import Recording

OCCIPITAL = {"occipital": ["O1", "Oz", "O2"]}
STIMULATION = ef.Window("stimulation", 1.0, 19.0)
WHOLE_EPOCH = ef.Window("epoch", 0.0, 30.0)


@pytest.fixture(scope="module")
def condition(ssvep_recording: Recording) -> np.ndarray:
    return ssvep_recording.metadata["condition"].to_numpy()


def _flicker_band(ssvep_recording: Recording, frequency: float) -> ef.BandSignal:
    band = ef.Band(f"f{int(frequency)}", frequency - 1.0, frequency + 1.0)
    return ef.BandSignal.from_epochs(ssvep_recording.epochs, band, recording=ssvep_recording.name)


@pytest.mark.parametrize("frequency", [12.0, 15.0])
def test_phase_locks_across_trials_only_when_driven(
    ssvep_recording: Recording, condition: np.ndarray, frequency: float
) -> None:
    signal = _flicker_band(ssvep_recording, frequency)
    driven = f"{int(frequency)}hz"
    for measure in (ef.itpc, ef.ppc):
        table = measure(
            [signal],
            windows=[STIMULATION],
            trials=condition,
            groups=OCCIPITAL,
            include_global=False,
        )
        by_condition = dict(zip(table.row_labels, table.values[:, 0], strict=True))
        other = next(label for label in by_condition if label != driven)
        assert by_condition[driven] > by_condition[other] + 0.1, (measure.__name__, by_condition)


def test_undriven_ppc_is_near_zero(ssvep_recording: Recording, condition: np.ndarray) -> None:
    """PPC has no small-sample bias, so trials without a locked response sit near zero."""
    signal = _flicker_band(ssvep_recording, 12.0)
    table = ef.ppc(
        [signal], windows=[STIMULATION], trials=condition, groups=OCCIPITAL, include_global=False
    )
    by_condition = dict(zip(table.row_labels, table.values[:, 0], strict=True))
    assert abs(by_condition["15hz"]) < 0.1, by_condition


def test_itpc_is_the_documented_formula(ssvep_recording: Recording, condition: np.ndarray) -> None:
    signal = _flicker_band(ssvep_recording, 12.0)
    table = ef.itpc([signal], windows=[STIMULATION], trials=condition, include_global=False)

    inside = (signal.times >= STIMULATION.tmin) & (signal.times <= STIMULATION.tmax)
    unit = np.exp(1j * np.angle(signal.analytic[:, :, inside]))
    for row, label in enumerate(table.row_labels):
        # Trials first, then time.
        expected = np.abs(unit[condition == label].mean(axis=0)).mean(axis=1)
        np.testing.assert_allclose(table.values[row], expected, rtol=1e-9)


def test_zero_crossings_follow_rices_formula_on_a_narrow_band(
    ssvep_recording: Recording,
) -> None:
    """For a band-limited signal the crossing rate is twice the Hjorth mobility."""
    signal = _flicker_band(ssvep_recording, 12.0)
    real = ef.Signal.from_arrays(
        data=np.real(signal.analytic),
        times=signal.times,
        ch_names=signal.ch_names,
        sfreq=signal.sfreq,
        row_ids=signal.row_ids,
    )
    crossings = ef.zero_crossing_rate([real], windows=[STIMULATION], include_global=False).values
    mobility = ef.hjorth_mobility([real], windows=[STIMULATION], include_global=False).values
    assert np.all((crossings > 20.0) & (crossings < 28.0))
    # "Within a few percent", as documented: the worst trial-channel cell sits near 5%.
    np.testing.assert_allclose(crossings, 2.0 * mobility, rtol=0.06)
    np.testing.assert_allclose(crossings.mean(), 2.0 * mobility.mean(), rtol=0.01)


def test_pac_is_the_documented_formula(sleep_recordings: list[Recording]) -> None:
    recording = sleep_recordings[0]
    slow = ef.BandSignal.from_epochs(recording.epochs, ef.Band("so", 0.5, 1.5), recording="x")
    fast = ef.BandSignal.from_epochs(recording.epochs, ef.Band("sigma", 12.0, 15.0), recording="x")
    table = ef.pac(slow, fast, windows=[WHOLE_EPOCH], include_global=False)

    amplitude = np.abs(fast.analytic)
    weighted = np.abs((amplitude * np.exp(1j * np.angle(slow.analytic))).sum(axis=2))
    np.testing.assert_allclose(table.values, weighted / amplitude.sum(axis=2), rtol=1e-9)


def test_raw_pac_needs_a_null(sleep_recordings: list[Recording]) -> None:
    """Even with the coupling destroyed, the mean vector length stays well above zero.

    Shifting the fast envelope by five seconds breaks any phase-amplitude relation
    while keeping both signals' autocorrelation. The surrogate value is what a raw
    reading looks like with nothing behind it, which is why the docstring says to
    compare against a null rather than read the number absolutely.
    """
    recording = sleep_recordings[0]
    slow = ef.BandSignal.from_epochs(recording.epochs, ef.Band("so", 0.5, 1.5), recording="x")
    fast = ef.BandSignal.from_epochs(recording.epochs, ef.Band("sigma", 12.0, 15.0), recording="x")
    shift = int(5.0 * fast.sfreq)
    surrogate = ef.BandSignal.from_arrays(
        analytic=np.roll(fast.analytic, shift, axis=2),
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids,
    )
    null = ef.pac(slow, surrogate, windows=[WHOLE_EPOCH], include_global=False).values
    assert np.all(np.isfinite(null)) and np.all((null >= 0.0) & (null <= 1.0))
    assert np.median(null) > 0.03, np.median(null)
