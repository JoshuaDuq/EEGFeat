from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.signal import Signal
from eegfeat.spectra import Window

SFREQ = 200.0
ALPHA = Band("alpha", 8.0, 13.0)
THRESHOLDS = {
    "amp_fraction_threshold": 0.0,
    "amp_consistency_threshold": 0.5,
    "period_consistency_threshold": 0.5,
    "monotonicity_threshold": 0.8,
    "min_n_cycles": 3,
}


def _signal():
    times = np.arange(2000) / SFREQ
    return Signal.from_arrays(
        data=np.sin(2 * np.pi * 10 * times).reshape(1, 1, -1),
        times=times,
        ch_names=("C3",),
        sfreq=SFREQ,
        row_ids=(("cycles", 0, "event"),),
    )


def _compute(signal, **parameters):
    assert find_spec("eegfeat.cycles") is not None, "cycle analysis is not implemented"
    from eegfeat.cycles import cycle_features

    return cycle_features(
        [signal],
        bands=(ALPHA,),
        include_global=False,
        windows=parameters.pop("windows", (Window("all", -np.inf, np.inf),)),
        **parameters,
    )


def test_sine_cycles_have_expected_amplitude_period_and_symmetry():
    table = _compute(_signal())
    for name, expected in [
        ("cycle_period", 0.1),
        ("cycle_rise_time", 0.05),
        ("cycle_decay_time", 0.05),
        ("cycle_rise_decay_symmetry", 0.5),
        ("cycle_peak_trough_symmetry", 0.5),
        ("cycle_amplitude", 2.0),
    ]:
        assert table.select(measure=name).values.item() == pytest.approx(expected, abs=0.015)
    assert table.select(measure="cycle_burst_fraction").values.item() > 0.9
    assert table.row_ids == (("cycles", 0, "event"),)
    assert all(meta.band == ALPHA for meta in table.meta)
    assert table.coverage.min() == 1.0


def test_cycles_match_bycycle_and_only_use_complete_windowed_cycles():
    assert find_spec("eegfeat.cycles") is not None, "cycle analysis is not implemented"
    from bycycle.features import compute_features

    signal = _signal()
    window = Window("middle", 2.03, 5.97)
    frames = compute_features(signal.data[0, 0], SFREQ, (8, 13), threshold_kwargs=THRESHOLDS)
    selected = frames[
        (signal.times[frames.sample_last_trough] >= window.tmin)
        & (signal.times[frames.sample_next_trough] <= window.tmax)
    ]
    bursting = selected[selected.is_burst]
    table = _compute(signal, windows=(window,))
    assert table.select(measure="cycle_period").values.item() == pytest.approx(
        bursting.period.mean() / SFREQ
    )
    assert table.select(measure="cycle_amplitude").values.item() == pytest.approx(
        bursting.volt_amp.mean()
    )
    assert table.select(measure="cycle_burst_fraction").values.item() == pytest.approx(
        selected.is_burst.mean()
    )
    assert table.select(measure="cycle_count").values.item() == len(selected)


def test_no_burst_is_flagged_and_waveform_summaries_remain_undefined():
    table = _compute(_signal(), burst_thresholds={"min_n_cycles": 1000})
    assert table.select(measure="cycle_burst_fraction").values.item() == 0
    shape = table.select(measure="cycle_amplitude")
    assert np.isnan(shape.values).all()
    assert shape.flags["cycle_no_burst"].all()


def test_window_without_complete_cycles_is_explicit():
    table = _compute(_signal(), windows=(Window("tiny", 2.0, 2.005),))
    assert table.select(measure="cycle_count").values.item() == 0
    assert np.isnan(table.select(measure="cycle_burst_fraction").values).all()
    assert table.flags["cycle_no_complete_cycles"].all()


@pytest.mark.parametrize("kind", ["nan", "passband", "constant", "short"])
def test_cycles_refuse_unsupported_inputs(kind):
    signal = _signal()
    if kind == "nan":
        signal.data[0, 0, 400] = np.nan
    if kind == "passband":
        signal = replace(signal, passband=(10, 20))
    if kind == "constant":
        signal.data[:] = 1
    if kind == "short":
        signal = Signal.from_arrays(
            data=signal.data[..., :40],
            times=signal.times[:40],
            sfreq=SFREQ,
            ch_names=signal.ch_names,
            row_ids=signal.row_ids,
        )
    with pytest.raises(ValueError):
        _compute(signal)


@pytest.mark.parametrize(
    "thresholds", [{"monotonicity_threshold": -0.1}, {"min_n_cycles": 0}, {"unknown": 0.5}]
)
def test_cycles_refuse_invalid_burst_thresholds(thresholds):
    with pytest.raises(ValueError):
        _compute(_signal(), burst_thresholds=thresholds)


def test_overlapping_windows_keep_their_own_cycle_flags():
    table = _compute(
        _signal(), windows=(Window("all", -np.inf, np.inf), Window("tiny", 2.0, 2.005))
    )
    assert not table.select(window="all").flags["cycle_no_complete_cycles"].any()
    assert table.select(window="tiny").flags["cycle_no_complete_cycles"].all()
