import numpy as np
import pytest

from eegfeat.microstates import (
    _coverage,
    _duration,
    _occurrence,
    _smooth,
    _transitions,
    microstate_coverage,
    microstate_duration,
    microstate_occurrence,
    microstate_transitions,
    segment,
)
from eegfeat.signal import Signal
from eegfeat.spectra import Window

# segment() clusters with scikit-learn, the optional "microstates" extra; the
# metric definitions below do not need it.
requires_sklearn = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("sklearn") is None,
    reason="scikit-learn is not installed in this environment",
)

SFREQ = 100.0
WINDOW = Window("all", 0.0, 3.99)
N_CHANNELS = 8


def _planted(n_epochs: int = 6, n_times: int = 400, noise: float = 0.05) -> tuple:
    """Blocks of four fixed topographies, each held for 25 samples."""
    rng = np.random.RandomState(0)
    maps = rng.randn(4, N_CHANNELS)
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    data = np.zeros((n_epochs, N_CHANNELS, n_times))
    truth = np.zeros((n_epochs, n_times), dtype=int)
    shape = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, 25))
    for epoch in range(n_epochs):
        for block in range(n_times // 25):
            state = (block + epoch) % 4
            data[epoch, :, block * 25 : (block + 1) * 25] = maps[state][:, None] * shape[None, :]
            truth[epoch, block * 25 : (block + 1) * 25] = state
    data += rng.randn(n_epochs, N_CHANNELS, n_times) * noise
    signal = Signal.from_arrays(
        data=data,
        times=np.arange(n_times) / SFREQ,
        ch_names=tuple(f"E{i}" for i in range(N_CHANNELS)),
        sfreq=SFREQ,
    )
    return signal, truth


@requires_sklearn
def test_planted_topographies_are_recovered() -> None:
    signal, truth = _planted()
    seg = segment(signal, n_states=4)
    purity = np.mean(
        [
            np.bincount(seg.states[truth == k], minlength=4).max() / (truth == k).sum()
            for k in range(4)
        ]
    )
    assert purity == pytest.approx(1.0)


@requires_sklearn
def test_four_states_get_canonical_labels_and_others_do_not() -> None:
    signal, _ = _planted()
    assert segment(signal, n_states=4).labels == ("a", "b", "c", "d")
    assert segment(signal, n_states=3).labels == ("state1", "state2", "state3")


@requires_sklearn
def test_a_topography_and_its_inversion_are_the_same_state() -> None:
    signal, _ = _planted(noise=0.0)
    seg = segment(signal, n_states=4)
    flipped = Signal.from_arrays(
        data=-signal.data, times=signal.times, ch_names=signal.ch_names, sfreq=signal.sfreq
    )
    from eegfeat.microstates import _assign

    original = _assign(signal.data[0], seg.templates)
    inverted = _assign(flipped.data[0], seg.templates)
    np.testing.assert_array_equal(original, inverted)


@requires_sklearn
def test_coverage_sums_to_one_across_states() -> None:
    signal, _ = _planted()
    table = microstate_coverage(segment(signal, n_states=4), windows=[WINDOW])
    np.testing.assert_allclose(table.values.sum(axis=1), 1.0)


@requires_sklearn
def test_measures_have_one_row_per_epoch() -> None:
    signal, _ = _planted(n_epochs=6)
    seg = segment(signal, n_states=4)
    for fn in (microstate_coverage, microstate_duration, microstate_occurrence):
        table = fn(seg, windows=[WINDOW])
        assert table.values.shape == (6, 4)
        assert table.row_labels is None
        assert all(m.space_kind == "state" for m in table.meta)


@requires_sklearn
def test_duration_recovers_the_planted_block_length() -> None:
    signal, _ = _planted(noise=0.0)
    table = microstate_duration(segment(signal, n_states=4), windows=[WINDOW])
    # Blocks are 25 samples at 100 Hz = 250 ms.
    assert np.nanmean(table.values) == pytest.approx(250.0, rel=0.1)


@requires_sklearn
def test_transitions_are_ordered_pairs_excluding_self() -> None:
    signal, _ = _planted()
    table = microstate_transitions(segment(signal, n_states=4), windows=[WINDOW])
    spaces = [m.space for m in table.meta]
    assert len(spaces) == 12  # 4 states, ordered, no self-transitions
    assert "a-to-a" not in spaces
    assert all(m.space_kind == "pair" for m in table.meta)


@requires_sklearn
def test_fit_on_restricts_which_trials_inform_the_templates() -> None:
    signal, _ = _planted(n_epochs=6)
    mask = np.array([True, True, True, False, False, False])
    restricted = segment(signal, n_states=4, fit_on=mask)
    # Every epoch is still segmented, including those excluded from fitting.
    assert restricted.states.shape[0] == 6


@requires_sklearn
def test_a_fit_mask_of_the_wrong_length_raises() -> None:
    signal, _ = _planted(n_epochs=6)
    with pytest.raises(ValueError, match="one entry per epoch"):
        segment(signal, n_states=4, fit_on=np.array([True, False]))


@requires_sklearn
def test_excluding_every_epoch_raises() -> None:
    signal, _ = _planted(n_epochs=4)
    with pytest.raises(ValueError, match="nothing to cluster"):
        segment(signal, n_states=4, fit_on=np.zeros(4, dtype=bool))


@requires_sklearn
@pytest.mark.parametrize("n_states", [1, 13])
def test_an_unsupported_state_count_raises(n_states: int) -> None:
    signal, _ = _planted()
    with pytest.raises(ValueError, match="n_states"):
        segment(signal, n_states=n_states)


def test_segment_reports_its_missing_dependency_clearly() -> None:
    import importlib.util

    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("scikit-learn is installed here")
    signal, _ = _planted()
    with pytest.raises(ImportError, match=r"eegfeat\[microstates\]"):
        segment(signal)


# --- metric definitions ---------------------------------------------------------------


def test_smoothing_absorbs_a_short_run_into_the_longer_neighbour() -> None:
    # The single sample of state 1 goes to state 0, whose run is longer.
    states = np.array([0, 0, 0, 0, 1, 2, 2])
    np.testing.assert_array_equal(_smooth(states, 2), [0, 0, 0, 0, 0, 2, 2])


def test_a_tie_splits_the_run_and_an_odd_sample_goes_to_the_later_state() -> None:
    # Neighbours are equally long, so the run is halved. A one-sample run has an
    # empty first half, so it lands entirely on the following state.
    np.testing.assert_array_equal(_smooth(np.array([0, 0, 1, 2, 2]), 2), [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(
        _smooth(np.array([0, 0, 0, 1, 1, 2, 2, 2]), 3), [0, 0, 0, 0, 2, 2, 2, 2]
    )


def test_a_short_run_at_the_edge_takes_its_only_neighbour() -> None:
    np.testing.assert_array_equal(_smooth(np.array([1, 0, 0, 0, 0]), 2), [0, 0, 0, 0, 0])


def test_duration_is_nan_for_an_absent_state_but_occurrence_is_zero() -> None:
    states = np.array([0, 0, 1, 1])
    assert np.isnan(_duration(states, 3, SFREQ)[2])
    assert _occurrence(states, 3, SFREQ)[2] == 0.0
    assert _coverage(states, 3, SFREQ)[2] == 0.0


def test_transitions_count_segments_not_samples() -> None:
    # Two long runs: one transition, not one per sample pair.
    states = np.array([0, 0, 0, 1, 1, 1])
    matrix = _transitions(states, 2)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert np.isnan(matrix[1, 0])  # state 1 is never left


def test_transition_rows_sum_to_one_where_defined() -> None:
    rng = np.random.RandomState(1)
    matrix = _transitions(rng.randint(0, 3, size=200), 3)
    rows = np.nansum(matrix, axis=1)
    assert np.allclose(rows[np.isfinite(matrix).any(axis=1)], 1.0)
