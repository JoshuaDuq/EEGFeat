import numpy as np
import pytest

import eegfeat as ef
from eegfeat.bands import Band
from eegfeat.phase import itpc, pac
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

SFREQ = 100.0
ALPHA, GAMMA = Band("alpha", 8.0, 13.0), Band("gamma", 30.0, 45.0)
WINDOW = Window("all", 0.0, 2.0)


def _from_phase(
    phase: np.ndarray, amplitude: float | np.ndarray = 1.0, band: Band = ALPHA
) -> BandSignal:
    analytic = amplitude * np.exp(1j * phase)
    return BandSignal.from_arrays(
        analytic=analytic,
        times=np.arange(phase.shape[-1]) / SFREQ,
        ch_names=("C3",),
        band=band,
        sfreq=SFREQ,
    )


def test_identical_phase_across_trials_gives_coherence_one() -> None:
    n_epochs, n_times = 20, 201
    ramp = np.linspace(0.0, 8 * np.pi, n_times)
    phase = np.tile(ramp, (n_epochs, 1, 1)).reshape(n_epochs, 1, n_times)
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(1.0)


def test_trials_are_averaged_before_time_not_after() -> None:
    # Phase sweeps over time but is identical across trials. Averaging trials
    # first gives 1.0; averaging time first would cancel the sweep to near zero.
    n_epochs, n_times = 20, 201
    ramp = np.linspace(0.0, 8 * np.pi, n_times)
    phase = np.tile(ramp, (n_epochs, 1, 1)).reshape(n_epochs, 1, n_times)
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    time_first = np.abs(np.mean(np.exp(1j * ramp)))
    assert table.values.item() == pytest.approx(1.0)
    assert time_first < 0.1


def test_uniformly_random_phase_gives_low_coherence() -> None:
    rng = np.random.RandomState(0)
    phase = rng.uniform(-np.pi, np.pi, (200, 1, 201))
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    # The expected value under the null is about 1/sqrt(N).
    assert table.values.item() < 3.0 / np.sqrt(200)


def test_without_trial_labels_there_is_one_row_named_all() -> None:
    rng = np.random.RandomState(1)
    table = itpc(
        [_from_phase(rng.uniform(-np.pi, np.pi, (8, 1, 201)))],
        windows=[WINDOW],
        include_global=False,
    )
    assert table.row_labels == ("all",)
    assert table.values.shape[0] == 1


def test_trial_labels_give_one_row_per_group_in_sorted_order() -> None:
    n_times = 201
    locked = np.tile(np.linspace(0, 8 * np.pi, n_times), (10, 1, 1)).reshape(10, 1, n_times)
    rng = np.random.RandomState(2)
    scattered = rng.uniform(-np.pi, np.pi, (10, 1, n_times))
    phase = np.concatenate([locked, scattered], axis=0)
    labels = ["locked"] * 10 + ["scattered"] * 10
    table = itpc([_from_phase(phase)], windows=[WINDOW], trials=labels, include_global=False)
    assert table.row_labels == ("locked", "scattered")
    values = dict(zip(table.row_labels, table.values[:, 0], strict=True))
    assert values["locked"] == pytest.approx(1.0)
    assert values["scattered"] < 0.5


def test_a_cross_trial_table_cannot_be_joined_to_a_per_epoch_one() -> None:
    rng = np.random.RandomState(3)
    signal = _from_phase(rng.uniform(-np.pi, np.pi, (4, 1, 201)))
    coherence = itpc([signal], windows=[WINDOW], include_global=False)
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    with pytest.raises(ValueError, match="row semantics"):
        ef.concat([coherence, per_epoch])


def test_group_rows_and_epoch_rows_differ_even_at_the_same_count() -> None:
    # One trial group and one epoch happen to give one row each; they still must
    # not be joined, because the rows mean different things.
    rng = np.random.RandomState(9)
    signal = _from_phase(rng.uniform(-np.pi, np.pi, (1, 1, 201)))
    coherence = itpc([signal], windows=[WINDOW], include_global=False)
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    assert coherence.n_rows == per_epoch.n_rows == 1
    with pytest.raises(ValueError, match="row semantics"):
        ef.concat([coherence, per_epoch])


def test_mismatched_trial_labels_raise() -> None:
    rng = np.random.RandomState(4)
    signal = _from_phase(rng.uniform(-np.pi, np.pi, (4, 1, 201)))
    with pytest.raises(ValueError, match="one label per epoch"):
        itpc([signal], windows=[WINDOW], trials=["a", "b"])


def test_coherence_is_bounded() -> None:
    rng = np.random.RandomState(5)
    table = itpc(
        [_from_phase(rng.uniform(-np.pi, np.pi, (30, 1, 201)))],
        windows=[WINDOW],
        include_global=False,
    )
    assert 0.0 <= table.values.item() <= 1.0


# --- phase-amplitude coupling ---------------------------------------------------------


def _coupled(strength: float, n_epochs: int = 6, n_times: int = 401) -> tuple:
    times = np.arange(n_times) / SFREQ
    slow_phase = np.tile(2 * np.pi * 5.0 * times, (n_epochs, 1, 1)).reshape(n_epochs, 1, n_times)
    envelope = 1.0 + strength * np.cos(slow_phase)
    slow = _from_phase(slow_phase, band=ALPHA)
    fast = BandSignal.from_arrays(
        analytic=envelope * np.exp(1j * 2 * np.pi * 40.0 * times),
        times=times,
        ch_names=("C3",),
        band=GAMMA,
        sfreq=SFREQ,
    )
    return slow, fast


def test_coupling_is_higher_when_amplitude_tracks_phase() -> None:
    weak = pac(*_coupled(0.0), windows=[WINDOW], include_global=False)
    strong = pac(*_coupled(0.8), windows=[WINDOW], include_global=False)
    assert strong.values.mean() > weak.values.mean()
    assert weak.values.mean() < 0.05


def test_normalized_coupling_is_invariant_to_overall_amplitude() -> None:
    slow, fast = _coupled(0.8)
    louder = BandSignal.from_arrays(
        analytic=fast.analytic * 1000.0,
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
    )
    a = pac(slow, fast, windows=[WINDOW], include_global=False)
    b = pac(slow, louder, windows=[WINDOW], include_global=False)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)


def test_unnormalized_coupling_scales_with_amplitude() -> None:
    slow, fast = _coupled(0.8)
    louder = BandSignal.from_arrays(
        analytic=fast.analytic * 10.0,
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
    )
    a = pac(slow, fast, windows=[WINDOW], normalize=False, include_global=False)
    b = pac(slow, louder, windows=[WINDOW], normalize=False, include_global=False)
    np.testing.assert_allclose(b.values, a.values * 10.0, rtol=1e-10)


def test_coupling_has_one_row_per_epoch() -> None:
    slow, fast = _coupled(0.5, n_epochs=6)
    table = pac(slow, fast, windows=[WINDOW], include_global=False)
    assert table.values.shape[0] == 6
    assert table.row_labels is None


def test_the_amplitude_band_is_recorded_on_the_column() -> None:
    slow, fast = _coupled(0.5)
    table = pac(slow, fast, windows=[WINDOW], include_global=False)
    assert table.meta[0].band is GAMMA


def test_a_phase_band_faster_than_the_amplitude_band_raises() -> None:
    slow, fast = _coupled(0.5)
    with pytest.raises(ValueError, match="must be slower"):
        pac(fast, slow, windows=[WINDOW], include_global=False)


def test_mismatched_channels_or_times_raise() -> None:
    slow, fast = _coupled(0.5)
    renamed = BandSignal.from_arrays(
        analytic=fast.analytic,
        times=fast.times,
        ch_names=("Cz",),
        band=fast.band,
        sfreq=fast.sfreq,
    )
    with pytest.raises(ValueError, match="same channels"):
        pac(slow, renamed, windows=[WINDOW])
    shifted = BandSignal.from_arrays(
        analytic=fast.analytic,
        times=fast.times + 9.0,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
    )
    with pytest.raises(ValueError, match="same time axis"):
        pac(slow, shifted, windows=[WINDOW])


def test_a_silent_amplitude_band_yields_nan_rather_than_zero() -> None:
    slow, fast = _coupled(0.0)
    silent = BandSignal.from_arrays(
        analytic=np.zeros_like(fast.analytic),
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
    )
    table = pac(slow, silent, windows=[WINDOW], include_global=False)
    assert np.isnan(table.values).all()


def test_the_result_is_a_feature_table() -> None:
    slow, fast = _coupled(0.5)
    assert isinstance(pac(slow, fast, windows=[WINDOW]), FeatureTable)
