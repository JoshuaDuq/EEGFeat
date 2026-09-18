import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.erds import erds
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window

ALPHA = Band("alpha", 8.0, 13.0)
SFREQ = 100.0
BASE = Window("base", -1.0, -1.0 / SFREQ)
STIM = Window("stim", 0.0, 1.0)


def _signal(envelope: np.ndarray) -> BandSignal:
    times = np.arange(envelope.shape[-1]) / SFREQ - 1.0
    return BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=times,
        ch_names=("C3",),
        band=ALPHA,
        sfreq=SFREQ,
    )


def _step(baseline_amp: float, active_amp: float, n: int = 201) -> BandSignal:
    envelope = np.full((1, 1, n), baseline_amp)
    envelope[:, :, n // 2 :] = active_amp
    return _signal(envelope)


def test_a_halved_power_gives_minus_fifty_percent() -> None:
    # amplitude 1 -> power 1 in baseline; amplitude sqrt(0.5) -> power 0.5 active
    table = erds(
        [_step(1.0, np.sqrt(0.5))],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
    )
    assert table.select(measure="mean").values.item() == pytest.approx(-50.0)


def test_the_same_signal_against_its_own_baseline_is_zero() -> None:
    table = erds([_step(1.0, 1.0)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="mean").values.item() == pytest.approx(0.0)


def test_db_and_percent_agree_on_a_doubling() -> None:
    signal = _step(1.0, np.sqrt(2.0))
    pct = (
        erds(
            [signal],
            baseline=BASE,
            windows=[STIM],
            include_global=False,
            normalize="percent",
        )
        .select(measure="mean")
        .values.item()
    )
    db = (
        erds(
            [signal],
            baseline=BASE,
            windows=[STIM],
            include_global=False,
            normalize="db",
        )
        .select(measure="mean")
        .values.item()
    )
    assert pct == pytest.approx(100.0)
    assert db == pytest.approx(10.0 * np.log10(2.0))


def test_a_wholly_negative_trace_has_full_erd_duration_and_no_ers() -> None:
    table = erds(
        [_step(1.0, np.sqrt(0.5))],
        baseline=BASE,
        windows=[STIM],
        include_global=False,
    )
    assert table.select(measure="erd_magnitude").values.item() == pytest.approx(50.0)
    assert table.select(measure="erd_duration").values.item() == pytest.approx(101 / SFREQ)
    assert table.select(measure="ers_magnitude").values.item() == 0.0
    assert table.select(measure="ers_duration").values.item() == 0.0


def test_slope_recovers_a_known_linear_ramp() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    # power ramps from 1.0 to 2.0 across the 1 s active window -> +100%/s in percent
    envelope[:, :, n // 2 :] = np.sqrt(np.linspace(1.0, 2.0, n - n // 2))
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="slope").values.item() == pytest.approx(100.0, rel=0.02)


def test_an_unusable_baseline_yields_nan_rather_than_an_enormous_ratio() -> None:
    envelope = np.zeros((1, 1, 201))
    envelope[:, :, 100:] = 1.0
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.select(measure="mean").values).all()


def test_erds_without_a_baseline_window_is_impossible_to_call() -> None:
    with pytest.raises(TypeError):
        erds([_step(1.0, 1.0)], windows=[STIM])  # type: ignore[call-arg]


def test_an_unknown_normalization_raises() -> None:
    with pytest.raises(ValueError, match="normalize"):
        erds([_step(1.0, 1.0)], baseline=BASE, windows=[STIM], normalize="log10")  # type: ignore[arg-type]


def test_peak_latency_finds_the_largest_excursion() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, 150] = np.sqrt(3.0)  # t = +0.5 s
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="peak_latency").values.item() == pytest.approx(0.5)


def test_onset_latency_is_the_first_crossing_of_the_baseline_variability() -> None:
    n = 201
    rng = np.random.RandomState(0)
    envelope = np.ones((1, 1, n))
    envelope[:, :, :100] += rng.normal(0.0, 0.01, (1, 1, 100))
    envelope[:, :, 130:] = np.sqrt(2.0)  # steps at t = +0.3 s
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="onset_latency").values.item() == pytest.approx(0.3, abs=0.02)


def test_a_trace_that_never_crosses_has_no_onset() -> None:
    table = erds([_step(1.0, 1.0)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.select(measure="onset_latency").values).all()


def test_rebound_is_the_largest_value_after_the_peak() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, 120] = np.sqrt(0.1)  # deep ERD at +0.2 s, the largest excursion
    envelope[:, :, 180] = np.sqrt(1.5)  # smaller ERS at +0.8 s
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="peak_latency").values.item() == pytest.approx(0.2)
    assert table.select(measure="rebound_latency").values.item() == pytest.approx(0.8)


def test_a_peak_at_the_window_end_leaves_no_rebound() -> None:
    n = 201
    envelope = np.ones((1, 1, n))
    envelope[:, :, -1] = np.sqrt(5.0)
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.select(measure="rebound_latency").values).all()


def test_a_near_dead_channel_is_withheld_rather_than_amplified() -> None:
    # A 0.1 uV baseline envelope is power 1e-14, below the reference's 1e-12 guard.
    # Without that guard this returns an ERDS of order 1e6 percent.
    n = 201
    envelope = np.full((1, 1, n), 1e-7)
    envelope[:, :, n // 2 :] = 1e-5
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert np.isnan(table.select(measure="mean").values).all()


def test_a_healthy_channel_is_not_withheld_by_the_guard() -> None:
    # 10 uV envelope is power 1e-10, comfortably above the guard.
    n = 201
    envelope = np.full((1, 1, n), 1e-5)
    envelope[:, :, n // 2 :] = 1e-5 * np.sqrt(0.5)
    table = erds([_signal(envelope)], baseline=BASE, windows=[STIM], include_global=False)
    assert table.select(measure="mean").values.item() == pytest.approx(-50.0)
