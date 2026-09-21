"""Connectivity on resting motor-task epochs and on SSVEP.

Sensor-level envelope correlation is dominated by volume conduction: neighbouring
electrodes correlate far more than distant ones, and orthogonalizing removes most
of it. wPLI is delegated to mne-connectivity and must reproduce a direct call.
Common visual drive raises coherence among occipital electrodes at the flicker
frequency in the driven condition.
"""

from __future__ import annotations

import mne_connectivity
import numpy as np
import pytest

import eegfeat as ef
from validation.loaders import Recording

ALPHA = ef.Band("alpha", 8.0, 13.0)
WINDOW = ef.Window("window", 0.5, 3.5)
NEIGHBOURS = [("C3", "C1"), ("C4", "C2"), ("O1", "Oz"), ("O2", "Oz"), ("Cz", "C1"), ("F3", "F1")]
DISTANT = [("Fp1", "O2"), ("Fp2", "O1"), ("F7", "P8"), ("F8", "P7"), ("Fpz", "Oz"), ("AF3", "PO8")]
OCCIPITAL = ["O1", "Oz", "O2", "PO9", "PO10", "P7", "P8"]
STIMULATION = ef.Window("stimulation", 1.0, 19.0)


@pytest.fixture(scope="module")
def resting(eegbci_recordings: list[Recording]) -> Recording:
    recording = eegbci_recordings[0]
    keep = recording.metadata["moving"].to_numpy() == 0
    return Recording(recording.name, recording.epochs[keep])


@pytest.fixture(scope="module")
def alpha(resting: Recording) -> ef.BandSignal:
    return ef.BandSignal.from_epochs(resting.epochs, ALPHA, recording=resting.name)


def _pairs(table: ef.FeatureTable) -> dict[tuple[str, str], float]:
    values = {}
    for meta, value in zip(table.meta, table.values[0], strict=True):
        assert meta.nodes is not None
        values[meta.nodes] = float(value)
        values[meta.nodes[::-1]] = float(value)
    return values


def test_raw_envelope_correlation_is_the_fisher_mean_of_trial_correlations(
    alpha: ef.BandSignal,
) -> None:
    table = ef.envelope_correlation([alpha], windows=[WINDOW], orthogonalize=None)
    inside = (alpha.times >= WINDOW.tmin) & (alpha.times <= WINDOW.tmax)
    envelope = alpha.envelope[:, :, inside]
    for meta, value in list(zip(table.meta, table.values[0], strict=True))[:50]:
        assert meta.nodes is not None
        i, j = (alpha.ch_names.index(name) for name in meta.nodes)
        per_trial = [np.corrcoef(trial[i], trial[j])[0, 1] for trial in envelope]
        assert value == pytest.approx(np.tanh(np.mean(np.arctanh(per_trial))), rel=1e-9)


def test_volume_conduction_shows_and_orthogonalization_removes_it(
    alpha: ef.BandSignal,
) -> None:
    raw = _pairs(ef.envelope_correlation([alpha], windows=[WINDOW], orthogonalize=None))
    orthogonalized = _pairs(ef.envelope_correlation([alpha], windows=[WINDOW]))

    near_raw = np.mean([raw[pair] for pair in NEIGHBOURS])
    far_raw = np.mean([raw[pair] for pair in DISTANT])
    assert near_raw > far_raw + 0.2, (near_raw, far_raw)

    near_orth = np.mean([orthogonalized[pair] for pair in NEIGHBOURS])
    far_orth = np.mean([orthogonalized[pair] for pair in DISTANT])
    assert near_orth < near_raw / 2.0, (near_orth, near_raw)
    assert abs(near_orth - far_orth) < 0.1, (near_orth, far_orth)


def test_graph_summaries_are_well_formed(alpha: ef.BandSignal) -> None:
    pairs = ef.envelope_correlation([alpha], windows=[WINDOW], orthogonalize=None)
    efficiency = ef.global_efficiency(pairs).values
    clustering = ef.clustering_coefficient(pairs, threshold=0.5).values
    assert efficiency.shape == clustering.shape == (1, 1)
    assert 0.0 < efficiency[0, 0] <= 1.0
    assert 0.0 <= clustering[0, 0] <= 1.0


def test_wpli_reproduces_a_direct_mne_connectivity_call(resting: Recording) -> None:
    signal = ef.Signal.from_epochs(resting.epochs, recording=resting.name)
    table = ef.wpli(signal, bands=[ALPHA], windows=[WINDOW])

    inside = (signal.times >= WINDOW.tmin) & (signal.times <= WINDOW.tmax)
    direct = mne_connectivity.spectral_connectivity_epochs(
        signal.data[:, :, inside],
        method="wpli",
        mode="multitaper",
        sfreq=signal.sfreq,
        fmin=1.0,
        fmax=40.0,
        mt_bandwidth=2.0,
        verbose=False,
    )
    dense = direct.get_data(output="dense")
    in_band = ALPHA.mask(np.asarray(direct.freqs))
    for meta, value in zip(table.meta, table.values[0], strict=True):
        assert meta.nodes is not None
        i, j = sorted(signal.ch_names.index(name) for name in meta.nodes)
        assert value == pytest.approx(dense[j, i, in_band].mean(), rel=1e-9)


METHODS = ("coh", "imcoh", "plv", "ciplv", "ppc", "pli", "wpli", "wpli2_debiased")


@pytest.fixture(scope="module")
def six_channels(resting: Recording) -> ef.Signal:
    return ef.Signal.from_epochs(
        resting.epochs, recording=resting.name, picks=["C3", "C4", "Cz", "O1", "O2", "Fz"]
    )


@pytest.mark.parametrize("method", METHODS)
def test_every_method_reproduces_mne_connectivity(six_channels: ef.Signal, method: str) -> None:
    table = ef.spectral_connectivity(
        six_channels, method=method, bands=[ALPHA], windows=[WINDOW]  # type: ignore[arg-type]
    )
    inside = (six_channels.times >= WINDOW.tmin) & (six_channels.times <= WINDOW.tmax)
    direct = mne_connectivity.spectral_connectivity_epochs(
        six_channels.data[:, :, inside],
        method=method,
        mode="multitaper",
        sfreq=six_channels.sfreq,
        fmin=1.0,
        fmax=40.0,
        mt_bandwidth=2.0,
        verbose=False,
    )
    dense = direct.get_data(output="dense")
    in_band = ALPHA.mask(np.asarray(direct.freqs))
    for meta, value in zip(table.meta, table.values[0], strict=True):
        assert meta.nodes is not None
        i, j = sorted(six_channels.ch_names.index(name) for name in meta.nodes)
        reference = dense[j, i, in_band]
        # Imaginary coherency is antisymmetric, so the unordered pair carries its magnitude.
        expected = np.abs(reference).mean() if method == "imcoh" else reference.mean()
        assert value == pytest.approx(expected, rel=1e-9)


def test_debiased_wpli_does_not_grow_as_trials_fall(six_channels: ef.Signal) -> None:
    """Plain wPLI is biased upward at low trial counts; the squared debiased form is not."""
    rng = np.random.default_rng(0)
    wpli, debiased = {}, {}
    for count in (42, 10):
        rows = np.sort(rng.choice(six_channels.n_epochs, count, replace=False))
        subset = ef.Signal.from_arrays(
            data=six_channels.data[rows],
            times=six_channels.times,
            ch_names=six_channels.ch_names,
            sfreq=six_channels.sfreq,
            row_ids=tuple(six_channels.row_ids[row] for row in rows),
        )
        wpli[count] = ef.wpli(subset, bands=[ALPHA], windows=[WINDOW]).values.mean()
        debiased[count] = ef.spectral_connectivity(
            subset, method="wpli2_debiased", bands=[ALPHA], windows=[WINDOW]
        ).values.mean()
    assert wpli[10] > wpli[42] + 0.1, wpli
    assert abs(debiased[10] - debiased[42]) < 0.05, debiased


def test_coherence_rises_at_the_flicker_frequency(ssvep_recording: Recording) -> None:
    signal = ef.Signal.from_epochs(
        ssvep_recording.epochs, recording=ssvep_recording.name, picks=OCCIPITAL
    )
    bands = [ef.Band("f12", 11.5, 12.5), ef.Band("f15", 14.5, 15.5)]
    table = ef.spectral_connectivity(
        signal,
        method="coh",
        bands=bands,
        windows=[STIMULATION],
        trials=ssvep_recording.metadata["condition"].to_numpy(),
        bandwidth=1.0,
    )
    rows = {label: row for row, label in enumerate(table.row_labels)}
    for band, driven in zip(bands, ("12hz", "15hz"), strict=True):
        mean_over_pairs = table.select(band=band).values.mean(axis=1)
        other = "15hz" if driven == "12hz" else "12hz"
        assert mean_over_pairs[rows[driven]] > mean_over_pairs[rows[other]] + 0.03, band.name
