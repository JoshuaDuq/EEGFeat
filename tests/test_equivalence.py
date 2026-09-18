import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import eegfeat as ef

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(
    not (FIXTURES / "manifest.json").exists(), reason="fixtures not generated"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(scope="module")
def spectra_npz(manifest: dict[str, Any]) -> Any:
    return np.load(FIXTURES / f"spectra_{manifest['subject']}.npz", allow_pickle=False)


@pytest.fixture(scope="module")
def reference(manifest: dict[str, Any]) -> Any:
    return np.load(FIXTURES / f"reference_{manifest['subject']}.npz", allow_pickle=False)


def _tfr_spectra(spectra_npz: Any, manifest: dict[str, Any]) -> ef.Spectra:
    windows = tuple(
        ef.Window(name, bounds[0], bounds[1]) for name, bounds in manifest["windows"].items()
    )
    data = spectra_npz["tfr"].astype(float)
    times = spectra_npz["tfr_times"]
    freqs = spectra_npz["tfr_freqs"]
    stacked = []
    coverage = []
    for window in windows:
        mask = (times >= window.tmin) & (times <= window.tmax)
        stacked.append(np.nanmean(data[:, :, :, mask], axis=3))
        coverage.append(np.isfinite(data[:, :, :, mask]).mean(axis=3))
    return ef.Spectra(
        data=np.stack(stacked, axis=2),
        freqs=freqs,
        ch_names=tuple(spectra_npz["ch_names"].tolist()),
        windows=windows,
        coverage=np.stack(coverage, axis=2),
        source="morlet",
    )


def _psd_spectra(reference: Any) -> ef.Spectra:
    data = reference["psd"].astype(float)[:, :, np.newaxis, :]
    return ef.Spectra(
        data=data,
        freqs=reference["psd_freqs"],
        ch_names=tuple(f"ch{i}" for i in range(data.shape[1])),
        windows=(ef.Window("all", -np.inf, np.inf),),
        coverage=np.isfinite(data).astype(float),
        source="welch",
    )


@pytest.mark.parametrize("band_name", ["delta", "theta", "alpha", "beta", "gamma"])
def test_band_power_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str
) -> None:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    spectra = _tfr_spectra(spectra_npz, manifest)
    table = ef.band_power(spectra, bands=(band,), include_global=False)
    for window_name in manifest["windows"]:
        expected = reference[f"power__{band_name}__{window_name}"]
        got = table.select(window=window_name).values
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    ("measure", "fn"),
    [
        ("centroid", ef.spectral_centroid),
        ("bandwidth", ef.spectral_bandwidth),
        ("entropy", ef.spectral_entropy),
    ],
)
@pytest.mark.parametrize("band_name", ["theta", "alpha", "beta"])
def test_descriptors_match_the_reference(
    reference: Any, manifest: dict[str, Any], measure: str, fn: Any, band_name: str
) -> None:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    got = fn(_psd_spectra(reference), band=band, include_global=False).values
    np.testing.assert_allclose(got, reference[f"{measure}__{band_name}"], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("band_name", ["theta", "alpha", "beta"])
def test_spectral_edge_matches_the_reference(
    reference: Any, manifest: dict[str, Any], band_name: str
) -> None:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    got = ef.spectral_edge(
        _psd_spectra(reference), band=band, percentile=0.95, include_global=False
    ).values
    np.testing.assert_allclose(got, reference[f"edge__{band_name}"], rtol=0.0, atol=1e-9)


def test_aperiodic_matches_the_reference(reference: Any) -> None:
    table = ef.aperiodic(_psd_spectra(reference), fit_range=(2.0, 40.0), include_global=False)
    np.testing.assert_allclose(
        table.select(measure="slope").values,
        reference["aperiodic_slope"],
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        table.select(measure="offset").values,
        reference["aperiodic_offset"],
        rtol=1e-6,
        atol=1e-9,
    )


def _band_signal(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str
) -> ef.BandSignal:
    band = ef.Band(band_name, manifest["bands"][band_name][0], manifest["bands"][band_name][1])
    analytic = reference[f"analytic__{band_name}"]
    sfreq = float(spectra_npz["sfreq"][0])
    n_times = analytic.shape[-1]
    times = np.arange(n_times) / sfreq - 7.0
    return ef.BandSignal.from_arrays(
        analytic=analytic,
        times=times,
        ch_names=tuple(spectra_npz["ch_names"].tolist()),
        band=band,
        sfreq=sfreq,
    )


@pytest.mark.parametrize("band_name", ["alpha", "beta"])
def test_band_signal_envelope_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str
) -> None:
    signal = _band_signal(spectra_npz, reference, manifest, band_name)
    expected = reference[f"envelope__{band_name}"]
    np.testing.assert_allclose(signal.envelope, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("band_name", ["alpha", "beta"])
@pytest.mark.parametrize(
    "measure",
    [
        "mean",
        "slope",
        "erd_magnitude",
        "erd_duration",
        "ers_magnitude",
        "ers_duration",
        "peak_latency",
        "onset_latency",
    ],
)
def test_erds_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str, measure: str
) -> None:
    signal = _band_signal(spectra_npz, reference, manifest, band_name)
    base = ef.Window("base", manifest["windows"]["base"][0], manifest["windows"]["base"][1])
    stim = ef.Window("stim", manifest["windows"]["stim"][0], manifest["windows"]["stim"][1])
    table = ef.erds([signal], baseline=base, windows=[stim], include_global=False)
    got = table.select(measure=measure).values
    expected = reference[f"erds_{measure}__{band_name}"]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("band_name", ["alpha", "beta"])
@pytest.mark.parametrize(
    "measure",
    ["count", "rate", "duration_mean", "amp_mean", "fraction_above"],
)
def test_bursts_matches_the_reference(
    spectra_npz: Any, reference: Any, manifest: dict[str, Any], band_name: str, measure: str
) -> None:
    signal = _band_signal(spectra_npz, reference, manifest, band_name)
    stim = ef.Window("stim", manifest["windows"]["stim"][0], manifest["windows"]["stim"][1])
    base = ef.Window("base", manifest["windows"]["base"][0], manifest["windows"]["base"][1])
    table = ef.burst_features(
        [signal],
        windows=[stim],
        baseline=base,
        threshold=0.75,
        min_duration_ms=100.0,
        include_global=False,
    )
    got = table.select(measure=measure).values
    expected = reference[f"burst_{measure}__{band_name}"]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)
