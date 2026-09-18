"""Generate equivalence fixtures from the reference pipeline.

Run once, with the reference repository importable and its data drive mounted:

    PYTHONPATH=/Users/joduq24/Desktop/EEG_fMRI_Pipeline \
        python scripts/make_fixtures.py --subject sub-0001

Writes tests/fixtures/. Never run from the test suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

REFERENCE_ROOT = Path("/Users/joduq24/Desktop/EEG_fMRI_Pipeline")
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
WINDOWS = {"base": (-5.0, -1.0), "stim": (0.0, 8.0)}
N_EPOCHS_KEPT, N_CHANNELS_KEPT = 8, 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-0001")
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    path = (
        REFERENCE_ROOT
        / "data/derivatives/preprocessed/eeg"
        / args.subject
        / "eeg"
        / f"{args.subject}_task-thermalactive_epo.fif"
    )
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    epochs = epochs[:N_EPOCHS_KEPT].pick(epochs.ch_names[:N_CHANNELS_KEPT])

    freqs = np.logspace(np.log10(1.0), np.log10(100.0), 40)
    n_cycles = np.clip(freqs / 2.0, 3.0, 15.0)
    tfr = epochs.compute_tfr(
        "morlet", freqs=freqs, n_cycles=n_cycles, decim=4, return_itc=False, verbose="ERROR"
    )

    np.savez_compressed(
        args.out / f"spectra_{args.subject}.npz",
        tfr=np.asarray(tfr.get_data(), dtype=np.float32),
        tfr_times=np.asarray(tfr.times),
        tfr_freqs=freqs,
        n_cycles=n_cycles,
        ch_names=np.array(tfr.ch_names),
        sfreq=np.array([epochs.info["sfreq"]]),
        epochs_data=np.asarray(epochs.get_data(), dtype=np.float32),
    )
    np.savez_compressed(args.out / f"reference_{args.subject}.npz", **_reference(tfr, epochs))

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "subject": args.subject,
                "bands": BANDS,
                "windows": WINDOWS,
                "tfr": {
                    "freq_min": 1.0,
                    "freq_max": 100.0,
                    "n_freqs": 40,
                    "n_cycles": "clip(freq/2, 3, 15)",
                    "decim": 4,
                    "method": "morlet",
                },
                "n_epochs": int(len(epochs)),
                "n_channels": len(epochs.ch_names),
                "reference_commit": _reference_commit(),
            },
            indent=2,
        )
        + "\n"
    )


def _reference(tfr: object, epochs: object) -> dict[str, np.ndarray]:
    """Call the reference implementations and collect their outputs."""
    from eeg_pipeline.analysis.features import spectral
    from eeg_pipeline.utils.analysis.spectral import compute_psd

    out: dict[str, np.ndarray] = {}
    data = np.asarray(tfr.get_data())  # type: ignore[attr-defined]
    times = np.asarray(tfr.times)  # type: ignore[attr-defined]
    freqs = np.asarray(tfr.freqs)  # type: ignore[attr-defined]

    for window_name, (tmin, tmax) in WINDOWS.items():
        time_mask = (times >= tmin) & (times <= tmax)
        for band_name, (fmin, fmax) in BANDS.items():
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            if not freq_mask.any():
                continue
            out[f"power__{band_name}__{window_name}"] = spectral._compute_frequency_weighted_power(
                data, freq_mask, time_mask, freqs
            )

    psd_data = compute_psd(
        np.asarray(epochs.get_data()),  # type: ignore[attr-defined]
        float(epochs.info["sfreq"]),  # type: ignore[attr-defined]
        config={"feature_engineering.psd": {"n_fft": 1024}},
    )
    psd, psd_freqs = np.asarray(psd_data.psd), np.asarray(psd_data.freqs)
    out["psd"] = psd.astype(np.float32)
    out["psd_freqs"] = psd_freqs

    for band_name, (fmin, fmax) in BANDS.items():
        for label, fn in (
            ("centroid", spectral.compute_spectral_center),
            ("bandwidth", spectral.compute_spectral_bandwidth),
            ("entropy", spectral.compute_spectral_entropy),
        ):
            out[f"{label}__{band_name}"] = np.array(
                [
                    [fn(psd[e, c, :], psd_freqs, fmin, fmax) for c in range(psd.shape[1])]
                    for e in range(psd.shape[0])
                ]
            )
        out[f"edge__{band_name}"] = np.array(
            [
                [
                    spectral.compute_spectral_edge(psd[e, c, :], psd_freqs, fmin, fmax, 0.95)
                    for c in range(psd.shape[1])
                ]
                for e in range(psd.shape[0])
            ]
        )

    fit_mask = (psd_freqs >= 2.0) & (psd_freqs < 40.0)
    log_f = np.log10(np.where(psd_freqs > 0, psd_freqs, np.nan))
    slopes = np.full(psd.shape[:2], np.nan)
    offsets = np.full(psd.shape[:2], np.nan)
    for e in range(psd.shape[0]):
        for c in range(psd.shape[1]):
            with np.errstate(divide="ignore", invalid="ignore"):
                log_p = np.log10(np.where(psd[e, c, :] > 0, psd[e, c, :], np.nan))
            slope, offset = spectral._robust_aperiodic_fit(log_f, log_p, fit_mask)
            slopes[e, c] = np.nan if slope is None else slope
            offsets[e, c] = np.nan if offset is None else offset
    out["aperiodic_slope"], out["aperiodic_offset"] = slopes, offsets

    from eeg_pipeline.analysis.features.bursts import _extract_burst_metrics
    from eeg_pipeline.utils.analysis.spectral import compute_band_data

    epochs_data = np.asarray(epochs.get_data())
    sfreq = float(epochs.info["sfreq"])
    epoch_times = np.asarray(epochs.times)
    base_mask = (epoch_times >= WINDOWS["base"][0]) & (epoch_times <= WINDOWS["base"][1])
    stim_mask = (epoch_times >= WINDOWS["stim"][0]) & (epoch_times <= WINDOWS["stim"][1])
    active_times = epoch_times[stim_mask]

    for band_name in ("alpha", "beta"):
        fmin, fmax = BANDS[band_name]
        band_data = compute_band_data(
            epochs_data,
            sfreq,
            band=band_name,
            fmin=fmin,
            fmax=fmax,
            pad_sec=0.5,
            pad_cycles=3.0,
        )
        out[f"analytic__{band_name}"] = band_data.analytic
        out[f"envelope__{band_name}"] = band_data.envelope

        power = band_data.power
        # NOTE: these ERDS values are TRANSCRIBED from the expressions in
        # eeg_pipeline/analysis/features/precomputed/erds.py (per-channel path,
        # around lines 500-620), not produced by calling it: that function needs a
        # full PrecomputedData and FeatureContext. Comparing against this checks the
        # implementation against the documented formula, not against a pipeline run.
        # The band envelope and the burst metrics below ARE genuine reference calls.
        for metric in (
            "mean",
            "slope",
            "erd_magnitude",
            "erd_duration",
            "ers_magnitude",
            "ers_duration",
            "peak_latency",
            "onset_latency",
        ):
            vals = np.zeros(epochs_data.shape[:2], dtype=np.float64)
            for e in range(epochs_data.shape[0]):
                for c in range(epochs_data.shape[1]):
                    b_trace = power[e, c, base_mask]
                    ref = np.mean(b_trace)
                    std = np.std(b_trace)
                    tr = (power[e, c, stim_mask] - ref) / ref * 100.0
                    if metric == "mean":
                        vals[e, c] = np.mean(tr)
                    elif metric == "slope":
                        slope, _ = np.polyfit(active_times, tr, 1)
                        vals[e, c] = slope
                    elif metric == "erd_magnitude":
                        erd_vals = tr[tr < 0]
                        vals[e, c] = np.mean(np.abs(erd_vals)) if len(erd_vals) > 0 else 0.0
                    elif metric == "erd_duration":
                        erd_vals = tr[tr < 0]
                        vals[e, c] = len(erd_vals) / sfreq if len(erd_vals) > 0 else 0.0
                    elif metric == "ers_magnitude":
                        ers_vals = tr[tr > 0]
                        vals[e, c] = np.mean(ers_vals) if len(ers_vals) > 0 else 0.0
                    elif metric == "ers_duration":
                        ers_vals = tr[tr > 0]
                        vals[e, c] = len(ers_vals) / sfreq if len(ers_vals) > 0 else 0.0
                    elif metric == "peak_latency":
                        vals[e, c] = active_times[int(np.nanargmax(np.abs(tr)))]
                    elif metric == "onset_latency":
                        thresh = std / ref * 100.0
                        cross = np.abs(tr) > thresh
                        vals[e, c] = (
                            active_times[int(np.argmax(cross))] if np.any(cross) else np.nan
                        )
            out[f"erds_{metric}__{band_name}"] = vals

        stim_env = band_data.envelope[:, :, stim_mask]
        # Calibrated on the baseline window, as _resolve_burst_reference_envelope does.
        # Calibrating on the whole epoch lets the stimulus response raise the very
        # threshold used to detect bursts within it.
        thr = np.nanpercentile(band_data.envelope[:, :, base_mask], 75.0, axis=2)
        min_samples = max(1, int(round(100.0 * sfreq / 1000.0)))
        for b_metric, ref_key in (
            ("count", "count"),
            ("rate", "rate"),
            ("duration_mean", "duration_mean"),
            ("amp_mean", "amp_mean"),
            ("fraction_above", "fraction"),
        ):
            b_vals = np.zeros(epochs_data.shape[:2], dtype=np.float64)
            for e in range(epochs_data.shape[0]):
                for c in range(epochs_data.shape[1]):
                    res = _extract_burst_metrics(stim_env[e, c], sfreq, thr[e, c], min_samples)
                    b_vals[e, c] = res[ref_key]
            out[f"burst_{b_metric}__{band_name}"] = b_vals

    return out


def _reference_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
