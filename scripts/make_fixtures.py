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
