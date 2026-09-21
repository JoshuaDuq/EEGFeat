"""The aperiodic (1/f) fit on Sleep-EDF.

Slow-wave sleep tilts the spectrum: power piles up below 4 Hz and drains from the
faster bands, so the log-log slope of the aperiodic component steepens from wake
through N2 to N3. The fit is also checked against plain least squares, which it
must reproduce exactly when peak rejection is switched off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import eegfeat as ef
from validation.loaders import Recording

FIT_RANGE = (2.0, 30.0)


@pytest.fixture(scope="module")
def spectra(sleep_recordings: list[Recording]) -> list[ef.Spectra]:
    out = []
    for recording in sleep_recordings:
        sfreq = recording.epochs.info["sfreq"]
        parameters = {
            "method": "welch",
            "fmin": 0.5,
            "fmax": 30.0,
            "n_fft": int(2 * sfreq),
            "n_per_seg": int(2 * sfreq),
            "n_overlap": int(sfreq),
        }
        spectrum = recording.epochs.compute_psd(**parameters)
        out.append(
            ef.Spectra.from_spectrum(
                spectrum, recording=recording.name, estimator_parameters=parameters
            )
        )
    return out


def _least_squares_slopes(spectra: ef.Spectra) -> np.ndarray:
    inside = ef.Band("fit", *FIT_RANGE).mask(spectra.freqs)
    log_f = np.log10(spectra.freqs[inside])
    log_p = np.log10(spectra.data[:, :, 0, inside])
    return np.array([[np.polyfit(log_f, channel, 1)[0] for channel in epoch] for epoch in log_p])


def test_without_peak_rejection_the_fit_is_ordinary_least_squares(
    spectra: list[ef.Spectra],
) -> None:
    for spectrum in spectra:
        fit = ef.aperiodic(
            spectrum,
            fit_range=FIT_RANGE,
            include_global=False,
            peak_rejection_z=1e9,
            max_iterations=1,
        )
        np.testing.assert_allclose(
            fit.select(measure="slope").values, _least_squares_slopes(spectrum), rtol=1e-9
        )


def test_peak_rejection_only_nudges_the_slope(spectra: list[ef.Spectra]) -> None:
    """Rejecting points above the line changes the slope a little, and never wildly."""
    for spectrum in spectra:
        robust = ef.aperiodic(spectrum, fit_range=FIT_RANGE, include_global=False)
        slope = robust.select(measure="slope").values
        plain = _least_squares_slopes(spectrum)
        assert np.corrcoef(slope.ravel(), plain.ravel())[0, 1] > 0.98
        assert np.median(np.abs(slope - plain)) < 0.1


def test_flattened_spectrum_is_centred_on_one(spectra: list[ef.Spectra]) -> None:
    """Dividing by the fitted line leaves a spectrum whose typical value is unity."""
    for spectrum in spectra:
        inside = ef.Band("fit", *FIT_RANGE).mask(spectrum.freqs)
        ratio = ef.aperiodic_ratio(spectrum, fit_range=FIT_RANGE)
        assert abs(np.nanmedian(np.log10(ratio.data[:, :, 0, inside]))) < 0.05


def test_slope_steepens_with_sleep_depth(
    sleep_recordings: list[Recording], spectra: list[ef.Spectra]
) -> None:
    for recording, spectrum in zip(sleep_recordings, spectra, strict=True):
        fit = ef.aperiodic(spectrum, fit_range=FIT_RANGE, include_global=False)
        slope = fit.select(measure="slope")
        frame = pd.DataFrame(slope.values, columns=[m.space for m in slope.meta])
        frame["stage"] = recording.metadata["stage"].to_numpy()

        medians = frame.groupby("stage").median()
        for channel in medians.columns:
            ordered = medians.loc[["W", "N2", "N3"], channel]
            assert ordered.is_monotonic_decreasing, (recording.name, channel, ordered)

        kept = frame[frame["stage"].isin(["W", "N3"])]
        deep = (kept["stage"] == "N3").astype(int)
        for channel in medians.columns:
            assert roc_auc_score(deep, kept[channel]) < 0.1, (recording.name, channel)
