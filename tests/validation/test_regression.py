"""Leakage-safe regression and its permutation null on Sleep-EDF.

Sleep depth, scored W, N1, N2, N3 as 0 to 3, is an ordinal target that relative
band power predicts well. Leave-one-subject-out ridge regression over three
subjects must recover it, the subject-level correlation's interval must exclude
zero, and refitting the whole pipeline under within-subject permuted targets must
put the null near zero and the observed value in its far tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eegfeat as ef
import eegfeat.model as efm
from validation.loaders import Recording, load_sleep

SEED = 0
SUBJECTS = (0, 1, 2)
DEPTH = {"W": 0, "N1": 1, "N2": 2, "N3": 3}
TOTAL = ef.Band("total", 0.5, 30.0)
BANDS = [
    ef.Band("delta", 0.5, 4.0),
    ef.Band("theta", 4.0, 8.0),
    ef.Band("alpha", 8.0, 12.0),
    ef.Band("sigma", 12.0, 15.0),
    ef.Band("beta", 15.0, 30.0),
]
CONFIG = efm.PreprocessingConfig(max_feature_missingness=0.2)
INNER = efm.InnerSplit(grouping="subject", n_splits=2)


def _relative_power(recording: Recording) -> ef.FeatureTable:
    sfreq = recording.epochs.info["sfreq"]
    parameters = {
        "method": "welch",
        "fmin": TOTAL.fmin,
        "fmax": TOTAL.fmax,
        "n_fft": int(2 * sfreq),
        "n_per_seg": int(2 * sfreq),
        "n_overlap": int(sfreq),
    }
    spectrum = recording.epochs.compute_psd(**parameters)
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=recording.name, estimator_parameters=parameters
    )
    power = ef.integrated_band_power(
        spectra, bands=[*BANDS, TOTAL], normalize="log10", include_global=False
    )
    return ef.concat([ef.band_ratio(power, band.name, TOTAL.name) for band in BANDS])


@pytest.fixture(scope="module")
def design() -> efm.Design:
    tables, targets = [], []
    for subject in SUBJECTS:
        recording = load_sleep(subject)
        keep = recording.metadata["stage"].isin(DEPTH).to_numpy()
        recording = Recording(recording.name, recording.epochs[keep])
        table = _relative_power(recording)
        tables.append(table)
        keys = pd.DataFrame(table.row_ids, columns=["recording", "epoch", "event"])
        metadata = recording.metadata.reset_index(drop=True)
        targets.append(
            pd.concat([keys, metadata.assign(depth=metadata["stage"].map(DEPTH))], axis=1)
        )
    return efm.build_design(
        ef.stack_rows(tables, columns="identical"),
        pd.concat(targets, ignore_index=True),
        target="depth",
        groups="subject",
    )


@pytest.fixture(scope="module")
def predictions(design: efm.Design) -> tuple[pd.DataFrame, dict[str, float]]:
    folds = efm.loso_folds(design.groups)
    results = efm.cross_fit_regression(
        folds,
        design.X,
        design.y,
        design.groups,
        efm.ridge_pipeline(CONFIG, seed=SEED),
        efm.ridge_grid(),
        inner=INNER,
        seed=SEED,
    )
    y_true, y_pred, groups, rows, _ = efm.fold_results(results, groups=design.groups)
    assert sorted(rows) == list(range(len(design.y))), "every epoch predicted exactly once"
    metrics, _ = efm.regression_metrics(y_true, y_pred, np.asarray(groups, dtype=object))
    frame = pd.DataFrame({"subject_id": groups, "y_true": y_true, "y_pred": y_pred})
    return frame, metrics


def test_sleep_depth_is_recovered_across_subjects(
    predictions: tuple[pd.DataFrame, dict[str, float]],
) -> None:
    frame, metrics = predictions
    summary = efm.subject_level_r(frame)
    assert summary.r > 0.7, summary
    assert summary.ci_low > 0.5, summary
    assert all(r > 0.7 for _, r in summary.per_subject), summary.per_subject
    assert metrics["r2"] > 0.5, metrics
    assert metrics["subject_level_r"] == pytest.approx(summary.r)


def test_permutation_null_sits_at_zero(
    design: efm.Design, predictions: tuple[pd.DataFrame, dict[str, float]]
) -> None:
    _, metrics = predictions
    null = efm.permutation_test(
        efm.loso_folds(design.groups),
        design.X,
        design.y,
        design.groups,
        None,
        efm.ridge_pipeline(CONFIG, seed=SEED),
        efm.ridge_grid(),
        metrics["subject_level_r"],
        config=efm.NullConfig(scheme="within_subject", n_permutations=50),
        inner=INNER,
        seed=SEED,
    )
    assert null.observed == pytest.approx(metrics["subject_level_r"])
    assert null.n_incomplete == 0
    assert np.abs(null.null).max() < 0.2, null.null.round(3)
    assert null.p_value < 0.05, null.p_value
    # Within-subject shuffling must actually move most labels.
    assert null.changed_fractions.min() > 0.5
