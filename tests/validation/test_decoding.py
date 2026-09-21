"""Leakage-safe decoding of known contrasts on public data.

The modeling layer is validated where the effect is strong enough that a correct
pipeline must find it: movement against rest on the motor dataset, and wake
against deep sleep on Sleep-EDF. Left against right hand is deliberately not a
decoding check here; on this dataset with these trial counts it is a weak
contrast, and the group-level lateralization tests already cover its direction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

import eegfeat as ef
import eegfeat.model as efm
from validation.loaders import Recording, load_sleep

SEED = 0
CONFIG = efm.PreprocessingConfig(max_feature_missingness=0.2)

# The sensorimotor strip, where movement-related desynchronization is expected.
STRIP = [
    *("FC5", "FC3", "FC1", "FC2", "FC4", "FC6"),
    *("C5", "C3", "C1", "Cz", "C2", "C4", "C6"),
    *("CP5", "CP3", "CP1", "CP2", "CP4", "CP6"),
]
MU = ef.Band("mu", 8.0, 13.0)
BETA = ef.Band("beta", 13.0, 30.0)
MOTOR_WINDOW = {"tmin": 0.5, "tmax": 3.5}

# Three, not two: leave-one-subject-out folds tune on the training subjects, and
# subject-grouped inner splits need at least two of them.
SLEEP_SUBJECTS = (0, 1, 2)
SLEEP_BANDS = [
    ef.Band("delta", 0.5, 4.0),
    ef.Band("theta", 4.0, 8.0),
    ef.Band("alpha", 8.0, 12.0),
    ef.Band("sigma", 12.0, 15.0),
    ef.Band("beta", 15.0, 30.0),
]

DATASET = "eegbci"


def _targets(table: ef.FeatureTable, metadata: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(table.row_ids, columns=["recording", "epoch", "event"])
    return pd.concat([keys, metadata.reset_index(drop=True)], axis=1)


def _log_band_power(
    recording: Recording, bands: list[ef.Band], picks: str | list[str], **window: float
) -> ef.FeatureTable:
    sfreq = recording.epochs.info["sfreq"]
    parameters = {
        "method": "welch",
        "fmin": min(band.fmin for band in bands),
        "fmax": max(band.fmax for band in bands),
        "n_fft": int(2 * sfreq),
        "n_per_seg": int(2 * sfreq),
        "n_overlap": int(sfreq),
        **window,
    }
    spectrum = recording.epochs.compute_psd(picks=picks, **parameters)
    spectra = ef.Spectra.from_spectrum(
        spectrum, recording=recording.name, estimator_parameters=parameters
    )
    return ef.integrated_band_power(spectra, bands=bands, normalize="log10", include_global=False)


def _classify(
    design: efm.Design, folds: tuple[efm.Fold, ...], inner: efm.InnerSplit
) -> tuple[efm.ClassificationResult, float]:
    """Cross-fit a tuned logistic regression; return metrics and a binomial p-value."""
    results = efm.cross_fit_classification(
        folds,
        design.X,
        design.y.astype(np.intp),
        design.groups,
        efm.logistic_pipeline(CONFIG, seed=SEED),
        efm.logistic_grid(),
        inner=inner,
        seed=SEED,
        runs=design.runs,
    )
    y_true = np.concatenate([r.y_true for r in results])
    y_pred = np.concatenate([r.y_pred for r in results])
    groups = np.concatenate([design.groups[r.rows] for r in results])
    assert len(y_true) == len(design.y), "every epoch must be predicted exactly once"
    metrics = efm.classification_metrics(y_true, y_pred, groups=groups)
    correct = int((y_true == y_pred).sum())
    return metrics, binomtest(correct, len(y_true), 0.5, alternative="greater").pvalue


def _describe(metrics: efm.ClassificationResult, p_value: float) -> str:
    per_subject = [m["balanced_accuracy"] for m in metrics.per_subject.values()]
    return (
        f"balanced accuracy {metrics.balanced_accuracy:.2f}, per subject "
        f"{min(per_subject):.2f} to {max(per_subject):.2f}, {len(metrics.y_true)} epochs, "
        f"p = {p_value:.0e}"
    )


@pytest.fixture(scope="module")
def motor_design(eegbci_recordings: list[Recording]) -> efm.Design:
    tables, targets = [], []
    for recording in eegbci_recordings:
        table = _log_band_power(recording, [MU, BETA], STRIP, **MOTOR_WINDOW)
        tables.append(table)
        targets.append(_targets(table, recording.metadata))
    return efm.build_design(
        ef.stack_rows(tables, columns="identical"),
        pd.concat(targets, ignore_index=True),
        target="moving",
        groups="subject",
        runs="run",
    )


@pytest.mark.validates(
    "stack_rows",
    "build_design",
    "within_subject_folds",
    "cross_fit_classification",
    "classification_metrics",
    kind="decoding",
    claim="Movement against rest decodes within subject from sensorimotor power",
    criterion="balanced accuracy above 0.65 (chance 0.5); binomial p below 1e-6",
)
def test_movement_is_decodable_from_sensorimotor_power_within_subject(
    motor_design: efm.Design, record: Callable[[str], None]
) -> None:
    folds = efm.within_subject_folds(motor_design.groups, motor_design.runs, inner_splits=3)
    metrics, p_value = _classify(motor_design, folds, efm.InnerSplit(grouping="run", n_splits=2))
    record(_describe(metrics, p_value))
    assert metrics.balanced_accuracy > 0.65, metrics.balanced_accuracy
    assert p_value < 1e-6, p_value


@pytest.mark.validates(
    "loso_folds",
    "cross_fit_classification",
    "classification_metrics",
    kind="decoding",
    claim="Movement against rest decodes across subjects, leave-one-subject-out",
    criterion="balanced accuracy above 0.55 (chance 0.5); binomial p below 1e-3",
)
def test_movement_decoding_transfers_across_subjects(
    motor_design: efm.Design, record: Callable[[str], None]
) -> None:
    folds = efm.loso_folds(motor_design.groups)
    metrics, p_value = _classify(
        motor_design, folds, efm.InnerSplit(grouping="subject", n_splits=3)
    )
    record(_describe(metrics, p_value))
    assert metrics.balanced_accuracy > 0.55, metrics.balanced_accuracy
    assert p_value < 1e-3, p_value


@pytest.fixture(scope="module")
def sleep_design() -> efm.Design:
    tables, targets = [], []
    for subject in SLEEP_SUBJECTS:
        recording = load_sleep(subject)
        keep = recording.metadata["stage"].isin(["W", "N3"]).to_numpy()
        epochs = recording.epochs[keep]
        recording = Recording(recording.name, epochs)
        table = _log_band_power(recording, SLEEP_BANDS, "eeg")
        tables.append(table)
        metadata = recording.metadata.assign(deep=(recording.metadata["stage"] == "N3").astype(int))
        targets.append(_targets(table, metadata))
    return efm.build_design(
        ef.stack_rows(tables, columns="identical"),
        pd.concat(targets, ignore_index=True),
        target="deep",
        groups="subject",
    )


@pytest.mark.validates(
    "loso_folds",
    "cross_fit_classification",
    "classification_metrics",
    kind="decoding",
    claim="Deep sleep against wake decodes across subjects",
    criterion="balanced accuracy above 0.9 (chance 0.5)",
)
def test_deep_sleep_is_decodable_across_subjects(
    sleep_design: efm.Design, record: Callable[[str], None]
) -> None:
    folds = efm.loso_folds(sleep_design.groups)
    metrics, p_value = _classify(
        sleep_design, folds, efm.InnerSplit(grouping="subject", n_splits=2)
    )
    record(_describe(metrics, p_value))
    assert metrics.balanced_accuracy > 0.9, metrics.balanced_accuracy
    assert p_value < 1e-10, p_value
