"""Regenerate the example outputs in this folder.

The recordings are simulated, not real participant data, so the files can live in the
repository and anyone can reproduce them:

    python examples/make_examples.py

Five subjects perform two runs of sixteen trials. Each trial has an ``intensity``; alpha
over the central channels is suppressed during the stimulus in proportion to it, and the
``rating`` follows the same intensity with a lot of noise on top. So there is a real signal
to find and a realistic amount of it to miss. None of this is a claim about a real effect.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np
import pandas as pd

import eegfeat.model as efm
from eegfeat.io import read_dataset
from eegfeat.runner import load_recipe, run

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "recipe.toml"
SHOWCASE = "sub-01_task-pain_run-01"

CHANNELS = ("F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz")
SFREQ = 200.0
TMIN, TMAX = -2.0, 4.0
SUBJECTS = 5
RUNS = 2
TRIALS = 16
SEED = 11


def simulate(subject: int, run_index: int, rng: np.random.Generator) -> mne.EpochsArray:
    """One run of alpha suppressed in proportion to a per-trial stimulus intensity."""
    times = np.arange(round(TMIN * SFREQ), round(TMAX * SFREQ) + 1) / SFREQ
    stimulus = (times >= 0.5) & (times <= 3.5)
    intensity = rng.choice([1.0, 2.0, 3.0, 4.0], size=TRIALS)
    # A per-subject offset in both the signal and the rating, so pooling trials across
    # subjects would flatter the model and leaving a subject out does not.
    offset = 6.0 * subject
    # Only the central channels carry the effect, so the feature importances have a
    # topography to find rather than the same story on every electrode.
    modulated = np.array([name in ("C3", "Cz", "C4") for name in CHANNELS])

    data = np.empty((TRIALS, len(CHANNELS), times.size))
    for trial, level in enumerate(intensity):
        suppression = np.where(stimulus, 1.0 - 0.18 * level, 1.0)
        envelope = np.where(modulated[:, None], suppression[None, :], 1.0)
        alpha = 12e-6 * envelope * np.sin(2 * np.pi * 10.0 * times + rng.uniform(0, 2 * np.pi))
        theta = 5e-6 * np.sin(2 * np.pi * 6.0 * times + rng.uniform(0, 2 * np.pi))
        data[trial] = alpha + theta + 6e-6 * rng.standard_normal((len(CHANNELS), times.size))

    # Most of what a participant reports is not in the EEG. A noiseless target would make
    # the example look like EEG predicts pain far better than it does.
    rating = 20.0 + 12.0 * intensity + offset + rng.normal(0.0, 25.0, TRIALS)
    metadata = pd.DataFrame(
        {
            "subject": f"sub-{subject + 1:02d}",
            "run": f"run-{run_index + 1:02d}",
            "trial": np.arange(1, TRIALS + 1),
            "intensity": intensity,
            "rating": rating.round(2),
            "painful": (intensity >= 3.0).astype(int),
        }
    )
    events = np.column_stack(
        [np.arange(TRIALS) * int(SFREQ * 10), np.zeros(TRIALS, int), np.ones(TRIALS, int)]
    )
    info = mne.create_info(list(CHANNELS), SFREQ, "eeg")
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=TMIN,
        event_id={"stim": 1},
        metadata=metadata,
        verbose="error",
    )


def write_epochs(root: Path) -> None:
    rng = np.random.default_rng(SEED)
    for subject in range(SUBJECTS):
        for run_index in range(RUNS):
            name = f"sub-{subject + 1:02d}_task-pain_run-{run_index + 1:02d}_epo.fif"
            path = root / f"sub-{subject + 1:02d}" / "eeg" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            simulate(subject, run_index, rng).save(path, overwrite=True, verbose="error")


def model(features: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-subject-out ridge regression on the cohort the runner just wrote."""
    dataset = read_dataset(sorted(features.rglob("*_features.tsv")))
    design = efm.build_design(
        dataset.table,
        dataset.targets,
        target="rating",
        groups="subject",
        runs="run",
        selection=efm.Selection(measure=("band_power", "erds_mean")),
    )

    config = efm.PreprocessingConfig(max_feature_missingness=0.2)
    pipeline = efm.ridge_pipeline(config, seed=SEED)
    folds = efm.loso_folds(design.groups)
    inner = efm.InnerSplit(grouping="subject", n_splits=3)
    # The permutation test has to repeat the cross-fitting exactly, so both calls take
    # the same settings; only their run labels differ in how they are passed.
    shared = {"inner": inner, "seed": SEED, "harmonization": "intersection"}

    results = efm.cross_fit_regression(
        folds,
        design.X,
        design.y,
        design.groups,
        pipeline,
        efm.ridge_grid(),
        runs=design.runs,
        **shared,
    )
    y_true, y_pred, groups, rows, fold_ids = efm.fold_results(results, groups=design.groups)
    groups = np.asarray(groups, dtype=object)
    rows = np.asarray(rows, dtype=int)
    metrics, _ = efm.regression_metrics(y_true, y_pred, groups)
    summary = efm.subject_level_r(
        pd.DataFrame({"subject_id": groups, "y_true": y_true, "y_pred": y_pred})
    )
    null = efm.permutation_test(
        folds,
        design.X,
        design.y,
        design.groups,
        design.runs,
        pipeline,
        efm.ridge_grid(),
        metrics["subject_level_r"],
        config=efm.NullConfig(scheme="within_subject", n_permutations=200),
        **shared,
    )

    predictions = pd.DataFrame(
        {
            "recording": [design.row_ids[row][0] for row in rows],
            "epoch": [design.row_ids[row][1] for row in rows],
            "subject": groups,
            "run": design.runs[rows],
            "held_out_fold": fold_ids,
            "rating_observed": np.round(y_true, 2),
            "rating_predicted": np.round(y_pred, 2),
        }
    )

    scores = pd.DataFrame(
        [
            {
                "scope": "cohort",
                "n_epochs": int(metrics["n"]),
                "r": round(summary.r, 4),
                "ci_low": round(summary.ci_low, 4),
                "ci_high": round(summary.ci_high, 4),
                "permutation_p": round(null.p_value, 4),
            },
            *(
                {
                    "scope": subject,
                    "n_epochs": int((groups == subject).sum()),
                    "r": round(value, 4),
                    "ci_low": None,
                    "ci_high": None,
                    "permutation_p": None,
                }
                for subject, value in summary.per_subject
            ),
        ]
    )
    return predictions, scores


def main() -> None:
    mne.set_log_level("ERROR")
    with TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        write_epochs(workspace / "epochs")
        shutil.copy(RECIPE, workspace / "recipe.toml")
        # Run from the recipe's own directory, as the command line does, so the recording
        # identities in the outputs stay relative and readable.
        with contextlib.chdir(workspace):
            result = run(load_recipe(Path("recipe.toml")))
            if not result.ok:
                failures = ", ".join(r.label for r in result.failed)
                raise SystemExit(f"feature extraction failed for: {failures}")

            features = workspace / "features"
            produced = features / "sub-01" / "eeg"
            for name in (
                f"{SHOWCASE}_features.tsv",
                f"{SHOWCASE}_features_coverage.tsv",
                f"{SHOWCASE}_features.json",
                f"{SHOWCASE}_crosstrial.tsv",
                f"{SHOWCASE}_crosstrial_coverage.tsv",
                f"{SHOWCASE}_crosstrial.json",
            ):
                shutil.copy(produced / name, HERE / name)

            predictions, scores = model(features)

    predictions.to_csv(HERE / "example_model_predictions.tsv", sep="\t", index=False, na_rep="n/a")
    scores.to_csv(HERE / "example_model_scores.tsv", sep="\t", index=False, na_rep="n/a")
    print(f"wrote {len(list(HERE.glob('*.tsv')))} TSV files to {HERE}")


if __name__ == "__main__":
    main()
