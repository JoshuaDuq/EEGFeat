"""Importance and prediction intervals on the sleep-depth regression.

Held-out permutation importance must point at the features that carry sleep
depth, which on this montage is the occipital derivation. Conformal intervals
must reach their nominal coverage when calibration and test epochs are
exchangeable; across subjects they are not guaranteed to, as the docstring says,
and that is recorded rather than asserted.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import eegfeat.model as efm
from validation.loaders import Recording, load_sleep
from validation.test_regression import CONFIG, DEPTH, INNER, SEED, _relative_power
from validation.test_regression import design as design_fixture

design = design_fixture

DATASET = "sleep"


@pytest.mark.validates(
    "permutation_importance_over_folds",
    kind="decoding",
    claim="Held-out permutation importance points at the derivation that carries sleep depth",
    criterion="Pz-Oz carries more than 60 percent of the total; every value finite",
)
def test_importance_points_at_the_occipital_derivation(
    design: efm.Design, record: Callable[[str], None]
) -> None:
    folds = efm.loso_folds(design.groups)
    importance = efm.permutation_importance_over_folds(
        folds,
        design.X,
        design.y,
        design.groups,
        efm.ridge_pipeline(CONFIG, seed=SEED),
        efm.ridge_grid(design.X),
        inner=INNER,
        feature_names=design.column_names,
        n_repeats=5,
        seed=SEED,
    )
    assert importance.per_fold.shape == (len(folds), design.X.shape[1])
    assert np.isfinite(importance.values).all()
    assert importance.values.sum() > 0.0

    # Metadata for aggregate_by comes from the same columns the design was built on.
    recording = load_sleep(0)
    keep = recording.metadata["stage"].isin(DEPTH).to_numpy()
    table = _relative_power(Recording(recording.name, recording.epochs[keep]))
    by_space = efm.aggregate_by(importance, table.meta, "space")
    total = sum(by_space.values())
    record(f"Pz-Oz carries {by_space['Pz-Oz'] / total:.2f} of the total importance")
    assert by_space["Pz-Oz"] / total > 0.6, by_space


@pytest.mark.validates(
    "prediction_intervals",
    kind="behaviour",
    claim="Conformal intervals reach nominal coverage on exchangeable epochs",
    criterion="split and CV+ within 0.85 to 0.95 at 90 percent nominal; quantile at least 0.9",
)
@pytest.mark.parametrize("method", ["split", "cv_plus", "quantile"])
def test_intervals_cover_exchangeable_epochs(
    design: efm.Design, method: str, record: Callable[[str], None]
) -> None:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(design.y))
    calibrate, test = order[: 2 * len(order) // 3], order[2 * len(order) // 3 :]
    intervals = efm.prediction_intervals(
        efm.ridge_pipeline(CONFIG, seed=SEED),
        design.X[calibrate],
        design.y[calibrate],
        design.X[test],
        alpha=0.1,
        method=method,  # type: ignore[arg-type]
        seed=SEED,
    )
    assert np.isfinite(intervals.lower).all() and np.isfinite(intervals.upper).all()
    assert np.all(intervals.lower <= intervals.upper)
    covered = np.mean((design.y[test] >= intervals.lower) & (design.y[test] <= intervals.upper))
    record(f"{method}: coverage {covered:.3f} at 90 percent nominal")
    if method == "quantile":
        assert covered >= 0.9, covered
    else:
        assert 0.85 <= covered <= 0.95, covered
