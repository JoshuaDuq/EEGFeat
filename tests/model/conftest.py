from __future__ import annotations

import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable

ALPHA = Band("alpha", 8.0, 13.0)
BETA = Band("beta", 13.0, 30.0)


@pytest.fixture
def alpha_beta_table() -> FeatureTable:
    meta_alpha = FeatureMeta(
        measure="power",
        band=ALPHA,
        space="C3",
        space_kind="channel",
        window="stim",
        normalization="log_ratio",
        unit="log10",
        source="morlet",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("band_power", weighting="trapezoid"),
    )
    meta_beta = FeatureMeta(
        measure="power",
        band=BETA,
        space="C3",
        space_kind="channel",
        window="stim",
        normalization="log_ratio",
        unit="log10",
        source="morlet",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("band_power", weighting="trapezoid"),
    )
    return FeatureTable(
        values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        coverage=np.ones((2, 2)),
        meta=(meta_alpha, meta_beta),
        row_ids=(("sub-01", 0, "stim"), ("sub-02", 0, "stim")),
    )
