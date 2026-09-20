"""Batch feature extraction from preprocessed epochs files, driven by a recipe.

The library computes features from MNE objects you build yourself. The runner is
the layer above it: it reads a recipe, finds the epochs files, computes the
spectra and band signals each measure needs, and writes one feature table per
recording. It is what ``eegfeat run`` executes.
"""

from __future__ import annotations

from eegfeat.runner.batch import (
    CheckReport,
    Recording,
    RecordingResult,
    RunError,
    RunResult,
    Trial,
    TrialError,
    check,
    discover,
    run,
)
from eegfeat.runner.recipe import Recipe, RecipeError, load_recipe

__all__ = [
    "CheckReport",
    "Recipe",
    "RecipeError",
    "Recording",
    "RecordingResult",
    "RunError",
    "RunResult",
    "Trial",
    "TrialError",
    "check",
    "discover",
    "load_recipe",
    "run",
]
