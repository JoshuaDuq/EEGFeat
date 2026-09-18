"""Labelled spectral feature extraction for MNE objects.

`eegfeat` turns a computed ``Spectrum`` or ``EpochsTFR`` into a
:class:`~eegfeat.table.FeatureTable`: values plus one metadata record per
column, so band, channel, window and normalization are structured fields rather
than fragments of a column name.

It computes no time-frequency transform of its own.
"""

from __future__ import annotations

from eegfeat.aperiodic import aperiodic, aperiodic_ratio
from eegfeat.bands import BANDS_STANDARD, Band
from eegfeat.bursts import burst_features
from eegfeat.derived import asymmetry, band_ratio
from eegfeat.descriptors import (
    peak_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_edge,
    spectral_entropy,
)
from eegfeat.erds import erds
from eegfeat.power import band_power
from eegfeat.signal import BandSignal
from eegfeat.spectra import Spectra, Window
from eegfeat.table import FeatureMeta, FeatureTable, concat

__version__ = "0.1.0.dev0"

__all__ = [
    "BANDS_STANDARD",
    "Band",
    "BandSignal",
    "FeatureMeta",
    "FeatureTable",
    "Spectra",
    "Window",
    "__version__",
    "aperiodic",
    "aperiodic_ratio",
    "asymmetry",
    "band_power",
    "band_ratio",
    "burst_features",
    "concat",
    "erds",
    "peak_frequency",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_edge",
    "spectral_entropy",
]
