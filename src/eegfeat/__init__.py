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
from eegfeat.bursts import (
    burst_amplitude,
    burst_count,
    burst_duration,
    burst_rate,
    fraction_above_threshold,
)
from eegfeat.complexity import multiscale_entropy, sample_entropy
from eegfeat.derived import asymmetry, band_ratio
from eegfeat.descriptors import (
    peak_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_edge,
    spectral_entropy,
)
from eegfeat.erds import (
    erd_duration,
    erd_magnitude,
    erds_mean,
    erds_onset_latency,
    erds_peak_latency,
    erds_rebound_latency,
    erds_slope,
    ers_duration,
    ers_magnitude,
)
from eegfeat.power import band_power
from eegfeat.signal import BandSignal, Signal
from eegfeat.spectra import Spectra, Window
from eegfeat.table import FeatureMeta, FeatureTable, concat
from eegfeat.temporal import (
    area_under_curve,
    mean_amplitude,
    peak_amplitude,
    peak_latency,
    peak_to_peak,
    variance,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "aperiodic",
    "aperiodic_ratio",
    "area_under_curve",
    "asymmetry",
    "Band",
    "band_power",
    "band_ratio",
    "BANDS_STANDARD",
    "BandSignal",
    "burst_amplitude",
    "burst_count",
    "burst_duration",
    "burst_rate",
    "concat",
    "erd_duration",
    "erd_magnitude",
    "erds_mean",
    "erds_onset_latency",
    "erds_peak_latency",
    "erds_rebound_latency",
    "erds_slope",
    "ers_duration",
    "ers_magnitude",
    "FeatureMeta",
    "FeatureTable",
    "fraction_above_threshold",
    "mean_amplitude",
    "multiscale_entropy",
    "peak_amplitude",
    "peak_frequency",
    "peak_latency",
    "peak_to_peak",
    "sample_entropy",
    "Signal",
    "Spectra",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_edge",
    "spectral_entropy",
    "variance",
    "Window",
]
