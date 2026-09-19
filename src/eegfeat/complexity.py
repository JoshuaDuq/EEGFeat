from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from eegfeat._expand import expand_signal
from eegfeat.signal import TimeSeries
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

_PAIR_BUDGET = 2_000_000
"""Template pairs held in memory at once.

Sample entropy is inherently O(n^2) in the window length. Comparing every pair at
once would need gigabytes on a multi-second window, so the comparison is chunked;
this bounds the working set without changing any result.
"""


def sample_entropy(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    order: int = 2,
    r: float = 0.2,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Sample entropy of the signal within each window.

    The negative log probability that two template vectors matching over ``order``
    samples still match over ``order + 1``. Larger values mean less
    self-similarity. Two templates match when their Chebyshev distance is strictly
    below ``r`` times the window's standard deviation.

    Undefined rather than zero when no pair of templates matches at all: the
    result is NaN when no length-``order`` pair matches, and infinite when some do
    but no length-``order + 1`` pair does.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows. Cost grows with the square of the window length.
    order : int, default 2
        Embedding dimension, the template length.
    r : float, default 0.2
        Tolerance as a fraction of the window's standard deviation.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Sample entropy, dimensionless.
    """
    _validate(order, r)

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times
        return {"sampen": _per_channel(trace, lambda x: _sample_entropy(x, order, r))}

    return expand_signal(
        series,
        trace_of=lambda s: s.amplitude,
        kernel=kernel,
        units={"sampen": "nats"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"order": order, "r": r},
    )


def multiscale_entropy(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    scales: Sequence[int] = (1, 2, 3, 4, 5),
    order: int = 2,
    r: float = 0.2,
    tolerance_mode: Literal["original_sd", "scale_sd"] = "original_sd",
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Sample entropy after coarse-graining, at each of several scales.

    Coarse-graining averages non-overlapping blocks of ``scale`` samples, so scale
    1 is the signal itself and larger scales describe slower structure. The
    ``tolerance_mode="original_sd"`` implements classical MSE by deriving the
    tolerance from the original window and holding it constant across scales.
    ``"scale_sd"`` recomputes it from each coarse-grained series.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    scales : sequence of int, default (1, 2, 3, 4, 5)
        Coarse-graining factors. Each becomes its own column, named
        ``mse{scale:02d}``.
    order : int, default 2
        Embedding dimension.
    r : float, default 0.2
        Tolerance as a fraction of the selected standard deviation.
    tolerance_mode : {"original_sd", "scale_sd"}, default "original_sd"
        Classical fixed tolerance, or scale-dependent tolerance.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        One column per scale. A scale that leaves too few samples to embed gives
        NaN rather than a value drawn from too little data.
    """
    _validate(order, r)
    if tolerance_mode not in ("original_sd", "scale_sd"):
        raise ValueError(
            "tolerance_mode must be 'original_sd' or 'scale_sd', got " f"{tolerance_mode!r}."
        )
    ordered = [int(s) for s in scales]
    if not ordered or any(s < 1 for s in ordered):
        raise ValueError(f"scales must be positive integers, got {list(scales)!r}.")

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times

        def at_scale(x: npt.NDArray[np.float64], scale: int) -> float:
            coarse = _coarse_grain(x, scale)
            tolerance = _tolerance(x, r) if tolerance_mode == "original_sd" else None
            return _sample_entropy(coarse, order, r, tolerance=tolerance)

        return {
            _scale_name(scale): _per_channel(trace, lambda x, k=scale: at_scale(x, k))
            for scale in ordered
        }

    return expand_signal(
        series,
        trace_of=lambda s: s.amplitude,
        kernel=kernel,
        units={_scale_name(scale): "nats" for scale in ordered},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={
            "scales": ordered,
            "order": order,
            "r": r,
            "tolerance_mode": tolerance_mode,
            "coarse_graining": "nonoverlapping_mean",
        },
    )


def _scale_name(scale: int) -> str:
    return f"mse{int(scale):02d}"


def _validate(order: int, r: float) -> None:
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}.")
    if not r > 0.0:
        raise ValueError(f"r must be positive, got {r}.")


def _per_channel(
    trace: npt.NDArray[np.float64],
    measure: object,
) -> npt.NDArray[np.float64]:
    n_epochs, n_channels, _ = trace.shape
    out = np.full((n_epochs, n_channels), np.nan)
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            out[epoch, channel] = measure(trace[epoch, channel])  # type: ignore[operator]
    return out


def _coarse_grain(x: npt.NDArray[np.float64], scale: int) -> npt.NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    if scale <= 1:
        return values.copy()
    blocks = values.size // scale
    if blocks <= 0:
        return np.array([], dtype=float)
    shaped = values[: blocks * scale].reshape(blocks, scale)
    valid = np.isfinite(shaped).all(axis=1)
    averaged = np.full(blocks, np.nan)
    averaged[valid] = shaped[valid].mean(axis=1)
    return averaged


def _tolerance(x: npt.NDArray[np.float64], r: float) -> float:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    deviation = float(np.std(finite))
    tolerance = r * deviation
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
            tolerance = max(float(np.finfo(float).eps), r * float(np.nanstd(finite)))
    return tolerance


def _sample_entropy(
    x: npt.NDArray[np.float64],
    order: int,
    r: float,
    *,
    tolerance: float | None = None,
) -> float:
    values = np.asarray(x, dtype=float)
    if values.size < order + 2:
        return float("nan")

    threshold = _tolerance(values, r) if tolerance is None else tolerance
    candidates = sliding_window_view(values, order + 1)
    templates = candidates[np.isfinite(candidates).all(axis=1)]
    n_templates = templates.shape[0]
    if n_templates < 2:
        return float("nan")

    matched_short = 0
    matched_long = 0
    chunk = max(1, _PAIR_BUDGET // max(n_templates * (order + 1), 1))
    for start in range(0, n_templates, chunk):
        stop = min(start + chunk, n_templates)
        distance = np.abs(templates[start:stop, np.newaxis, :] - templates[np.newaxis, :, :])
        # Each unordered pair once, matching antropy's positive-offset iteration.
        upper = np.arange(n_templates)[np.newaxis, :] > np.arange(start, stop)[:, np.newaxis]
        matched_short += int(((distance[..., :order].max(axis=2) < threshold) & upper).sum())
        matched_long += int(((distance.max(axis=2) < threshold) & upper).sum())

    if matched_short == 0:
        return float("nan")
    if matched_long == 0:
        return float("inf")
    return float(-np.log(matched_long / matched_short))
