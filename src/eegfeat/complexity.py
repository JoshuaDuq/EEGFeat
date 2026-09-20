from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from eegfeat._expand import expand_signal
from eegfeat.signal import TimeSeries
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable

_PAIR_BUDGET = 2_000_000
"""Template pairs compared at once.

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
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask
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
    raw_scales = list(scales)
    if (
        not raw_scales
        or any(
            isinstance(scale, bool) or not isinstance(scale, Integral) or scale < 1
            for scale in raw_scales
        )
        or len(set(raw_scales)) != len(raw_scales)
    ):
        raise ValueError(f"scales must be positive integers, got {list(scales)!r}.")
    ordered = [int(scale) for scale in raw_scales]

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask

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


def higuchi_fractal_dimension(
    series: Sequence[TimeSeries],
    *,
    windows: Sequence[Window],
    k_max: int = 10,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    r"""Higuchi's fractal dimension of the signal within each window.

    The curve is re-traced at a range of strides :math:`k`, and its mean length
    :math:`L(k)` falls as :math:`k^{-D}`. The dimension :math:`D` is the slope of
    :math:`\log L(k)` against :math:`-\log k`, and it measures how much structure
    survives coarse sampling: about 1 for a smooth oscillation, about 1.5 for
    Brownian motion, about 2 for white noise.

    It is not an alternative spelling of :func:`~eegfeat.sample_entropy`. Sample
    entropy asks how predictable the signal is from its own recent past; this
    asks how its length scales, and the two order real recordings differently.
    It is also far cheaper: linear in the window length rather than quadratic.

    Parameters
    ----------
    series : sequence of TimeSeries
        Raw signals or band envelopes.
    windows : sequence of Window
        Analysis windows.
    k_max : int, default 10
        Largest stride. The window needs more than ``k_max`` samples, and the
        estimate settles as ``k_max`` grows; 10 is the common choice.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Fractal dimension, dimensionless. NaN for a window with a non-finite
        sample or too few samples for the largest stride.

    References
    ----------
    Higuchi, T. (1988). Approach to an irregular time series on the basis of the
    fractal theory. Physica D: Nonlinear Phenomena, 31(2), 277-283.
    """
    if isinstance(k_max, bool) or not isinstance(k_max, Integral) or k_max < 2:
        raise ValueError(f"k_max must be an integer of at least 2, got {k_max!r}.")
    strides = int(k_max)

    def kernel(
        s: TimeSeries,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del s, times, mask
        return {"higuchi_fd": _per_channel(trace, lambda x: _higuchi(x, strides))}

    return expand_signal(
        series,
        trace_of=lambda s: s.amplitude,
        kernel=kernel,
        units={"higuchi_fd": "a.u."},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={"k_max": strides},
    )


def _higuchi(x: npt.NDArray[np.float64], k_max: int) -> float:
    """Curve length against stride, fitted in log-log space."""
    values = np.asarray(x, dtype=float)
    n = values.size
    # A gap would shorten one sub-curve and not the others, tilting the fit; the
    # measure is about how length scales, so a partial curve is not comparable.
    if n <= k_max or not np.isfinite(values).all():
        return float("nan")

    lengths = np.empty(k_max, dtype=float)
    for k in range(1, k_max + 1):
        per_start = np.empty(k, dtype=float)
        for m in range(k):
            sub = values[m::k]
            if sub.size < 2:
                per_start[m] = np.nan
                continue
            steps = int((n - m - 1) // k)
            # (n - 1) / (steps * k) restores the scale the stride removed, and the
            # remaining 1/k makes lengths at different strides comparable.
            per_start[m] = np.abs(np.diff(sub)).sum() * (n - 1) / (steps * k * k)
        lengths[k - 1] = np.nanmean(per_start)

    usable = np.isfinite(lengths) & (lengths > 0.0)
    if int(usable.sum()) < 2:
        return float("nan")
    log_k = -np.log(np.arange(1, k_max + 1, dtype=float))
    slope = np.polyfit(log_k[usable], np.log(lengths[usable]), 1)[0]
    return float(slope)


def _scale_name(scale: int) -> str:
    return f"mse{int(scale):02d}"


def _validate(order: int, r: float) -> None:
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}.")
    if isinstance(r, bool) or not isinstance(r, Real) or not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"r must be finite and positive, got {r!r}.")


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
    chunk = max(1, _PAIR_BUDGET // n_templates)
    for start in range(0, n_templates, chunk):
        stop = min(start + chunk, n_templates)
        block = templates[start:stop]
        # A Chebyshev match is a conjunction over the embedding dimensions, so the
        # pair mask is narrowed one dimension at a time. Reducing a stacked
        # (pairs x dimensions) distance array instead would hold order + 1 floats
        # per pair where this holds one byte.
        close = np.abs(block[:, 0, np.newaxis] - templates[np.newaxis, :, 0]) < threshold
        # Each unordered pair once, matching antropy's positive-offset iteration.
        close &= np.arange(n_templates)[np.newaxis, :] > np.arange(start, stop)[:, np.newaxis]
        for dimension in range(1, order):
            close &= (
                np.abs(block[:, dimension, np.newaxis] - templates[np.newaxis, :, dimension])
                < threshold
            )
        matched_short += int(close.sum())
        close &= np.abs(block[:, order, np.newaxis] - templates[np.newaxis, :, order]) < threshold
        matched_long += int(close.sum())

    if matched_short == 0:
        return float("nan")
    if matched_long == 0:
        return float("inf")
    return float(-np.log(matched_long / matched_short))
