from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt

from eegfeat._expand import check_signals, window_mask
from eegfeat.bands import Band
from eegfeat.signal import BandSignal, Signal
from eegfeat.spectra import Window
from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable


def envelope_correlation(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
) -> FeatureTable:
    """Amplitude envelope correlation between every pair of nodes.

    The Pearson correlation of band envelopes, averaged over the trials in each
    group. Nodes are channels, or ROIs when ``groups`` is given, in which case a
    node's envelope is the mean envelope of its member channels.

    **Estimated across trials, so the result has one row per trial group.** See
    :func:`~eegfeat.itpc` for why that is not broadcast to per-epoch rows.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None uses every channel as a node, which on
        a 64-channel montage is 2016 pairs.
    trials : sequence of str, optional
        A label per epoch. One row per distinct label; None gives a single row
        labelled ``"all"``.

    Returns
    -------
    FeatureTable
        One column per node pair, band and window, with ``space_kind="pair"``.
    """
    return _pairwise(
        signals,
        measure="aec",
        unit="r",
        windows=windows,
        groups=groups,
        trials=trials,
        matrix=_correlation_matrix,
    )


def wpli(
    signal: Signal,
    *,
    bands: Sequence[Band],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
) -> FeatureTable:
    """Weighted phase lag index between every pair of nodes.

    Delegates the estimation to ``mne_connectivity.spectral_connectivity_epochs``,
    ensuring standard cross-spectral density calculation.

    Takes a broadband :class:`~eegfeat.Signal` and a list of bands, rather than
    pre-filtered :class:`~eegfeat.BandSignal` objects, because the band is a
    parameter of the spectral estimation rather than a prior filtering step.

    Requires the optional dependency: ``pip install eegfeat[connectivity]``.

    Parameters
    ----------
    signal : Signal
        Broadband epochs.
    bands : sequence of Band
        Bands to estimate in.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. Channel-level connectivity is averaged
        within each ROI block.
    trials : sequence of str, optional
        A label per epoch. One row per distinct label.

    Returns
    -------
    FeatureTable
        One column per node pair, band and window, with ``space_kind="pair"``.
    """
    estimator = _require_mne_connectivity()
    row_groups, labels = _resolve_rows(trials, signal.data.shape[0])
    node_names, picks = _nodes(signal.ch_names, groups)

    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for band in bands:
        for window in windows:
            mask = window_mask(signal.times, window)
            matrices = []
            for row in range(len(labels)):
                data = signal.data[row_groups == row][:, :, mask]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = estimator(
                        data,
                        method="wpli",
                        sfreq=signal.sfreq,
                        fmin=band.fmin,
                        fmax=band.fmax,
                        faverage=True,
                        verbose=False,
                    )
                matrices.append(_dense(result))
            columns.extend(
                _pair_columns(
                    np.stack([_reduce_to_nodes(m, picks) for m in matrices]),
                    node_names,
                    band,
                    window,
                    "wpli",
                    "a.u.",
                    signal,
                )
            )
    return _table(columns, labels)


def global_efficiency(pairs: FeatureTable) -> FeatureTable:
    """Average inverse shortest path length over the network.

    Nonzero edge strength is transformed to distance as ``1 / |weight|`` and
    paths are shortest by that distance. Zero and non-finite weights are absent
    edges; disconnected pairs contribute exactly zero efficiency.

    Parameters
    ----------
    pairs : FeatureTable
        A pairwise table from :func:`envelope_correlation` or :func:`wpli`.

    Returns
    -------
    FeatureTable
        One column per band and window, with ``space="global"``.
    """
    return _graph_measure(pairs, "global_efficiency", _global_efficiency, unit="a.u.")


def clustering_coefficient(pairs: FeatureTable, *, threshold: float) -> FeatureTable:
    """Mean clustering coefficient of the binarized network.

    Edges above ``threshold`` in absolute weight are retained (:math:`A_{ij} = 1`)
    and subthreshold edges set to zero, then the unweighted clustering coefficient
    is averaged over nodes with at least two neighbors. The threshold is an explicit
    required argument.

    Parameters
    ----------
    pairs : FeatureTable
        A pairwise table from :func:`envelope_correlation` or :func:`wpli`.
    threshold : float
        Absolute weight above which an edge is kept.

    Returns
    -------
    FeatureTable
        One column per band and window, with ``space="global"``.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be finite and non-negative, got {threshold}.")
    return _graph_measure(
        pairs,
        "clustering",
        lambda m: _clustering(m, threshold),
        unit=f"a.u. (>{threshold})",
        threshold=threshold,
    )


def _require_mne_connectivity() -> Callable[..., Any]:
    try:
        from mne_connectivity import spectral_connectivity_epochs
    except ImportError as exc:  # pragma: no cover - exercised by the import guard test
        raise ImportError(
            "wpli needs mne-connectivity, which is not installed. "
            "Install it with: pip install eegfeat[connectivity]"
        ) from exc
    estimator: Callable[..., Any] = spectral_connectivity_epochs
    return estimator


def _resolve_rows(
    trials: Sequence[str] | npt.NDArray[np.str_] | None, n_epochs: int
) -> tuple[npt.NDArray[np.int_], tuple[str, ...]]:
    if trials is None:
        return np.zeros(n_epochs, dtype=int), ("all",)
    labels = np.asarray(trials)
    if labels.shape != (n_epochs,):
        raise ValueError(
            f"trials must have one label per epoch; got {labels.shape} for {n_epochs} epochs."
        )
    unique = tuple(str(value) for value in sorted(set(labels.tolist())))
    index = {name: position for position, name in enumerate(unique)}
    return np.array([index[str(value)] for value in labels], dtype=int), unique


def _nodes(
    ch_names: tuple[str, ...], groups: Mapping[str, Sequence[str]] | None
) -> tuple[tuple[str, ...], list[list[int]]]:
    if groups is None:
        return ch_names, [[i] for i in range(len(ch_names))]
    lookup = {name: i for i, name in enumerate(ch_names)}
    picks = []
    for roi, members in groups.items():
        missing = [m for m in members if m not in lookup]
        if missing:
            raise KeyError(f"group {roi!r} names unknown channels: {missing}")
        if not members:
            raise ValueError(f"group {roi!r} has no channels.")
        picks.append([lookup[m] for m in members])
    return tuple(groups), picks


def _correlation_matrix(
    envelope: npt.NDArray[np.float64], picks: list[list[int]]
) -> npt.NDArray[np.float64]:
    # One series per node: a channel, or an ROI's mean envelope.
    series = np.stack([envelope[:, pick, :].mean(axis=1) for pick in picks], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_trial = np.stack([np.atleast_2d(np.corrcoef(trial)) for trial in series])
        matrix = np.nanmean(per_trial, axis=0)
    return np.asarray(np.atleast_2d(matrix), dtype=float)


def _reduce_to_nodes(
    matrix: npt.NDArray[np.float64], picks: list[list[int]]
) -> npt.NDArray[np.float64]:
    n = len(picks)
    out = np.full((n, n), np.nan)
    for i, left in enumerate(picks):
        for j, right in enumerate(picks):
            block = matrix[np.ix_(left, right)]
            if i == j:
                # Within a node, the diagonal is self-connectivity and is not one.
                block = block[~np.eye(len(left), dtype=bool)] if len(left) > 1 else np.array([])
            values = np.asarray(block).ravel()
            values = values[np.isfinite(values)]
            out[i, j] = values.mean() if values.size else np.nan
    return out


def _dense(result: Any) -> npt.NDArray[np.float64]:
    """Square connectivity matrix from an mne-connectivity result.

    Taken from the estimator's own ``output="dense"`` rather than rebuilt from the
    flat vector: the flat form is the whole matrix raveled, not a triangle, and
    reconstructing it by hand silently permutes every pair.
    """
    dense = np.asarray(result.get_data(output="dense"), dtype=float)
    matrix = dense[:, :, 0] if dense.ndim == 3 else dense
    # Only the lower triangle is populated.
    return np.asarray(matrix + matrix.T, dtype=float)


def _pair_columns(
    matrices: npt.NDArray[np.float64],
    node_names: tuple[str, ...],
    band: Band | None,
    window: Window,
    measure: str,
    unit: str,
    signal: BandSignal | Signal,
) -> list[tuple[FeatureMeta, npt.NDArray[np.float64]]]:
    computation = ComputationSpec.create(
        measure,
        estimator=("per-trial-pearson-mean" if measure == "aec" else "mne-connectivity-wpli"),
        input_source=signal.source,
        input_computation=signal.computation.record(),
        # The node set belongs to the specification: channel-level and ROI-level
        # estimates of the same band and window are otherwise indistinguishable,
        # and a graph built from their union is a graph of neither.
        nodes=list(node_names),
    )
    columns = []
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            meta = FeatureMeta(
                measure=measure,
                band=band,
                space=f"{node_names[i]}-{node_names[j]}",
                space_kind="pair",
                window=window.name,
                normalization="raw",
                unit=unit,
                source=signal.source,
                window_bounds=(window.tmin, window.tmax),
                computation=computation,
                freq_resolution_hz=None,
                nodes=(node_names[i], node_names[j]),
            )
            columns.append((meta, matrices[:, i, j]))
    return columns


def _table(
    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]], labels: tuple[str, ...]
) -> FeatureTable:
    values = np.stack([column for _, column in columns], axis=1)
    return FeatureTable(
        values=values,
        coverage=np.isfinite(values).astype(float),
        meta=tuple(meta for meta, _ in columns),
        row_labels=labels,
    )


def _pairwise(
    signals: Sequence[BandSignal],
    *,
    measure: str,
    unit: str,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None,
    trials: Sequence[str] | npt.NDArray[np.str_] | None,
    matrix: Callable[[npt.NDArray[np.float64], list[list[int]]], npt.NDArray[np.float64]],
) -> FeatureTable:
    check_signals(signals, windows)
    row_groups, labels = _resolve_rows(trials, signals[0].n_epochs)
    node_names, picks = _nodes(signals[0].ch_names, groups)

    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for signal in signals:
        envelope = signal.envelope
        for window in windows:
            mask = window_mask(signal.times, window)
            stacked = np.stack(
                [
                    matrix(envelope[row_groups == row][:, :, mask], picks)
                    for row in range(len(labels))
                ]
            )
            columns.extend(
                _pair_columns(stacked, node_names, signal.band, window, measure, unit, signal)
            )
    return _table(columns, labels)


def _graph_measure(
    pairs: FeatureTable,
    measure: str,
    reduce: Callable[[npt.NDArray[np.float64]], float],
    *,
    unit: str,
    **parameters: object,
) -> FeatureTable:
    if any(m.space_kind != "pair" for m in pairs.meta):
        raise ValueError(
            f"{measure} needs a pairwise table from envelope_correlation or wpli; "
            "this table has columns that are not pairs."
        )
    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for indices in _by_estimator(pairs).values():
        nodes = _node_order(pairs, indices)
        values = np.empty(pairs.n_rows)
        for row in range(pairs.n_rows):
            values[row] = reduce(_square(pairs, indices, nodes, row))
        template = pairs.meta[indices[0]]
        columns.append(
            (
                replace(
                    template,
                    measure=measure,
                    space="global",
                    space_kind="global",
                    unit=unit,
                    computation=ComputationSpec.create(
                        measure,
                        input_measure=template.measure,
                        input_computation=template.computation.record(),
                        nodes=nodes,
                        **parameters,
                    ),
                    nodes=None,
                ),
                values,
            )
        )
    return _table(columns, pairs.row_labels or ("all",))


def _by_estimator(pairs: FeatureTable) -> dict[str, list[int]]:
    """Index the pair columns by everything that defines them except the pair itself.

    Grouping on band and window alone merges estimators: concatenated AEC and
    wPLI columns for one band land in a single graph, where each edge keeps
    whichever estimate was written last.
    """
    out: dict[str, list[int]] = {}
    for index, meta in enumerate(pairs.meta):
        key = replace(meta, space="", nodes=None).parameter_hash
        out.setdefault(key, []).append(index)
    return out


def _node_order(pairs: FeatureTable, indices: list[int]) -> list[str]:
    seen: list[str] = []
    for index in indices:
        nodes = pairs.meta[index].nodes
        if nodes is None:
            raise ValueError("pair metadata must carry its two node identities.")
        for name in nodes:
            if name not in seen:
                seen.append(name)
    return seen


def _square(
    pairs: FeatureTable, indices: list[int], nodes: list[str], row: int
) -> npt.NDArray[np.float64]:
    position = {name: i for i, name in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    filled: set[frozenset[str]] = set()
    for index in indices:
        nodes_pair = pairs.meta[index].nodes
        if nodes_pair is None:
            raise ValueError("pair metadata must carry its two node identities.")
        left, right = nodes_pair
        edge = frozenset(nodes_pair)
        if edge in filled:
            raise ValueError(
                f"edge {left}-{right} is measured twice within one estimator; a graph "
                "cannot take two values for the same pair."
            )
        filled.add(edge)
        value = pairs.values[row, index]
        matrix[position[left], position[right]] = value
        matrix[position[right], position[left]] = value
    return matrix


def _global_efficiency(matrix: npt.NDArray[np.float64]) -> float:
    n = matrix.shape[0]
    if n <= 1:
        return float("nan")
    weights = np.abs(np.nan_to_num(matrix, nan=0.0))
    length = np.full_like(weights, np.inf)
    present = weights > 0.0
    length[present] = 1.0 / weights[present]
    np.fill_diagonal(length, 0.0)
    # Floyd-Warshall: exact all-pairs shortest paths, and n is the node count.
    distance = length.copy()
    for k in range(n):
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    upper = np.triu_indices(n, k=1)
    pair_distances = distance[upper]
    efficiencies = np.where(
        np.isfinite(pair_distances) & (pair_distances > 0.0),
        1.0 / pair_distances,
        0.0,
    )
    return float(efficiencies.mean())


def _clustering(matrix: npt.NDArray[np.float64], threshold: float) -> float:
    adjacency = (np.abs(np.nan_to_num(matrix, nan=0.0)) > threshold).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    triangles = np.diag(adjacency @ adjacency @ adjacency)
    eligible = degree >= 2
    if not eligible.any():
        return float("nan")
    coefficients = triangles[eligible] / (degree[eligible] * (degree[eligible] - 1.0))
    return float(coefficients.mean())
