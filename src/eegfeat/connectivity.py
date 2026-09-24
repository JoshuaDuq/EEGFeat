from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from eegfeat._expand import check_signals, window_mask
from eegfeat._validation import validate_multitaper_bandwidth
from eegfeat.bands import Band, check_passband
from eegfeat.phase import _resolve_rows
from eegfeat.signal import BandSignal, Signal
from eegfeat.spectra import Window
from eegfeat.table import ComputationSpec, FeatureMeta, FeatureTable

Orthogonalization = Literal["pairwise"] | None

_FISHER_LIMIT = 0.999999
"""Clip applied before ``arctanh``, so a correlation of exactly +/-1 stays finite."""


def _aec_unit(orthogonalize: Orthogonalization, absolute: bool) -> str:
    if orthogonalize is None:
        return "r"
    return "|r| (orthogonalized)" if absolute else "r (orthogonalized)"


def envelope_correlation(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
    orthogonalize: Orthogonalization = "pairwise",
    absolute: bool = True,
) -> FeatureTable:
    """Amplitude envelope correlation between every pair of nodes.

    The correlation of band envelopes, averaged across the trials in each group
    in Fisher ``z``. Nodes are channels, or ROIs when ``groups`` is given, in
    which case a node's series is the mean analytic signal of its members.

    **On sensor data this measure is dominated by volume conduction unless it is
    orthogonalized.** One source seen by two electrodes produces a zero-lag
    envelope correlation with no interaction behind it, so ``orthogonalize``
    defaults to ``"pairwise"``, matching ``mne_connectivity.envelope_correlation``.
    Passing ``None`` gives the raw envelope correlation, which is interpretable
    on source-reconstructed or otherwise leakage-corrected data and misleading on
    sensors.

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
    orthogonalize : {"pairwise", None}, default "pairwise"
        Remove the zero-lag component before correlating, following Hipp et al.
        (2012). None correlates the envelopes directly.
    absolute : bool, default True
        Take the magnitude of each trial's correlation before averaging. Applies
        only when orthogonalizing, where the residual sign carries little
        information; matches MNE's default.

    Returns
    -------
    FeatureTable
        One column per node pair, band and window, with ``space_kind="pair"``.

    References
    ----------
    Hipp, J. F. et al. (2012). Large-scale cortical correlation structure of
    spontaneous oscillatory activity. Nature Neuroscience, 15(6), 884-890.
    """
    if orthogonalize not in ("pairwise", None):
        raise ValueError(f"orthogonalize must be 'pairwise' or None, got {orthogonalize!r}.")

    def matrix(
        analytic: npt.NDArray[np.complex128], picks: list[list[int]]
    ) -> npt.NDArray[np.float64]:
        return _correlation_matrix(analytic, picks, orthogonalize=orthogonalize, absolute=absolute)

    return _pairwise(
        signals,
        measure="aec",
        unit=_aec_unit(orthogonalize, absolute),
        windows=windows,
        groups=groups,
        trials=trials,
        matrix=matrix,
        estimator_parameters={"orthogonalize": orthogonalize, "absolute": absolute},
    )


ConnectivityMethod = Literal[
    "coh",
    "imcoh",
    "plv",
    "ciplv",
    "ppc",
    "pli",
    "wpli",
    "wpli2_debiased",
]

_METHOD_UNITS: dict[str, str] = {
    "coh": "coherence",
    "imcoh": "|imaginary coherency|",
    "plv": "plv",
    "ciplv": "ciplv",
    "ppc": "ppc",
    "pli": "pli",
    "wpli": "wpli",
    "wpli2_debiased": "wpli2 (debiased)",
}

_RECTIFIED: frozenset[str] = frozenset({"imcoh"})
"""Methods whose sign follows which node of a pair came first.

Imaginary coherency is antisymmetric -- swapping the two channels negates it --
so a table of *unordered* pairs cannot carry its sign without also carrying an
order. The magnitude is taken instead, which is the usual scalar summary and is
what makes the column well defined. Verified against the estimator rather than
assumed: ciplv looks like it should behave the same way and does not.
"""

_DIRECTED: dict[str, str] = {
    "dpli": (
        "dPLI is directional: dPLI(a, b) and dPLI(b, a) sum to one, so a table of "
        "unordered pairs would report one direction and imply the other. Symmetrizing "
        "it is a silent error rather than an approximation."
    ),
    "cohy": (
        "Coherency is complex, and a FeatureTable column is real. Use 'coh' for its "
        "magnitude or 'imcoh' for its imaginary part."
    ),
}


def spectral_connectivity(
    signal: Signal,
    *,
    method: ConnectivityMethod,
    bands: Sequence[Band],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
    mode: Literal["multitaper", "fourier"] = "multitaper",
    bandwidth: float = 2.0,
) -> FeatureTable:
    """Spectral connectivity between every pair of nodes.

    Delegates the estimation to ``mne_connectivity.spectral_connectivity_epochs``,
    so the cross-spectral density is computed once, in one place, for every method.

    Takes a broadband :class:`~eegfeat.Signal` and a list of bands, rather than
    pre-filtered :class:`~eegfeat.BandSignal` objects, because the band is a
    parameter of the spectral estimation rather than a prior filtering step.

    Bands are reduced on this side with :meth:`~eegfeat.Band.mask`, not by the
    estimator's ``faverage``, so the half-open convention holds: mne-connectivity
    treats ``[fmin, fmax]`` as closed and would put a shared edge bin in both of
    two adjacent bands.

    Requires the optional dependency: ``pip install eegfeat[connectivity]``.

    Parameters
    ----------
    signal : Signal
        Broadband epochs.
    method : str
        One of ``"coh"``, ``"imcoh"``, ``"plv"``, ``"ciplv"``, ``"ppc"``,
        ``"pli"``, ``"wpli"``, ``"wpli2_debiased"``. Prefer ``"wpli2_debiased"``
        over ``"wpli"`` at low trial counts: it removes the sample-size bias that
        makes wPLI rise as trials fall.

        ``"imcoh"`` is reported as a magnitude, because its sign is a statement
        about which node came first and these pairs are unordered. ``"dpli"`` and
        ``"cohy"`` are refused rather than misrepresented; see the error each
        raises.
    bands : sequence of Band
        Bands to estimate in.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. Channel-level connectivity is averaged
        within each ROI block.
    trials : sequence of str, optional
        A label per epoch. One row per distinct label.
    mode : {"multitaper", "fourier"}, default "multitaper"
        Spectral estimator passed to mne-connectivity.
    bandwidth : float, default 2.0
        Frequency smoothing of the multitaper estimate, in Hz; ignored by
        ``"fourier"``. Fixed in hertz on purpose: mne-connectivity's own default
        is ``8 / window_length`` Hz, which on a 1 s window smooths over ±4 Hz,
        wider than the delta or theta band, and changes with every window
        length. Must be at least 1.35 frequency bins, ``1.35 * sfreq / n_samples``:
        in a narrower band no Slepian taper keeps 90% of its power.

    Returns
    -------
    FeatureTable
        One column per node pair, band and window, with ``space_kind="pair"``.
    """
    if method in _DIRECTED:
        raise ValueError(f"{method!r} is not available here. {_DIRECTED[method]}")
    if method not in _METHOD_UNITS:
        raise ValueError(
            f"unknown connectivity method {method!r}; expected one of {sorted(_METHOD_UNITS)}."
        )
    if mode not in ("multitaper", "fourier"):
        raise ValueError(f"mode must be 'multitaper' or 'fourier', got {mode!r}.")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be finite and positive, got {bandwidth}.")

    row_groups, labels = _resolve_rows(trials, signal.data.shape[0])
    if np.any(np.bincount(row_groups, minlength=len(labels)) < 2):
        raise ValueError(
            f"{method} requires at least two epochs in every trial group: these "
            "estimators average a cross-spectrum over epochs."
        )
    estimator = _require_mne_connectivity()
    node_names, picks = _nodes(signal.ch_names, groups)

    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for band in bands:
        if signal.passband is not None:
            check_passband(band, *signal.passband, source=method)
        for window in windows:
            mask = window_mask(signal.times, window)
            if mode == "multitaper":
                validate_multitaper_bandwidth(
                    bandwidth, signal.sfreq, int(mask.sum()), window.name, "bandwidth"
                )
            matrices = []
            for row in range(len(labels)):
                data = signal.data[row_groups == row][:, :, mask]
                result = estimator(
                    data,
                    method=method,
                    sfreq=signal.sfreq,
                    fmin=band.fmin,
                    fmax=band.fmax,
                    mode=mode,
                    mt_bandwidth=bandwidth if mode == "multitaper" else None,
                    faverage=False,
                    verbose=False,
                )
                matrices.append(_band_mean(result, band, rectify=method in _RECTIFIED))
            columns.extend(
                _pair_columns(
                    np.stack([_reduce_to_nodes(m, picks) for m in matrices]),
                    node_names,
                    band,
                    window,
                    method,
                    _METHOD_UNITS[method],
                    signal,
                    {
                        "mode": mode,
                        "bandwidth_hz": bandwidth if mode == "multitaper" else None,
                        "band_edges": "half_open",
                        "frequency_reduction": "mean",
                        "rectified": method in _RECTIFIED,
                    },
                )
            )
    return _table(columns, labels)


def wpli(
    signal: Signal,
    *,
    bands: Sequence[Band],
    windows: Sequence[Window],
    groups: Mapping[str, Sequence[str]] | None = None,
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
    bandwidth: float = 2.0,
) -> FeatureTable:
    """Weighted phase lag index between every pair of nodes.

    A shorthand for :func:`spectral_connectivity` with ``method="wpli"``; see it
    for what the estimation does and for the other measures it reaches. At low
    trial counts prefer ``method="wpli2_debiased"``, which corrects the
    sample-size bias that makes wPLI grow as the number of trials falls.

    Parameters
    ----------
    signal : Signal
        Broadband epochs.
    bands : sequence of Band
        Bands to estimate in.
    windows : sequence of Window
        Analysis windows.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels.
    trials : sequence of str, optional
        A label per epoch. One row per distinct label.
    bandwidth : float, default 2.0
        Multitaper frequency smoothing in Hz; see :func:`spectral_connectivity`.

    Returns
    -------
    FeatureTable
        One column per node pair, band and window, with ``space_kind="pair"``.
    """
    return spectral_connectivity(
        signal,
        method="wpli",
        bands=bands,
        windows=windows,
        groups=groups,
        trials=trials,
        bandwidth=bandwidth,
    )


def global_efficiency(pairs: FeatureTable) -> FeatureTable:
    """Average inverse shortest path length over the network.

    Nonzero edge strength is transformed to distance as ``1 / |weight|`` and
    paths are shortest by that distance. Zero weights are absent edges;
    disconnected pairs contribute exactly zero efficiency. A non-finite edge
    makes the graph summary undefined, since missing is not disconnected.

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
    required argument. A non-finite edge makes the graph summary undefined.

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


def _nodes(
    ch_names: tuple[str, ...], groups: Mapping[str, Sequence[str]] | None
) -> tuple[tuple[str, ...], list[list[int]]]:
    if groups is None:
        return _checked(ch_names, [[i] for i in range(len(ch_names))])
    lookup = {name: i for i, name in enumerate(ch_names)}
    picks = []
    for roi, members in groups.items():
        missing = [m for m in members if m not in lookup]
        if missing:
            raise KeyError(f"group {roi!r} names unknown channels: {missing}")
        if not members:
            raise ValueError(f"group {roi!r} has no channels.")
        picks.append([lookup[m] for m in members])
    return _checked(tuple(groups), picks)


def _checked(
    node_names: tuple[str, ...], picks: list[list[int]]
) -> tuple[tuple[str, ...], list[list[int]]]:
    # One node is zero pairs, and a table of no columns says nothing about why.
    if len(node_names) < 2:
        raise ValueError(
            f"a pairwise measure needs at least two nodes, got {list(node_names)}; "
            "name more channels or more ROIs."
        )
    return node_names, picks


def _correlation_matrix(
    analytic: npt.NDArray[np.complex128],
    picks: list[list[int]],
    *,
    orthogonalize: Orthogonalization,
    absolute: bool,
) -> npt.NDArray[np.float64]:
    # One series per node: a channel, or the mean analytic signal of an ROI. The
    # analytic signal rather than the envelope, because orthogonalization needs phase.
    series = np.stack([analytic[:, pick, :].mean(axis=1) for pick in picks], axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        per_trial = np.stack(
            [
                _trial_correlation(trial, orthogonalize=orthogonalize, absolute=absolute)
                for trial in series
            ]
        )
        # Fisher z, not a plain mean: r is not additive, and a correlation averaged
        # on its own scale is biased toward zero. The same reasoning as
        # eegfeat.model.aggregate.subject_level_r, and the same clip to keep
        # arctanh finite at exactly +/-1.
        bounded = np.clip(per_trial, -_FISHER_LIMIT, _FISHER_LIMIT)
        matrix = np.tanh(np.nanmean(np.arctanh(bounded), axis=0))
    return np.asarray(np.atleast_2d(matrix), dtype=float)


def _trial_correlation(
    trial: npt.NDArray[np.complex128],
    *,
    orthogonalize: Orthogonalization,
    absolute: bool,
) -> npt.NDArray[np.float64]:
    """Envelope correlation matrix for one trial.

    The orthogonalized branch follows Hipp et al. (2012) as implemented by
    ``mne_connectivity.envelope_correlation``: node ``i`` is projected onto the
    plane orthogonal to node ``j`` before its envelope is correlated with ``j``'s,
    which removes the zero-lag component that volume conduction produces. The
    result is asymmetric, so it is averaged with its transpose.
    """
    magnitude = np.abs(trial)
    if orthogonalize is None:
        return np.asarray(np.atleast_2d(np.corrcoef(magnitude)), dtype=float)

    conjugate_scaled = np.conj(trial) / magnitude
    centred = magnitude - magnitude.mean(axis=-1, keepdims=True)
    spread = np.linalg.norm(centred, axis=-1)
    spread = np.where(spread == 0.0, 1.0, spread)

    n_nodes = trial.shape[0]
    corr = np.empty((n_nodes, n_nodes), dtype=float)
    for node in range(n_nodes):
        orth = np.abs((trial[node] * conjugate_scaled).imag)
        orth[node] = 1.0  # self-projection is degenerate; zeroed by the centring below
        orth = orth - orth.mean(axis=-1, keepdims=True)
        orth_spread = np.linalg.norm(orth, axis=-1)
        orth_spread = np.where(orth_spread == 0.0, 1.0, orth_spread)
        corr[node] = (orth * centred).sum(axis=-1) / spread / orth_spread
    if absolute:
        corr = np.abs(corr)
    return np.asarray((corr.T + corr) / 2.0, dtype=float)


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


def _band_mean(result: Any, band: Band, *, rectify: bool = False) -> npt.NDArray[np.float64]:
    """Square connectivity matrix averaged over the band's half-open frequencies.

    Taken from the estimator's own ``output="dense"`` rather than rebuilt from the
    flat vector: the flat form is the whole matrix raveled, not a triangle, and
    reconstructing it by hand silently permutes every pair.
    """
    dense = np.asarray(result.get_data(output="dense"), dtype=float)
    if dense.ndim != 3:
        raise ValueError(f"expected a (nodes, nodes, freqs) connectivity result, got {dense.shape}")
    freqs = np.asarray(result.freqs, dtype=float)
    keep = band.mask(freqs)
    if not keep.any():
        raise ValueError(
            f"band {band.name!r} [{band.fmin}, {band.fmax}) contains none of the "
            f"frequencies the estimator returned ({freqs.min()} to {freqs.max()} Hz). "
            "Use a longer window or a wider band."
        )
    selected = np.abs(dense[:, :, keep]) if rectify else dense[:, :, keep]
    # Rectify before averaging: a sign that flips across the band would otherwise
    # cancel to nearly nothing and read as an absence of coupling.
    matrix = selected.mean(axis=2)
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
    estimator_parameters: Mapping[str, object],
) -> list[tuple[FeatureMeta, npt.NDArray[np.float64]]]:
    computation = ComputationSpec.create(
        measure,
        estimator=("per-trial-fisher-z-mean" if measure == "aec" else "mne-connectivity"),
        # Orthogonalization and band-edge handling change what is measured, not
        # merely how precisely; two columns that differ in them are different features.
        estimator_parameters=dict(estimator_parameters),
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
    matrix: Callable[[npt.NDArray[np.complex128], list[list[int]]], npt.NDArray[np.float64]],
    estimator_parameters: Mapping[str, object],
) -> FeatureTable:
    check_signals(signals, windows)
    row_groups, labels = _resolve_rows(trials, signals[0].n_epochs)
    node_names, picks = _nodes(signals[0].ch_names, groups)

    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for signal in signals:
        analytic = signal.analytic
        for window in windows:
            mask = window_mask(signal.times, window)
            stacked = np.stack(
                [
                    matrix(analytic[row_groups == row][:, :, mask], picks)
                    for row in range(len(labels))
                ]
            )
            columns.extend(
                _pair_columns(
                    stacked,
                    node_names,
                    signal.band,
                    window,
                    measure,
                    unit,
                    signal,
                    estimator_parameters,
                )
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
        nodes = _validate_graph_edges(pairs, indices)
        values = np.empty(pairs.n_rows)
        for row in range(pairs.n_rows):
            matrix = _square(pairs, indices, nodes, row)
            values[row] = reduce(matrix) if np.isfinite(matrix).all() else np.nan
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
        parameters = dict(meta.computation.parameters)
        declared_nodes = parameters.get("nodes")
        if isinstance(declared_nodes, list) and all(
            isinstance(node, str) for node in declared_nodes
        ):
            parameters["nodes"] = sorted(declared_nodes)
        canonical_computation = ComputationSpec.create(meta.computation.method, **parameters)
        key = replace(
            meta,
            space="",
            nodes=None,
            computation=canonical_computation,
        ).parameter_hash
        out.setdefault(key, []).append(index)
    return out


def _validate_graph_edges(pairs: FeatureTable, indices: list[int]) -> list[str]:
    template = pairs.meta[indices[0]]
    declared_nodes = template.computation.parameters.get("nodes")
    if not isinstance(declared_nodes, list) or not declared_nodes:
        raise ValueError("pair computation metadata must declare its node set.")
    if any(not isinstance(node, str) or not node for node in declared_nodes):
        raise ValueError("pair computation nodes must be non-empty strings.")
    nodes = list(declared_nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("pair computation nodes must be unique.")

    expected = {
        frozenset((nodes[left], nodes[right]))
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    }
    observed: set[frozenset[str]] = set()
    for index in indices:
        edge_nodes = pairs.meta[index].nodes
        if edge_nodes is None or len(edge_nodes) != 2:
            raise ValueError("pair metadata must carry its two node identities.")
        left, right = edge_nodes
        if left == right:
            raise ValueError(f"self-edge {left}-{right} is not valid graph input.")
        if left not in nodes or right not in nodes:
            raise ValueError(f"edge {left}-{right} contains a node outside the declared node set.")
        edge = frozenset(edge_nodes)
        if edge in observed:
            raise ValueError(
                f"edge {left}-{right} is measured twice within one estimator; a graph "
                "cannot take two values for the same pair."
            )
        observed.add(edge)

    missing = expected - observed
    if missing:
        labels = ["-".join(node for node in nodes if node in edge) for edge in missing]
        raise ValueError(f"graph edge set is incomplete; missing: {', '.join(sorted(labels))}.")
    return nodes


def _square(
    pairs: FeatureTable, indices: list[int], nodes: list[str], row: int
) -> npt.NDArray[np.float64]:
    position = {name: i for i, name in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    for index in indices:
        nodes_pair = pairs.meta[index].nodes
        assert nodes_pair is not None
        left, right = nodes_pair
        value = pairs.values[row, index]
        matrix[position[left], position[right]] = value
        matrix[position[right], position[left]] = value
    return matrix


def _global_efficiency(matrix: npt.NDArray[np.float64]) -> float:
    n = matrix.shape[0]
    if n <= 1:
        return float("nan")
    weights = np.abs(matrix)
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
    adjacency = (np.abs(matrix) > threshold).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    triangles = np.diag(adjacency @ adjacency @ adjacency)
    eligible = degree >= 2
    if not eligible.any():
        return float("nan")
    coefficients = triangles[eligible] / (degree[eligible] * (degree[eligible] - 1.0))
    return float(coefficients.mean())
