from dataclasses import replace

import numpy as np
import pytest

import eegfeat as ef
from eegfeat.bands import Band
from eegfeat.connectivity import (
    _clustering,
    _global_efficiency,
    clustering_coefficient,
    envelope_correlation,
    global_efficiency,
)
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window

SFREQ = 100.0
ALPHA = Band("alpha", 8.0, 13.0)
WINDOW = Window("all", 0.0, 4.0)
CHANNELS = ("C3", "C4", "P3", "P4")


def _signal(envelope: np.ndarray) -> BandSignal:
    return BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=np.arange(envelope.shape[-1]) / SFREQ,
        ch_names=CHANNELS,
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(envelope.shape[0])),
    )


def _shared_driver(strength: float, n_epochs: int = 8, n_times: int = 401) -> BandSignal:
    rng = np.random.RandomState(0)
    driver = rng.rand(n_epochs, 1, n_times)
    envelope = rng.rand(n_epochs, 4, n_times)
    envelope[:, :2, :] += strength * driver
    return _signal(envelope)


def test_channels_sharing_a_driver_correlate_and_others_do_not() -> None:
    table = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    values = dict(zip([m.space for m in table.meta], table.values[0], strict=True))
    assert values["C3-C4"] > 0.8
    assert abs(values["P3-P4"]) < 0.3


def test_the_coupled_pair_is_the_one_labelled_as_coupled() -> None:
    # Pair placement, not pair values: a matrix reconstructed with the wrong index
    # order still produces the right set of numbers against the wrong labels.
    table = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    values = dict(zip([m.space for m in table.meta], table.values[0], strict=True))
    assert max(values, key=lambda name: values[name]) == "C3-C4"


def test_every_unordered_pair_appears_once() -> None:
    table = envelope_correlation([_shared_driver(1.0)], windows=[WINDOW])
    spaces = [m.space for m in table.meta]
    assert len(spaces) == 6  # 4 nodes -> 4*3/2
    assert len(set(spaces)) == 6
    assert all(m.space_kind == "pair" for m in table.meta)


def test_the_result_has_one_row_per_trial_group() -> None:
    signal = _shared_driver(5.0)
    table = envelope_correlation([signal], windows=[WINDOW])
    assert table.row_labels == ("all",)
    grouped = envelope_correlation([signal], windows=[WINDOW], trials=["a"] * 4 + ["b"] * 4)
    assert grouped.row_labels == ("a", "b")
    assert grouped.values.shape[0] == 2


def test_it_cannot_be_joined_to_per_epoch_features() -> None:
    signal = _shared_driver(1.0)
    pairs = envelope_correlation([signal], windows=[WINDOW])
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    with pytest.raises(ValueError, match="row semantics"):
        ef.concat([pairs, per_epoch])


def test_groups_make_rois_the_nodes() -> None:
    table = envelope_correlation(
        [_shared_driver(1.0)],
        windows=[WINDOW],
        groups={"front": ["C3", "C4"], "back": ["P3", "P4"]},
    )
    assert [m.space for m in table.meta] == ["front-back"]


def test_an_unknown_channel_in_a_group_raises() -> None:
    with pytest.raises(KeyError, match="Fz"):
        envelope_correlation([_shared_driver(1.0)], windows=[WINDOW], groups={"front": ["Fz"]})


def test_correlation_is_bounded() -> None:
    table = envelope_correlation([_shared_driver(3.0)], windows=[WINDOW])
    assert np.all(np.abs(table.values) <= 1.0 + 1e-12)


def test_trial_correlations_are_averaged_instead_of_pooling_samples() -> None:
    x = np.linspace(0.0, 1.0, 401)
    envelope = np.ones((2, 4, x.size))
    envelope[0, 0] = 10.0 + x
    envelope[0, 1] = 10.0 + x
    envelope[1, 0] = 1.0 + x
    envelope[1, 1] = 2.0 - x

    table = envelope_correlation([_signal(envelope)], windows=[WINDOW])
    values = dict(zip([meta.space for meta in table.meta], table.values[0], strict=True))

    assert values["C3-C4"] == pytest.approx(0.0, abs=1e-12)


# --- graph measures -------------------------------------------------------------------


def test_global_efficiency_of_a_fully_connected_unit_graph() -> None:
    # Every edge has weight 1, so every shortest path is one step of length ~1.
    matrix = np.ones((4, 4))
    np.fill_diagonal(matrix, 0.0)
    assert _global_efficiency(matrix) == pytest.approx(1.0, rel=1e-6)


def test_global_efficiency_rises_with_connection_strength() -> None:
    weak = np.full((4, 4), 0.1)
    strong = np.full((4, 4), 0.9)
    np.fill_diagonal(weak, 0.0)
    np.fill_diagonal(strong, 0.0)
    assert _global_efficiency(strong) > _global_efficiency(weak)


def test_global_efficiency_uses_paths_not_direct_edges() -> None:
    # A chain A-B-C: A to C has no direct edge but a two-step path exists.
    chain = np.zeros((3, 3))
    chain[0, 1] = chain[1, 0] = 1.0
    chain[1, 2] = chain[2, 1] = 1.0
    complete = np.ones((3, 3))
    np.fill_diagonal(complete, 0.0)
    assert 0.0 < _global_efficiency(chain) < _global_efficiency(complete)


def test_disconnected_node_pairs_contribute_exactly_zero_efficiency() -> None:
    disconnected = np.zeros((2, 2))
    assert _global_efficiency(disconnected) == 0.0


def test_clustering_of_a_triangle_is_one_and_of_a_chain_is_zero() -> None:
    triangle = np.ones((3, 3))
    np.fill_diagonal(triangle, 0.0)
    assert _clustering(triangle, 0.5) == pytest.approx(1.0)
    chain = np.zeros((3, 3))
    chain[0, 1] = chain[1, 0] = chain[1, 2] = chain[2, 1] = 1.0
    assert _clustering(chain, 0.5) == pytest.approx(0.0)


def test_clustering_is_nan_when_no_node_has_two_neighbours() -> None:
    sparse = np.zeros((4, 4))
    sparse[0, 1] = sparse[1, 0] = 1.0
    assert np.isnan(_clustering(sparse, 0.5))


def test_clustering_average_excludes_nodes_with_fewer_than_two_neighbours() -> None:
    triangle_and_isolate = np.zeros((4, 4))
    triangle_and_isolate[:3, :3] = 1.0
    np.fill_diagonal(triangle_and_isolate, 0.0)
    assert _clustering(triangle_and_isolate, 0.5) == pytest.approx(1.0)


def test_hyphenated_node_names_are_not_parsed_from_display_strings() -> None:
    envelope = np.random.RandomState(21).rand(4, 2, 401)
    signal = BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=np.arange(401) / SFREQ,
        ch_names=("EEG-C3", "EEG-C4"),
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(4)),
    )
    pairs = envelope_correlation([signal], windows=[WINDOW])
    assert pairs.meta[0].nodes == ("EEG-C3", "EEG-C4")
    assert np.isfinite(global_efficiency(pairs).values).all()


def test_graph_measures_reduce_a_pair_table_to_one_global_column() -> None:
    pairs = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    for table in (global_efficiency(pairs), clustering_coefficient(pairs, threshold=0.2)):
        assert table.values.shape == (1, 1)
        assert table.meta[0].space == "global"
        assert table.meta[0].space_kind == "global"
        assert table.row_labels == pairs.row_labels


def test_the_clustering_threshold_is_recorded() -> None:
    pairs = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    assert "0.2" in clustering_coefficient(pairs, threshold=0.2).meta[0].unit


def test_negative_clustering_threshold_raises() -> None:
    pairs = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    with pytest.raises(ValueError, match="non-negative"):
        clustering_coefficient(pairs, threshold=-0.1)


def test_graph_measures_refuse_a_non_pairwise_table() -> None:
    signal = _shared_driver(1.0)
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    with pytest.raises(ValueError, match="pairwise"):
        global_efficiency(per_epoch)


def test_wpli_reports_its_missing_dependency_clearly() -> None:
    import importlib.util

    if importlib.util.find_spec("mne_connectivity") is not None:
        pytest.skip("mne-connectivity is installed here")
    from eegfeat.connectivity import wpli
    from eegfeat.signal import Signal

    signal = Signal.from_arrays(
        data=np.zeros((2, 4, 401)),
        times=np.arange(401) / SFREQ,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"), ("test", 1, "event")),
    )
    with pytest.raises(ImportError, match=r"eegfeat\[connectivity\]"):
        wpli(signal, bands=[ALPHA], windows=[WINDOW])


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("mne_connectivity") is None,
    reason="mne-connectivity is not installed in this environment",
)
def test_wpli_places_a_planted_coupling_on_the_right_pair() -> None:
    from eegfeat.connectivity import wpli
    from eegfeat.signal import Signal

    n, n_epochs = 800, 12
    times = np.arange(n) / 200.0
    rng = np.random.RandomState(0)
    data = rng.randn(n_epochs, 4, n) * 0.5
    data[:, 0, :] += np.sin(2 * np.pi * 10 * times)
    data[:, 1, :] += np.sin(2 * np.pi * 10 * times + 0.8)  # constant lag with C3
    signal = Signal.from_arrays(
        data=data,
        times=times,
        ch_names=CHANNELS,
        sfreq=200.0,
        row_ids=tuple(("test", index, "event") for index in range(n_epochs)),
    )
    table = wpli(signal, bands=[ALPHA], windows=[Window("all", 0.0, 3.995)])
    values = dict(zip([m.space for m in table.meta], table.values[0], strict=True))
    assert max(values, key=lambda name: values[name]) == "C3-C4"


def _pair_table(
    measure: str, value: float, nodes: tuple[str, ...] = CHANNELS[:3]
) -> ef.FeatureTable:
    """A pairwise table shaped like a connectivity result, with a constant edge weight."""
    metas, columns = [], []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            metas.append(
                ef.FeatureMeta(
                    measure=measure,
                    band=ALPHA,
                    space=f"{nodes[i]}-{nodes[j]}",
                    space_kind="pair",
                    window=WINDOW.name,
                    normalization="raw",
                    unit="r" if measure == "aec" else "a.u.",
                    source="hilbert",
                    window_bounds=(WINDOW.tmin, WINDOW.tmax),
                    computation=ef.ComputationSpec.create(
                        measure, estimator=measure, nodes=list(nodes)
                    ),
                    nodes=(nodes[i], nodes[j]),
                )
            )
            columns.append(np.full(1, value))
    return ef.FeatureTable(
        values=np.stack(columns, axis=1),
        coverage=np.ones((1, len(columns))),
        meta=tuple(metas),
        row_labels=("all",),
    )


@pytest.mark.parametrize("order", [("aec", "wpli"), ("wpli", "aec")])
def test_concatenated_estimators_yield_separate_graphs(order: tuple[str, str]) -> None:
    # Grouping on band and window alone collapsed these into one graph whose edges
    # came from whichever estimator was concatenated last.
    weights = {"aec": 0.8, "wpli": 0.2}
    table = ef.concat([_pair_table(name, weights[name]) for name in order])
    out = global_efficiency(table)
    assert out.values.shape == (1, 2)
    by_estimator = {
        m.computation.parameters["input_measure"]: value
        for m, value in zip(out.meta, out.values[0], strict=True)
    }
    assert by_estimator == {"aec": pytest.approx(0.8), "wpli": pytest.approx(0.2)}


@pytest.mark.parametrize("order", [("aec", "wpli"), ("wpli", "aec")])
def test_each_graph_keeps_the_provenance_of_its_own_estimator(order: tuple[str, str]) -> None:
    table = ef.concat([_pair_table(name, 0.5) for name in order])
    for meta in global_efficiency(table).meta:
        parameters = meta.computation.parameters
        assert parameters["input_computation"]["method"] == parameters["input_measure"]


def test_a_graph_measure_matches_the_one_built_from_that_estimator_alone() -> None:
    aec = _pair_table("aec", 0.8)
    mixed = global_efficiency(ef.concat([aec, _pair_table("wpli", 0.2)]))
    alone = global_efficiency(aec)
    assert alone.values.shape == (1, 1)
    np.testing.assert_allclose(mixed.values[:, 0], alone.values[:, 0])
    assert mixed.meta[0].name == alone.meta[0].name


def test_the_same_estimator_over_different_node_sets_stays_separate() -> None:
    # Channel-level and ROI-level AEC differ only in their nodes; merged, they
    # would form one graph over the union, which is a graph of neither.
    signal = _shared_driver(5.0)
    channels = envelope_correlation([signal], windows=[WINDOW])
    rois = envelope_correlation(
        [signal], windows=[WINDOW], groups={"left": ["C3", "P3"], "right": ["C4", "P4"]}
    )
    out = global_efficiency(ef.concat([channels, rois]))
    assert out.values.shape == (1, 2)
    np.testing.assert_allclose(
        sorted(out.values[0]),
        sorted([global_efficiency(rois).values[0, 0], global_efficiency(channels).values[0, 0]]),
    )


def test_one_edge_measured_twice_within_a_group_raises() -> None:
    table = _pair_table("aec", 0.8, nodes=("C3", "C4"))
    mirrored = ef.FeatureTable(
        values=table.values,
        coverage=table.coverage,
        meta=(replace(table.meta[0], space="C4-C3", nodes=("C4", "C3")),),
        row_labels=table.row_labels,
    )
    with pytest.raises(ValueError, match="measured twice"):
        global_efficiency(ef.concat([table, mirrored]))


def test_permuting_declared_nodes_does_not_create_a_second_estimator() -> None:
    first = _pair_table("aec", 0.8, nodes=("C3", "C4", "P3"))
    permuted = _pair_table("aec", 0.8, nodes=("P3", "C4", "C3"))

    with pytest.raises(ValueError, match="measured twice"):
        global_efficiency(ef.concat([first, permuted]))


def test_graph_measures_reject_an_incomplete_edge_set() -> None:
    complete = _pair_table("aec", 0.8)
    incomplete = ef.FeatureTable(
        values=complete.values[:, :-1],
        coverage=complete.coverage[:, :-1],
        meta=complete.meta[:-1],
        row_labels=complete.row_labels,
    )

    with pytest.raises(ValueError, match=r"missing.*C4-P3"):
        global_efficiency(incomplete)


def test_graph_measures_reject_self_edges() -> None:
    complete = _pair_table("aec", 0.8)
    malformed = ef.FeatureTable(
        values=complete.values,
        coverage=complete.coverage,
        meta=(replace(complete.meta[0], nodes=("C3", "C3")), *complete.meta[1:]),
        row_labels=complete.row_labels,
    )

    with pytest.raises(ValueError, match="self-edge"):
        global_efficiency(malformed)


def test_the_clustering_threshold_is_recorded_as_a_number() -> None:
    out = clustering_coefficient(_pair_table("aec", 0.8), threshold=0.5)
    assert out.meta[0].computation.parameters["threshold"] == 0.5


def test_a_single_node_set_has_no_pairs_to_report() -> None:
    # One ROI yields zero node pairs; without this the empty column list reaches numpy.
    with pytest.raises(ValueError, match="needs at least two nodes"):
        envelope_correlation(
            [_shared_driver(5.0)], windows=[WINDOW], groups={"central": ["C3", "C4"]}
        )


def test_two_node_sets_give_the_one_pair_between_them() -> None:
    table = envelope_correlation(
        [_shared_driver(5.0)],
        windows=[WINDOW],
        groups={"central": ["C3", "C4"], "parietal": ["P3", "P4"]},
    )

    assert [m.nodes for m in table.meta] == [("central", "parietal")]
