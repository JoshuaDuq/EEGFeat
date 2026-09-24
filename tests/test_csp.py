"""Common spatial patterns, and the leak cross-fitting exists to prevent."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import eegfeat as ef
from eegfeat.spectra import Window

SFREQ = 250.0
N_TIMES = int(2 * SFREQ)
N_CHANNELS = 8
_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _signal(data: np.ndarray, band: ef.Band | None = None) -> ef.Signal:
    return ef.Signal.from_arrays(
        data=data,
        times=np.arange(data.shape[-1]) / SFREQ,
        ch_names=tuple(f"E{i}" for i in range(data.shape[1])),
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(data.shape[0])),
    )


def _lateralised(n_per_class: int = 40, seed: int = 0) -> tuple[ef.Signal, np.ndarray]:
    """Class 0 loads source 0, class 1 loads source 1, both mixed to every sensor."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(N_CHANNELS, N_CHANNELS))
    epochs, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            sources = rng.normal(size=(N_CHANNELS, N_TIMES))
            sources[0] *= 3.0 if label == 0 else 1.0
            sources[1] *= 1.0 if label == 0 else 3.0
            epochs.append(mixing @ sources)
            labels.append(label)
    return _signal(np.stack(epochs)), np.asarray(labels)


def _folds(n: int, k: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.arange(n)
    return [(np.setdiff1d(order, part), part) for part in np.array_split(order, k)]


# --- the estimator --------------------------------------------------------------------


def test_the_leading_components_separate_the_classes() -> None:
    signal, y = _lateralised()
    features = ef.CommonSpatialPattern.fit(signal, y, n_components=4).transform(signal)
    # Components alternate ends of the spectrum, so the first favours class 0 and
    # the second class 1; a filter that did neither would leave these means equal.
    assert features[y == 0, 0].mean() > features[y == 1, 0].mean() + 1.0
    assert features[y == 1, 1].mean() > features[y == 0, 1].mean() + 1.0


def test_features_do_not_depend_on_overall_amplitude() -> None:
    # Trace normalization: one loud epoch must not decide the filters, and a
    # rescaled recording must not move the features.
    signal, y = _lateralised()
    loud = _signal(signal.data * 1e6)
    quiet = ef.CommonSpatialPattern.fit(signal, y, n_components=4).transform(signal)
    scaled = ef.CommonSpatialPattern.fit(loud, y, n_components=4).transform(loud)
    np.testing.assert_allclose(np.abs(quiet), np.abs(scaled), rtol=1e-6)


def test_patterns_and_filters_are_kept_apart() -> None:
    signal, y = _lateralised()
    fitted = ef.CommonSpatialPattern.fit(signal, y, n_components=4)
    assert fitted.filters.shape == (4, N_CHANNELS)
    assert fitted.patterns.shape == (4, N_CHANNELS)
    assert fitted.n_components == 4
    assert fitted.classes == (0, 1)
    # A filter is not a pattern; conflating them is the classic topography error.
    assert not np.allclose(fitted.filters, fitted.patterns)


@pytest.mark.parametrize("n_components", [0, 1, 3, 5, -2, 2.0, True])
def test_an_unusable_component_count_raises(n_components) -> None:
    signal, y = _lateralised(n_per_class=6)
    with pytest.raises(ValueError, match="n_components"):
        ef.CommonSpatialPattern.fit(signal, y, n_components=n_components)


def test_more_components_than_channels_raises() -> None:
    signal, y = _lateralised(n_per_class=6)
    with pytest.raises(ValueError, match="exceeds"):
        ef.CommonSpatialPattern.fit(signal, y, n_components=N_CHANNELS + 2)


def test_a_single_class_raises() -> None:
    signal, y = _lateralised(n_per_class=6)
    with pytest.raises(ValueError, match="exactly two classes"):
        ef.CommonSpatialPattern.fit(signal, np.zeros_like(y))


def test_three_classes_raise() -> None:
    signal, y = _lateralised(n_per_class=6)
    y = y.copy()
    y[:3] = 2
    with pytest.raises(ValueError, match="exactly two classes"):
        ef.CommonSpatialPattern.fit(signal, y)


def test_transforming_different_channels_raises() -> None:
    signal, y = _lateralised(n_per_class=6)
    fitted = ef.CommonSpatialPattern.fit(signal, y)
    other = ef.Signal.from_arrays(
        data=signal.data[:, ::-1, :],
        times=signal.times,
        ch_names=tuple(reversed(signal.ch_names)),
        sfreq=SFREQ,
        row_ids=signal.row_ids,
    )
    with pytest.raises(ValueError, match="different channels"):
        fitted.transform(other)


def test_regularization_rescues_more_channels_than_epochs() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(6, N_CHANNELS, N_TIMES))
    signal = _signal(data)
    y = np.array([0, 0, 0, 1, 1, 1])
    fitted = ef.CommonSpatialPattern.fit(signal, y, n_components=4, regularization=0.3)
    assert np.isfinite(fitted.filters).all()


@pytest.mark.parametrize("regularization", [-0.1, 1.0, 1.5])
def test_an_out_of_range_regularization_raises(regularization) -> None:
    signal, y = _lateralised(n_per_class=6)
    with pytest.raises(ValueError, match="regularization"):
        ef.CommonSpatialPattern.fit(signal, y, regularization=regularization)


@pytest.mark.skipif(not _HAS_SKLEARN, reason="mne.decoding.CSP needs scikit-learn")
def test_the_informative_components_agree_with_mne() -> None:
    from mne.decoding import CSP

    signal, y = _lateralised(n_per_class=60)
    ours = ef.CommonSpatialPattern.fit(signal, y, n_components=2).transform(signal)
    reference = CSP(
        n_components=2,
        reg=None,
        log=True,
        norm_trace=True,
        cov_est="epoch",
        component_order="alternate",
        transform_into="average_power",
    ).fit_transform(signal.data, y)
    # Up to sign and scale, so compare by correlation. Only the informative
    # components: the middle eigenvalues carry no class difference and their
    # order is arbitrary in any implementation.
    for component in range(2):
        assert abs(np.corrcoef(ours[:, component], reference[:, component])[0, 1]) > 0.99


def _average_referenced(signal: ef.Signal) -> ef.Signal:
    # Every epoch now sums to zero across channels: its covariance loses one rank.
    return _signal(signal.data - signal.data.mean(axis=1, keepdims=True))


def test_an_average_reference_does_not_prevent_fitting() -> None:
    # The pooled covariance of average-referenced data is singular. Refusing it would make
    # CSP unusable on the most common EEG reference, and on ICA-cleaned data, without
    # shrinkage that invents variance the recording does not have.
    signal, y = _lateralised()
    referenced = _average_referenced(signal)
    features = ef.CommonSpatialPattern.fit(referenced, y, n_components=4).transform(referenced)
    assert features[y == 0, 0].mean() > features[y == 1, 0].mean() + 1.0
    assert features[y == 1, 1].mean() > features[y == 0, 1].mean() + 1.0


@pytest.mark.skipif(not _HAS_SKLEARN, reason="mne.decoding.CSP needs scikit-learn")
def test_on_an_average_reference_the_informative_components_agree_with_mne() -> None:
    from mne.decoding import CSP

    signal, y = _lateralised(n_per_class=60)
    referenced = _average_referenced(signal)
    ours = ef.CommonSpatialPattern.fit(referenced, y, n_components=2).transform(referenced)
    reference = CSP(
        n_components=2, reg=None, log=True, norm_trace=True, cov_est="epoch"
    ).fit_transform(referenced.data, y)
    for component in range(2):
        assert abs(np.corrcoef(ours[:, component], reference[:, component])[0, 1]) > 0.99


def test_more_components_than_the_data_rank_raises() -> None:
    signal, y = _lateralised(n_per_class=6)
    with pytest.raises(ValueError, match="rank"):
        ef.CommonSpatialPattern.fit(_average_referenced(signal), y, n_components=N_CHANNELS)


# --- cross-fitting --------------------------------------------------------------------


def test_cross_fitted_features_cover_every_row_once() -> None:
    signal, y = _lateralised(n_per_class=20)
    table = ef.csp_features(signal, y, folds=_folds(len(y)), n_components=4)
    assert table.values.shape == (len(y), 4)
    assert np.isfinite(table.values).all()
    assert table.row_ids == signal.row_ids
    assert [m.space for m in table.meta] == [f"component{i:02d}" for i in (1, 2, 3, 4)]
    assert all(m.space_kind == "global" for m in table.meta)
    assert all(m.unit == "log relative power" for m in table.meta)


def test_cross_fitting_still_separates_a_real_difference() -> None:
    signal, y = _lateralised(n_per_class=30)
    values = ef.csp_features(signal, y, folds=_folds(len(y)), n_components=4).values
    assert values[y == 0, 0].mean() > values[y == 1, 0].mean() + 0.5


@pytest.mark.skipif(not _HAS_SKLEARN, reason="needs scikit-learn")
def test_fold_local_csp_removes_the_accuracy_a_whole_dataset_fit_invents() -> None:
    # Pure noise and random labels: the honest accuracy is chance. A CSP fitted on
    # every row puts the labels into the features, and the classifier then reads
    # part of its own answer key back out.
    #
    # Averaged over several draws, not asserted on one. A single draw of this
    # inflated accuracy ranges from 0.63 to 0.89 across seeds, so a threshold
    # pinned to one seed passes by luck; the claim is about the expected leak.
    from sklearn.linear_model import LogisticRegression

    def one_draw(seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        n_epochs = 80
        signal = _signal(rng.normal(size=(n_epochs, 20, N_TIMES)))
        y = rng.integers(0, 2, size=n_epochs)
        folds = _folds(n_epochs, k=5)

        def accuracy(features: np.ndarray) -> float:
            hits = [
                LogisticRegression(max_iter=2000).fit(features[tr], y[tr]).predict(features[te])
                == y[te]
                for tr, te in folds
            ]
            return float(np.mean(np.concatenate(hits)))

        hits = []
        for train, test in folds:
            fitted = ef.CommonSpatialPattern.fit(signal, y, rows=train, n_components=6)
            model = LogisticRegression(max_iter=2000).fit(
                fitted.transform(signal, rows=train), y[train]
            )
            hits.append(model.predict(fitted.transform(signal, rows=test)) == y[test])
        return (
            accuracy(ef.CommonSpatialPattern.fit(signal, y, n_components=6).transform(signal)),
            float(np.mean(np.concatenate(hits))),
        )

    draws = np.array([one_draw(seed) for seed in range(5)])
    leaky, clean = draws[:, 0].mean(), draws[:, 1].mean()
    assert leaky > 0.65, leaky
    assert clean < 0.60, clean
    assert leaky - clean > 0.15, (leaky, clean)


def test_the_split_is_recorded_so_two_splits_are_different_features() -> None:
    signal, y = _lateralised(n_per_class=20)
    four = ef.csp_features(signal, y, folds=_folds(len(y), 4))
    five = ef.csp_features(signal, y, folds=_folds(len(y), 5))
    parameters = four.meta[0].computation.parameters
    assert parameters["cross_fitted"] is True
    assert parameters["n_folds"] == 4
    assert parameters["folds"] != five.meta[0].computation.parameters["folds"]
    assert four.names[0] != five.names[0]


def test_folds_that_leave_a_row_untested_are_refused() -> None:
    signal, y = _lateralised(n_per_class=10)
    partial = _folds(len(y))[:-1]
    with pytest.raises(ValueError, match="in no fold's test set"):
        ef.csp_features(signal, y, folds=partial)


def test_folds_that_test_a_row_twice_are_refused() -> None:
    signal, y = _lateralised(n_per_class=10)
    folds = _folds(len(y))
    with pytest.raises(ValueError, match="tested twice"):
        ef.csp_features(signal, y, folds=[*folds, folds[0]])


def test_a_fold_whose_train_and_test_overlap_is_refused() -> None:
    signal, y = _lateralised(n_per_class=10)
    order = np.arange(len(y))
    with pytest.raises(ValueError, match="overlap"):
        ef.csp_features(signal, y, folds=[(order, order)])


def test_no_folds_at_all_is_refused() -> None:
    signal, y = _lateralised(n_per_class=6)
    with pytest.raises(ValueError, match="at least one fold"):
        ef.csp_features(signal, y, folds=[])


@pytest.mark.parametrize("invalid", ["negative", "fractional", "duplicate", "boolean"])
def test_invalid_csp_fold_indices_are_refused(invalid) -> None:
    signal, y = _lateralised(n_per_class=10)
    folds = _folds(len(y))
    train, test = folds[0]
    if invalid == "negative":
        train = train.copy()
        train[0] = -len(y)  # Aliases test row zero despite disjoint integer sets.
    elif invalid == "fractional":
        train = train.astype(float) + 0.25
    elif invalid == "duplicate":
        test = np.r_[test, test[0]]
    else:
        train = np.ones(len(y), dtype=bool)
    with pytest.raises(ValueError, match="indices"):
        ef.csp_features(signal, y, folds=[(train, test), *folds[1:]])


@pytest.mark.skipif(not _HAS_SKLEARN, reason="needs scikit-learn")
def test_oof_csp_table_is_refused_as_a_classifier_design() -> None:
    from eegfeat.model import Selection, build_design

    signal, y = _lateralised(n_per_class=10)
    folds = _folds(len(y))
    train, test = folds[0]
    original = ef.csp_features(signal, y, folds=folds)
    changed_labels = y.copy()
    changed_labels[test[:2]] = 1 - changed_labels[test[:2]]
    changed = ef.csp_features(signal, changed_labels, folds=folds)
    np.testing.assert_array_equal(original.values[test], changed.values[test])
    assert not np.allclose(original.values[train], changed.values[train])

    targets = pd.DataFrame(signal.row_ids, columns=["recording", "epoch", "event"])
    targets["target"] = y
    targets["subject_id"] = np.repeat(np.arange(4), 5)
    with pytest.raises(ValueError, match="CSP.*inside.*fold"):
        build_design(original, targets, target="target")

    variance = ef.variance([signal], windows=[Window("all", 0.0, 1.9)], include_global=False)
    design = build_design(
        ef.concat([original, variance]),
        targets,
        target="target",
        selection=Selection(measure=("variance",)),
    )
    np.testing.assert_array_equal(design.X, variance.values)


def test_it_accepts_the_model_layer_own_fold_objects() -> None:
    from eegfeat.model.splits import loso_folds

    signal, y = _lateralised(n_per_class=15)
    groups = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    table = ef.csp_features(signal, y, folds=loso_folds(groups), n_components=4)
    assert np.isfinite(table.values).all()


def test_the_window_is_recorded_when_given() -> None:
    signal, y = _lateralised(n_per_class=10)
    window = Window("stimulus", 0.0, 1.5)
    table = ef.csp_features(signal, y, folds=_folds(len(y)), window=window)
    assert table.meta[0].window == "stimulus"
    assert table.meta[0].window_bounds == (0.0, 1.5)


def test_it_joins_per_epoch_tables() -> None:
    signal, y = _lateralised(n_per_class=10)
    csp = ef.csp_features(signal, y, folds=_folds(len(y)))
    variance = ef.variance([signal], windows=[Window("all", 0.0, 1.9)], include_global=False)
    joined = ef.concat([csp, variance])
    assert joined.n_rows == len(y)
    assert len(set(joined.names)) == len(joined.names)
