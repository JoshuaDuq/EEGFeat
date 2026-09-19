import numpy as np
import pytest

from eegfeat._expand import expand
from eegfeat.bands import Band
from eegfeat.spectra import Spectra, Window
from eegfeat.table import ComputationSpec

ALPHA = Band("alpha", 8.0, 13.0)
BETA = Band("beta", 13.0, 30.0)


def _spectra(n_windows: int = 2) -> Spectra:
    freqs = np.arange(8.0, 30.0, 1.0)
    data = np.ones((3, 2, n_windows, freqs.size))
    windows = tuple(
        Window(name, float(i), float(i) + 1.0)
        for i, name in enumerate(("base", "stim", "post")[:n_windows])
    )
    return Spectra(
        data=data,
        freqs=freqs,
        ch_names=("C3", "C4"),
        windows=windows,
        coverage=np.ones_like(data),
        source="test",
        representation="psd",
        support=np.ones_like(data),
        row_ids=tuple(("test", index, "event") for index in range(data.shape[0])),
        computation=ComputationSpec.create("test"),
    )


def _mean_kernel(data, freqs, weights):
    w = np.broadcast_to(weights, data.shape)
    return (data * w).sum(axis=3) / w.sum(axis=3), {}


def test_columns_are_the_product_of_bands_spaces_and_windows() -> None:
    table = expand(
        _spectra(),
        _mean_kernel,
        measure="power",
        unit="V^2/Hz",
        bands=(ALPHA, BETA),
        groups=None,
        include_global=True,
        baseline=None,
        mode="raw",
        min_bins=1,
        parameters={},
    )
    # 2 bands x (2 channels + 1 global) x 2 windows
    assert table.values.shape == (3, 12)
    assert {m.band.name for m in table.meta if m.band} == {"alpha", "beta"}
    assert {m.space for m in table.meta} == {"C3", "C4", "global"}


def test_metadata_records_the_in_band_resolution() -> None:
    table = expand(
        _spectra(),
        _mean_kernel,
        measure="power",
        unit="V^2/Hz",
        bands=(ALPHA,),
        groups=None,
        include_global=False,
        baseline=None,
        mode="raw",
        min_bins=1,
        parameters={},
    )
    assert table.meta[0].freq_resolution_hz == pytest.approx(1.0)
    assert table.meta[0].source == "test"


def test_baseline_window_is_consumed_and_not_emitted() -> None:
    table = expand(
        _spectra(),
        _mean_kernel,
        measure="power",
        unit="log10",
        bands=(ALPHA,),
        groups=None,
        include_global=False,
        baseline="base",
        mode="log_ratio",
        min_bins=1,
        parameters={},
    )
    assert {m.window for m in table.meta} == {"stim"}
    np.testing.assert_allclose(table.values, 0.0)  # flat data, so every ratio is 1


def test_normalization_precedes_aggregation() -> None:
    spectra = _spectra()
    data = spectra.data.copy()
    data[:, 0, 1, :] = 100.0  # C3 is 100x baseline in the stim window
    data[:, 1, 1, :] = 1.0  # C4 is unchanged
    spectra = Spectra(
        data=data,
        freqs=spectra.freqs,
        ch_names=spectra.ch_names,
        windows=spectra.windows,
        coverage=np.ones_like(data),
        source="test",
        representation="psd",
        support=np.ones_like(data),
        row_ids=spectra.row_ids,
        computation=spectra.computation,
    )
    table = expand(
        spectra,
        _mean_kernel,
        measure="power",
        unit="log10",
        bands=(ALPHA,),
        groups={"central": ["C3", "C4"]},
        include_global=False,
        baseline="base",
        mode="log_ratio",
        min_bins=1,
        parameters={},
    )
    # mean of log10 ratios = (2 + 0) / 2 = 1.0. log10 of mean ratio would be log10(50.5) = 1.70.
    np.testing.assert_allclose(table.values, 1.0)


def test_a_band_with_too_few_bins_raises() -> None:
    with pytest.raises(ValueError, match="3 bins"):
        expand(
            _spectra(),
            _mean_kernel,
            measure="peak_freq",
            unit="Hz",
            bands=(Band("narrow", 8.0, 10.0),),
            groups=None,
            include_global=False,
            baseline=None,
            mode="raw",
            min_bins=3,
            parameters={},
        )


def test_a_band_outside_the_frequency_axis_raises() -> None:
    with pytest.raises(ValueError, match="no frequencies"):
        expand(
            _spectra(),
            _mean_kernel,
            measure="power",
            unit="V^2/Hz",
            bands=(Band("hf", 100.0, 200.0),),
            groups=None,
            include_global=False,
            baseline=None,
            mode="raw",
            min_bins=1,
            parameters={},
        )


def test_an_unknown_baseline_window_raises() -> None:
    with pytest.raises(ValueError, match="nope"):
        expand(
            _spectra(),
            _mean_kernel,
            measure="power",
            unit="log10",
            bands=(ALPHA,),
            groups=None,
            include_global=False,
            baseline="nope",
            mode="log_ratio",
            min_bins=1,
            parameters={},
        )


def test_kernel_flags_survive_spatial_aggregation_as_any() -> None:
    def flagging_kernel(data, freqs, weights):
        values, _ = _mean_kernel(data, freqs, weights)
        flags = np.zeros(values.shape, dtype=bool)
        flags[:, 0, :] = True  # only C3
        return values, {"edge_hit": flags}

    table = expand(
        _spectra(),
        flagging_kernel,
        measure="peak_freq",
        unit="Hz",
        bands=(ALPHA,),
        groups={"central": ["C3", "C4"]},
        include_global=False,
        baseline=None,
        mode="raw",
        min_bins=1,
        parameters={},
    )
    assert table.flags["edge_hit"].all()


def test_bands_none_yields_one_broadband_column_per_space_and_window() -> None:
    table = expand(
        _spectra(),
        _mean_kernel,
        measure="slope",
        unit="a.u.",
        bands=None,
        groups=None,
        include_global=False,
        baseline=None,
        mode="raw",
        min_bins=1,
        parameters={},
    )
    assert table.values.shape == (3, 4)
    assert all(m.band is None for m in table.meta)
    assert table.names[0].startswith("eeg_slope_broadband_c3_base_raw_p")


def test_column_order_is_band_then_space_then_window() -> None:
    table = expand(
        _spectra(),
        _mean_kernel,
        measure="power",
        unit="V^2/Hz",
        bands=(ALPHA, BETA),
        groups=None,
        include_global=True,
        baseline=None,
        mode="raw",
        min_bins=1,
        parameters={},
    )
    # Order is load-bearing: the refactor must not permute columns.
    assert [name.rsplit("_p", 1)[0] for name in table.names[:3]] == [
        "eeg_power_alpha_c3_base_raw",
        "eeg_power_alpha_c3_stim_raw",
        "eeg_power_alpha_c4_base_raw",
    ]
