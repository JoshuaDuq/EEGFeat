"""The runner computes, per recording, what the recipe asks for.

Expected values come from the simulated signal, not from the library: a 10 Hz
sine locked to every epoch has a known peak, a known variance and no event-related
power change.
"""

import json

import mne
import numpy as np
import pytest

import eegfeat as ef
from eegfeat.runner import load_recipe
from eegfeat.runner.compute import compute_features
from synthetic import AMPLITUDE, NOISE, SFREQ, make_epochs

HEAD = """
[inputs]
root = "data"

[output]
root = "out"

"""


def features(tmp_path, body: str, epochs: mne.EpochsArray | None = None):
    path = tmp_path / "recipe.toml"
    path.write_text(HEAD + body)
    return compute_features(
        make_epochs() if epochs is None else epochs,
        load_recipe(path),
        recording="sub-test_task-test",
    )


def test_per_epoch_measures_give_one_row_per_epoch(tmp_path) -> None:
    result = features(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert result.epochs is not None and result.epochs.n_rows == 12
    assert result.crosstrial is None


def test_peak_frequency_finds_the_simulated_oscillation(tmp_path) -> None:
    # Smoothing is off: its default even-width kernel shifts peaks by half a bin.
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "peak_frequency"\nbands = ["alpha"]\nspatial = ["global"]\n'
        "smoothing_hz = 0.0\n",
    )

    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values, 10.0, atol=0.05)


def test_band_power_puts_the_oscillation_in_its_band(tmp_path) -> None:
    result = features(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["global"]\n'
    )

    assert result.epochs is not None
    alpha = result.epochs.select(band=ef.Band("alpha", 8.0, 13.0)).values
    theta = result.epochs.select(band=ef.Band("theta", 4.0, 8.0)).values
    assert np.all(alpha > 100 * theta)


@pytest.mark.parametrize("sfreq", [100.0, 200.0])
def test_multitaper_power_has_physical_units_independent_of_sampling_rate(tmp_path, sfreq):
    times = np.arange(int(4 * sfreq)) / sfreq
    amplitude = 2e-6
    data = np.tile(amplitude * np.sin(2 * np.pi * 10 * times), (3, 1, 1))
    epochs = mne.EpochsArray(data, mne.create_info(["Cz"], sfreq, "eeg"), verbose=False)
    result = features(
        tmp_path,
        '[spectra]\nmethod = "multitaper"\nbandwidth = 1.0\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
        epochs,
    )
    np.testing.assert_allclose(result.epochs.values, amplitude**2 / 2, rtol=0.02, atol=0)


def test_spatial_levels_choose_the_columns(tmp_path) -> None:
    result = features(
        tmp_path,
        '[rois]\nfrontal = ["Fz", "F3", "F4"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["rois", "global"]\n',
    )

    assert result.epochs is not None
    assert [m.space for m in result.epochs.meta] == ["frontal", "global"]


def test_broadband_variance_is_that_of_the_simulated_signal(tmp_path) -> None:
    result = features(tmp_path, '[[features]]\nmeasure = "variance"\nspatial = ["global"]\n')

    # A sine of amplitude A has variance A^2 / 2; the noise adds its own variance.
    expected = AMPLITUDE**2 / 2 + NOISE**2
    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values, expected, rtol=0.05)


def test_windows_are_measured_separately(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nearly = [0.0, 0.5]\nlate = [0.5, 1.0]\n\n"
        '[[features]]\nmeasure = "variance"\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.window for m in result.epochs.meta] == ["early", "late"]


def test_stationary_signal_shows_no_event_related_power_change(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.25, 1.25]\n\n"
        '[[features]]\nmeasure = "erds_mean"\nbands = ["alpha"]\nbaseline = "base"\n'
        'spatial = ["global"]\n',
    )

    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values, 0.0, atol=10.0)


def test_welch_spectra_are_computed_per_window(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.window for m in result.epochs.meta] == ["base", "stim"]
    # Both windows hold the same stationary sine, so their power agrees.
    base, stim = result.epochs.values.T
    np.testing.assert_allclose(base, stim, rtol=0.35)


def test_morlet_spectra_restrict_each_window_to_its_support(tmp_path) -> None:
    # A 1 Hz wavelet outlasts a 2 s epoch, so the grid starts at 4 Hz.
    result = features(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nfmin = 4.0\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "mean_tfr_power"\nbands = ["theta", "alpha", "beta"]\n'
        'spatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert {m.source for m in result.epochs.meta} == {"morlet"}
    alpha = result.epochs.select(band=ef.Band("alpha", 8.0, 13.0)).values
    beta = result.epochs.select(band=ef.Band("beta", 13.0, 30.0)).values
    assert np.all(alpha > 10 * beta)


def test_periodic_power_measures_the_oscillation_against_the_noise_floor(tmp_path) -> None:
    # White noise is a flat power law, which the fit takes as the floor; the 10 Hz sine
    # stands above it and the beta band does not.
    result = features(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nfmin = 4.0\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "periodic_power"\nbands = ["alpha", "beta"]\n'
        'fit_range = [4.0, 40.0]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert {m.measure for m in result.epochs.meta} == {"periodic_power"}
    alpha = result.epochs.select(band=ef.Band("alpha", 8.0, 13.0)).values
    beta = result.epochs.select(band=ef.Band("beta", 13.0, 30.0)).values
    assert np.all(alpha > 2.0 * beta)
    assert 0.5 < np.median(beta) < 2.0
    fit = json.loads(result.epochs.meta[0].computation.parameters_json)["input_computation"]
    assert fit["parameters"]["fit_range"] == [4.0, 40.0]


def test_baseline_normalized_power_consumes_the_baseline(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nbaseline = "base"\n'
        'normalize = "db"\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.window for m in result.epochs.meta] == ["stim"]
    np.testing.assert_allclose(result.epochs.values, 0.0, atol=1.5)


def test_band_ratios_and_asymmetry_follow_band_power(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["theta", "alpha"]\n'
        'ratios = [["alpha", "theta"]]\nasymmetry = [["F3", "F4"]]\n',
    )

    assert result.epochs is not None
    measures = {m.measure for m in result.epochs.meta}
    assert {"band_power", "ratio_alpha_theta", "asymmetry"} <= measures


def test_cross_trial_measures_have_one_row_per_trial_group(tmp_path) -> None:
    result = features(
        tmp_path,
        '[trials]\nby = "event"\n\n'
        '[[features]]\nmeasure = "itpc"\nbands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert result.epochs is None
    assert result.crosstrial is not None
    assert result.crosstrial.row_labels == ("left", "right")
    # The sine has the same phase on every trial.
    assert np.all(result.crosstrial.values > 0.9)


def test_trials_can_be_grouped_by_a_metadata_column(tmp_path) -> None:
    result = features(
        tmp_path,
        '[trials]\nby = "metadata"\ncolumn = "rating"\n\n'
        '[[features]]\nmeasure = "itpc"\nbands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert result.crosstrial is not None
    assert result.crosstrial.row_labels == ("0", "1", "2", "3", "4")


def test_graph_summaries_join_the_connectivity_table(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "envelope_correlation"\nbands = ["alpha"]\n'
        'graph = ["global_efficiency", "clustering_coefficient"]\nclustering_threshold = 0.1\n',
    )

    assert result.crosstrial is not None
    measures = {m.measure for m in result.crosstrial.meta}
    assert measures == {"aec", "global_efficiency", "clustering"}


def test_pac_is_computed_for_each_pair(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "pac"\npairs = [["theta", "gamma"]]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.measure for m in result.epochs.meta] == ["pac"]


def test_microstate_coverage_sums_to_one_in_every_window(tmp_path) -> None:
    pytest.importorskip("sklearn")
    result = features(
        tmp_path,
        '[microstates]\nn_states = 3\n\n[[features]]\nmeasure = "microstate_coverage"\n',
    )

    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values.sum(axis=1), 1.0)


def test_recipe_parameters_reach_the_library(tmp_path) -> None:
    # Channel columns: the global column averages channels after the log, not before.
    raw = features(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["channels"]\n'
    )
    logged = features(
        tmp_path,
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'spatial = ["channels"]\nnormalize = "log10"\n',
    )

    assert raw.epochs is not None and logged.epochs is not None
    np.testing.assert_allclose(logged.epochs.values, np.log10(raw.epochs.values))


def test_window_outside_the_epoch_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="late"):
        features(tmp_path, '[windows]\nlate = [1.0, 3.0]\n\n[[features]]\nmeasure = "variance"\n')


def test_roi_naming_an_absent_channel_is_an_error(tmp_path) -> None:
    with pytest.raises(KeyError, match="Oz"):
        features(
            tmp_path,
            '[rois]\nback = ["Pz", "Oz"]\n\n'
            '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
        )


def test_each_entry_is_reported_as_a_step(tmp_path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        HEAD
        + '[[features]]\nmeasure = "integrated_band_power"\n\n[[features]]\nmeasure = "variance"\n'
    )
    steps: list[tuple[str, int, int]] = []

    compute_features(
        make_epochs(),
        load_recipe(path),
        recording="sub-test_task-test",
        on_step=lambda *s: steps.append(s),
    )

    assert steps == [("integrated_band_power", 1, 2), ("variance", 2, 2)]


def test_multitaper_spectra_span_the_whole_epoch_by_default(tmp_path) -> None:
    result = features(
        tmp_path,
        '[spectra]\nmethod = "multitaper"\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert {m.source for m in result.epochs.meta} == {"multitaper"}
    assert {m.window for m in result.epochs.meta} == {"all"}
    # Multitaper resolves sfreq / n_samples over the whole epoch; welch would give
    # sfreq / n_fft with its 500-sample segments.
    resolutions = [m.freq_resolution_hz for m in result.epochs.meta]
    assert resolutions == pytest.approx([SFREQ / 501] * len(resolutions))


def test_multitaper_windows_measured_together_must_be_equally_long(tmp_path) -> None:
    with pytest.raises(ValueError, match="equally long"):
        features(
            tmp_path,
            '[spectra]\nmethod = "multitaper"\nbandwidth = 3.0\n\n'
            "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_a_multitaper_bandwidth_too_narrow_for_its_window_is_an_error(tmp_path) -> None:
    # 2 Hz is 1.01 frequency bins of a 0.5 s window: MNE would fall back, with only a
    # warning, to a single taper that keeps 79% of its power in the band.
    with pytest.raises(ValueError, match="1.35 frequency bins.*window 'base'"):
        features(
            tmp_path,
            '[spectra]\nmethod = "multitaper"\n\n'
            "[windows]\nbase = [-0.5, 0.0]\n\n"
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_welch_segment_longer_than_a_window_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_fft = 400"):
        features(
            tmp_path,
            "[spectra]\nn_fft = 400\n\n[windows]\nbase = [-0.5, 0.0]\n\n"
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_wpli_is_estimated_per_trial_group(tmp_path) -> None:
    pytest.importorskip("mne_connectivity")
    result = features(
        tmp_path,
        '[trials]\nby = "event"\n\n[[features]]\nmeasure = "wpli"\nbands = ["alpha"]\n',
    )

    assert result.crosstrial is not None
    assert result.crosstrial.row_labels == ("left", "right")
    assert {m.space_kind for m in result.crosstrial.meta} == {"pair"}


def test_morlet_log_spacing_reaches_the_top_band_edge(tmp_path) -> None:
    # fmax defaults to the top band's upper edge, so the grid has to reach it exactly:
    # a last frequency an ULP short leaves that band unintegrable.
    result = features(
        tmp_path,
        "[bands]\ntheta = [4.0, 8.0]\nalpha = [8.0, 13.0]\nbeta = [13.0, 30.0]\n\n"
        '[spectra]\nmethod = "morlet"\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "mean_tfr_power"\nbands = ["beta"]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None and result.epochs.n_rows == 12


@pytest.mark.parametrize("measure", ["variance", "erds_mean"])
@pytest.mark.parametrize("channel_kind", ["bad_eeg", "eog"])
def test_time_domain_features_preserve_runner_channel_selection(tmp_path, measure, channel_kind):
    epochs = make_epochs()
    if channel_kind == "bad_eeg":
        epochs.info["bads"] = ["Fz"]
    else:
        epochs.set_channel_types({"Fz": "eog"})
    body = f'[[features]]\nmeasure = "{measure}"\nspatial = ["channels"]\n'
    if measure == "erds_mean":
        body = (
            "[windows]\nbase = [-0.5, 0.0]\nstim = [0.25, 1.25]\n"
            + body
            + 'bands = ["alpha"]\nbaseline = "base"\n'
        )

    result = features(tmp_path, body, epochs)

    assert {meta.space for meta in result.epochs.meta} == set(epochs.ch_names)
    assert epochs.info["bads"] == (["Fz"] if channel_kind == "bad_eeg" else [])


def test_an_roi_pattern_is_resolved_against_the_recordings_channels(tmp_path) -> None:
    listed = features(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3", "F4"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )
    matched = features(
        tmp_path,
        '[rois]\nfront = { match = ["^F[z34]$"] }\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )

    assert listed.epochs is not None and matched.epochs is not None
    np.testing.assert_array_equal(matched.epochs.values, listed.epochs.values)


def test_an_roi_pattern_matching_no_channel_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="'temporal'.*T7"):
        features(
            tmp_path,
            '[rois]\ntemporal = { match = ["^T7$"] }\n\n'
            '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
        )


def test_an_error_names_the_file_entry_even_after_a_multi_measure_entry(tmp_path) -> None:
    with pytest.raises(ValueError) as caught:
        features(
            tmp_path,
            "[windows]\nlate = [1.0, 3.0]\n\n"
            '[[features]]\nmeasures = ["variance", "kurtosis"]\nwindows = ["all"]\n\n'
            '[[features]]\nmeasure = "variance"\nwindows = ["late"]\n',
        )

    assert caught.value.__notes__ == ["features[1] (variance)"]


def test_each_entry_is_timed(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasures = ["integrated_band_power", "variance"]\n\n'
        '[[features]]\nmeasure = "kurtosis"\n',
    )

    assert [(t.entry, t.measure) for t in result.timings] == [
        (0, "integrated_band_power"),
        (0, "variance"),
        (1, "kurtosis"),
    ]
    assert all(t.seconds >= 0.0 for t in result.timings)
