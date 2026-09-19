import math
from pathlib import Path

import pytest

from eegfeat.bands import BANDS_STANDARD, Band
from eegfeat.runner import RecipeError, load_recipe
from eegfeat.spectra import Window

HEAD = """
[inputs]
root = "data"

[output]
root = "out"
"""


def _load(tmp_path: Path, body: str, head: str = HEAD):
    path = tmp_path / "recipe.toml"
    path.write_text(head + body)
    return load_recipe(path)


def _problems(tmp_path: Path, body: str, head: str = HEAD) -> str:
    with pytest.raises(RecipeError) as caught:
        _load(tmp_path, body, head)
    return str(caught.value)


# --- defaults -----------------------------------------------------------------


def test_minimal_recipe_takes_the_library_defaults(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert recipe.bands == BANDS_STANDARD
    assert recipe.windows == ()
    assert recipe.inputs.pattern == "**/*_epo.fif"
    assert recipe.inputs.picks == "eeg"
    assert recipe.inputs.exclude_bads is True
    (spec,) = recipe.features
    assert spec.measure == "integrated_band_power"
    assert spec.bands == BANDS_STANDARD
    assert spec.spatial == ("channels", "global")


def test_roots_resolve_against_the_recipe_directory(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert recipe.inputs.root == tmp_path / "data"
    assert recipe.output.root == tmp_path / "out"


def test_without_windows_a_measure_spans_the_whole_epoch(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "variance"\n')

    (window,) = recipe.features[0].windows
    assert window.name == "all"
    assert math.isinf(window.tmin) and math.isinf(window.tmax)


def test_spectra_default_to_the_span_of_the_recipe_bands(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[bands]\ntheta = [4.0, 8.0]\nbeta = [13.0, 30.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert recipe.spectra.method == "welch"
    assert (recipe.spectra.fmin, recipe.spectra.fmax) == (4.0, 30.0)


def test_bands_keep_recipe_order(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[bands]\nbeta = [13.0, 30.0]\ntheta = [4.0, 8.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert [b.name for b in recipe.bands] == ["beta", "theta"]


def test_default_windows_leave_out_the_entry_baseline(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nearly = [0.0, 0.5]\nlate = [0.5, 1.0]\n\n"
        '[[features]]\nmeasure = "burst_rate"\nbaseline = "base"\n',
    )

    (spec,) = recipe.features
    assert spec.baseline == Window("base", -0.5, 0.0)
    assert [w.name for w in spec.windows] == ["early", "late"]


def test_series_default_to_broadband_and_accept_band_names(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[[features]]\nmeasure = "variance"\n\n'
        '[[features]]\nmeasure = "sample_entropy"\nseries = ["broadband", "alpha"]\n',
    )

    assert recipe.features[0].series == (None,)
    assert recipe.features[1].series == (None, Band("alpha", 8.0, 13.0))


def test_pac_pairs_resolve_to_bands(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "pac"\npairs = [["theta", "gamma"]]\n')

    assert recipe.features[0].pairs == ((Band("theta", 4.0, 8.0), Band("gamma", 30.0, 45.0)),)


# --- parameters passed through to the library ---------------------------------


def test_list_parameters_become_the_tuples_the_library_expects(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[[features]]\nmeasure = "aperiodic"\nfit_range = [2, 30]\n\n'
        '[[features]]\nmeasure = "multiscale_entropy"\nscales = [1, 2, 3]\n',
    )

    assert recipe.features[0].params == {"fit_range": (2.0, 30.0)}
    assert recipe.features[1].params == {"scales": (1, 2, 3)}


def test_integer_given_for_a_float_parameter_becomes_a_float(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "burst_rate"\nmin_duration_ms = 150\n')

    value = recipe.features[0].params["min_duration_ms"]
    assert value == 150.0 and isinstance(value, float)


def test_wrongly_typed_parameter_is_rejected(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "burst_rate"\nthreshold = "high"\n')

    assert "threshold" in problems and "number" in problems


def test_parameter_outside_its_allowed_values_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nnormalize = "zscore"\n'
    )

    assert "normalize" in problems and "'log10'" in problems


def test_parameter_the_measure_does_not_take_is_rejected(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "aperiodic"\nbands = ["alpha"]\n')

    assert "aperiodic" in problems and "'bands'" in problems


# --- cross-references ---------------------------------------------------------


def test_unknown_measure_is_rejected(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "band_powr"\n')

    assert "band_powr" in problems


def test_undefined_band_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nbands = ["mu"]\n'
    )

    assert "'mu'" in problems


def test_undefined_window_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[windows]\nstim = [0.0, 1.0]\n\n[[features]]\nmeasure = "variance"\nwindows = ["late"]\n',
    )

    assert "'late'" in problems


def test_measure_that_needs_a_baseline_requires_one(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[windows]\nstim = [0.0, 1.0]\n\n[[features]]\nmeasure = "erds_mean"\n'
    )

    assert "erds_mean" in problems and "baseline" in problems


def test_baseline_needs_windows_to_name(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "burst_rate"\nbaseline = "base"\n')

    assert "'base'" in problems


def test_band_power_cannot_both_consume_and_report_its_baseline(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'baseline = "base"\nwindows = ["base", "stim"]\n'
        'normalize = "db"\n',
    )

    assert "'base'" in problems


def test_roi_level_requires_rois(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n'
    )

    assert "[rois]" in problems


def test_graph_measures_need_a_single_node_level(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\nback = ["Pz", "P3"]\n\n'
        '[[features]]\nmeasure = "envelope_correlation"\nspatial = ["channels", "rois"]\n'
        'graph = ["global_efficiency"]\n',
    )

    assert "graph" in problems


def test_clustering_coefficient_needs_its_threshold(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[[features]]\nmeasure = "envelope_correlation"\ngraph = ["clustering_coefficient"]\n',
    )

    assert "clustering_threshold" in problems


def test_pac_requires_pairs(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "pac"\n')

    assert "pairs" in problems


# --- sections -----------------------------------------------------------------


def test_unknown_section_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[plots]\nsize = 3\n\n[[features]]\nmeasure = "integrated_band_power"\n'
    )

    assert "plots" in problems


def test_unknown_input_key_is_rejected(tmp_path) -> None:
    head = '[inputs]\nroot = "data"\nglob = "*.fif"\n\n[output]\nroot = "out"\n'
    problems = _problems(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n', head)

    assert "glob" in problems


def test_input_root_is_required(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n', '[output]\nroot = "out"\n'
    )

    assert "inputs" in problems and "root" in problems


def test_a_recipe_without_features_is_rejected(tmp_path) -> None:
    assert "features" in _problems(tmp_path, "")


def test_invalid_band_bounds_are_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[bands]\nalpha = [13.0, 8.0]\n\n[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert "alpha" in problems


def test_all_is_reserved_for_the_whole_epoch(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[windows]\nall = [0.0, 1.0]\n\n[[features]]\nmeasure = "variance"\n'
    )

    assert "'all'" in problems


def test_global_is_reserved_among_rois(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[rois]\nglobal = ["Fz"]\n\n[[features]]\nmeasure = "integrated_band_power"\n'
    )

    assert "'global'" in problems


def test_metadata_trials_need_a_column(tmp_path) -> None:
    problems = _problems(tmp_path, '[trials]\nby = "metadata"\n\n[[features]]\nmeasure = "itpc"\n')

    assert "column" in problems


def test_spectral_option_of_another_method_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nn_fft = 512\n\n[[features]]\nmeasure = "mean_tfr_power"\n',
    )

    assert "n_fft" in problems and "welch" in problems


def test_microstate_settings_are_type_checked(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[microstates]\nn_states = "four"\n\n[[features]]\nmeasure = "microstate_coverage"\n',
    )

    assert "n_states" in problems


def test_every_problem_is_reported_at_once(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["mu"]\n\n'
        '[[features]]\nmeasure = "burst_rate"\nthreshold = "high"\n',
    )

    assert "'mu'" in problems and "threshold" in problems


def test_malformed_toml_names_the_file(tmp_path) -> None:
    problems = _problems(tmp_path, "[[features]\nmeasure = \n")

    assert "recipe.toml" in problems


def test_pac_pairs_in_one_entry_need_distinct_amplitude_bands(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[[features]]\nmeasure = "pac"\npairs = [["theta", "gamma"], ["alpha", "gamma"]]\n',
    )

    assert "'gamma'" in problems


def test_measure_whose_optional_dependency_is_missing_is_rejected(tmp_path) -> None:
    import importlib.util

    if importlib.util.find_spec("mne_connectivity") is not None:
        pytest.skip("mne-connectivity is installed")
    problems = _problems(tmp_path, '[[features]]\nmeasure = "wpli"\n')

    assert "eegfeat[connectivity]" in problems
