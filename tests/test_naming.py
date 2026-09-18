import pytest

from eegfeat.naming import feature_name


def test_renders_all_six_fields_in_canonical_order() -> None:
    assert (
        feature_name(
            measure="power", band="alpha", space="C3", window="stim", normalization="log_ratio"
        )
        == "eeg_power_alpha_c3_stim_log-ratio"
    )


def test_absent_band_and_window_use_explicit_placeholders() -> None:
    assert (
        feature_name(measure="slope", band=None, space="global", window=None, normalization="raw")
        == "eeg_slope_broadband_global_all_raw"
    )


def test_name_always_splits_into_exactly_six_fields() -> None:
    name = feature_name(
        measure="peak freq",
        band="alpha",
        space="Left Central",
        window="post stim",
        normalization="log_ratio",
    )
    assert len(name.split("_")) == 6
    assert name == "eeg_peak-freq_alpha_left-central_post-stim_log-ratio"


@pytest.mark.parametrize("field", ["measure", "space", "normalization"])
def test_empty_required_field_raises(field: str) -> None:
    kwargs = {
        "measure": "power",
        "band": "alpha",
        "space": "C3",
        "window": "stim",
        "normalization": "raw",
    }
    kwargs[field] = ""
    with pytest.raises(ValueError):
        feature_name(**kwargs)  # type: ignore[arg-type]
