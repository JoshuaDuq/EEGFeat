from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

from eegfeat.bands import Band
from eegfeat.spectra import Spectra, Window
from eegfeat.table import ComputationSpec

FREQS = np.arange(2.0, 40.25, 0.25)
ALPHA = Band("alpha", 8.0, 13.0)


def _spectra(power, freqs=FREQS):
    data = np.asarray(power).reshape(1, 1, 1, -1)
    return Spectra(
        data=data,
        freqs=freqs,
        ch_names=("C3",),
        windows=(Window("all", -np.inf, np.inf),),
        coverage=np.full(data.shape, 0.8),
        support=np.ones(data.shape),
        source="welch",
        representation="psd",
        row_ids=(("spectral", 0, "event"),),
        computation=ComputationSpec.create("test"),
        flags={"input_flag": np.ones(data.shape[:3], dtype=bool)},
    )


def _compute(spectra, **parameters):
    assert find_spec("eegfeat.spectral_model") is not None, "spectral parameterization missing"
    from eegfeat.spectral_model import spectral_parameterization

    return spectral_parameterization(spectra, include_global=False, **parameters)


def test_fixed_model_recovers_aperiodic_and_gaussian_peak_parameters():
    power = 10 ** (1.0 - 1.7 * np.log10(FREQS) + 0.5 * np.exp(-0.5 * ((FREQS - 10) / 1) ** 2))
    table = _compute(_spectra(power), bands=(ALPHA,))
    for measure, expected, tolerance in [
        ("specparam_offset", 1.0, 0.01),
        ("specparam_exponent", 1.7, 0.01),
        ("specparam_peak_cf", 10.0, 0.1),
        ("specparam_peak_height", 0.5, 0.02),
        ("specparam_peak_width", 2.0, 0.1),
    ]:
        assert table.select(measure=measure).values.item() == pytest.approx(expected, abs=tolerance)
    assert table.select(measure="specparam_r_squared").values.item() > 0.99
    assert table.select(measure="specparam_error").values.item() < 0.01
    assert table.select(measure="specparam_n_peaks").values.item() == 1.0
    assert table.row_ids == (("spectral", 0, "event"),)
    assert table.flags["input_flag"].all()
    np.testing.assert_allclose(table.coverage, 0.8)
    assert table.select(measure="specparam_peak_cf").meta[0].band == ALPHA
    assert table.select(measure="specparam_offset").meta[0].band is None


def test_knee_model_recovers_the_knee_in_the_aperiodic_denominator():
    power = 10 ** (1.0 - np.log10(20.0 + FREQS**2))
    table = _compute(_spectra(power), aperiodic_mode="knee", max_n_peaks=0)
    assert table.select(measure="specparam_knee").values.item() == pytest.approx(20.0, rel=0.01)
    assert table.select(measure="specparam_exponent").values.item() == pytest.approx(2.0, abs=0.01)


def test_no_peak_is_nan_and_flagged_without_replacing_the_model():
    table = _compute(_spectra(FREQS**-2), bands=(ALPHA,), max_n_peaks=0)
    peaks = table.select(measure="specparam_peak_cf")
    assert np.isnan(peaks.values).all()
    assert peaks.flags["spectral_no_peak"].all()
    assert not table.select(measure="specparam_offset").flags["spectral_no_peak"].any()


def test_strongest_peak_is_selected_when_one_band_has_two_peaks():
    periodic = 0.3 * np.exp(-0.5 * ((FREQS - 9) / 0.3) ** 2)
    periodic += 0.7 * np.exp(-0.5 * ((FREQS - 12) / 0.3) ** 2)
    table = _compute(_spectra(10 ** (-2 * np.log10(FREQS) + periodic)), bands=(ALPHA,))
    assert table.select(measure="specparam_peak_cf").values.item() == pytest.approx(12, abs=0.1)


@pytest.mark.parametrize("kind", ["zero", "nan", "inf", "grid", "representation", "outside"])
def test_spectral_model_refuses_invalid_input(kind):
    power = FREQS**-2
    spectrum = _spectra(power)
    if kind in ("zero", "nan", "inf"):
        power[4] = {"zero": 0, "nan": np.nan, "inf": np.inf}[kind]
        spectrum = _spectra(power)
    if kind == "grid":
        spectrum = _spectra(power, np.geomspace(2, 40, FREQS.size))
    if kind == "representation":
        spectrum = replace(spectrum, representation="aperiodic_ratio")
    if kind == "outside":
        spectrum = replace(spectrum, passband=(4, 45))
    with pytest.raises(ValueError):
        _compute(spectrum)


@pytest.mark.parametrize(
    "parameters",
    [
        {"aperiodic_mode": "auto"},
        {"max_n_peaks": -1},
        {"peak_width_limits": (2, 1)},
        {"peak_threshold": -1},
        {"min_peak_height": -1},
        {"fit_range": (1, 41)},
        {"bands": (Band("outside", 1, 3),)},
    ],
)
def test_spectral_model_refuses_invalid_settings(parameters):
    with pytest.raises(ValueError):
        _compute(_spectra(FREQS**-2), **parameters)
