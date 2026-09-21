"""Evoked-potential measures on ERP CORE's Flankers task.

Three textbook components on one subject: the visual N1 over lateral occipital
cortex about 180 ms after the arrows appear, the P3 over the parietal midline
from 300 to 500 ms, and the error-related negativity at FCz in the 100 ms after
a wrong button press. The single-trial peak and area measures must find them,
and must be the documented formulas.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import ttest_ind

import eegfeat as ef
from validation.loaders import Recording, load_erp_core

LATERAL_OCCIPITAL = {"occipital": ["PO7", "PO8", "O1", "O2"]}
PARIETAL = {"parietal": ["Pz", "CPz"]}
FCZ = {"fcz": ["FCz"]}
N1 = ef.Window("n1", 0.1, 0.25)
P3 = ef.Window("p3", 0.3, 0.5)
PRESTIMULUS = ef.Window("prestimulus", -0.2, 0.0)
ERN = ef.Window("ern", 0.0, 0.1)

DATASET = "erp_core"


@pytest.fixture(scope="module")
def stimulus() -> Recording:
    return load_erp_core("stimulus")


@pytest.fixture(scope="module")
def response() -> Recording:
    return load_erp_core("response")


@pytest.fixture(scope="module")
def stimulus_signal(stimulus: Recording) -> ef.Signal:
    return ef.Signal.from_epochs(stimulus.epochs, recording=stimulus.name)


@pytest.mark.validates(
    "Signal.from_epochs",
    kind="behaviour",
    claim="The loaded epochs match the published task",
    criterion="1024 Hz, 400 stimuli, more than 300 correct and 30 error trials",
)
def test_the_epochs_are_the_published_task(stimulus: Recording) -> None:
    assert stimulus.epochs.info["sfreq"] == 1024.0
    assert len(stimulus.epochs) == 400
    counts = stimulus.metadata["correct"].value_counts()
    assert counts[1] > 300 and counts[0] > 30, counts.to_dict()


@pytest.mark.validates(
    "peak_latency",
    kind="physiology",
    claim="Single-trial N1 latency over lateral occipital cortex clusters near 180 ms",
    criterion="median in 150 to 220 ms; interquartile range below 50 ms; 60 percent within 40 ms",
)
def test_n1_latency_clusters_on_the_component(
    stimulus_signal: ef.Signal, record: Callable[[str], None]
) -> None:
    """Single-trial negative peaks land near 180 ms with an interquartile range of a few
    tens of milliseconds, on a window that is 150 ms wide."""
    latency = ef.peak_latency(
        [stimulus_signal],
        windows=[N1],
        polarity="negative",
        groups=LATERAL_OCCIPITAL,
        include_global=False,
    ).values[:, 0]
    lower, upper = np.percentile(latency, [25, 75])
    record(
        f"median {np.median(latency) * 1000:.0f} ms, interquartile range {lower * 1000:.0f} to "
        f"{upper * 1000:.0f} ms on a 150 ms window"
    )
    assert 0.15 < np.median(latency) < 0.22, np.median(latency)
    assert upper - lower < 0.05, (lower, upper)
    assert np.mean(np.abs(latency - np.median(latency)) < 0.04) > 0.6


@pytest.mark.validates(
    "area_under_curve",
    kind="physiology",
    claim="The P3 area over Pz and CPz is positive on most trials",
    criterion="positive on more than 70 percent of trials; pre-stimulus area near zero",
)
def test_p3_area_is_positive_over_parietal_cortex(
    stimulus_signal: ef.Signal, record: Callable[[str], None]
) -> None:
    p3 = ef.area_under_curve(
        [stimulus_signal], windows=[P3], groups=PARIETAL, include_global=False
    ).values[:, 0]
    pre = ef.area_under_curve(
        [stimulus_signal], windows=[PRESTIMULUS], groups=PARIETAL, include_global=False
    ).values[:, 0]
    record(f"positive on {np.mean(p3 > 0.0) * 100:.0f} percent of {p3.size} trials")
    assert np.mean(p3 > 0.0) > 0.7, np.mean(p3 > 0.0)
    assert np.median(p3) > 0.0 and abs(np.median(pre)) < 0.1 * np.median(p3)


@pytest.mark.validates(
    "mean_amplitude",
    "peak_amplitude",
    kind="physiology",
    claim="Wrong presses produce a negativity at FCz in the following 100 ms",
    criterion="error trials below correct by more than 3 microvolts; p below 1e-6",
)
def test_errors_produce_a_negativity_at_fcz(
    response: Recording, record: Callable[[str], None]
) -> None:
    signal = ef.Signal.from_epochs(response.epochs, recording=response.name)
    correct = response.metadata["correct"].to_numpy() == 1
    mean = ef.mean_amplitude([signal], windows=[ERN], groups=FCZ, include_global=False).values[:, 0]
    trough = ef.peak_amplitude(
        [signal], windows=[ERN], polarity="negative", groups=FCZ, include_global=False
    ).values[:, 0]
    record(
        f"mean 0 to 100 ms at FCz: {np.mean(mean[correct]) * 1e6:+.1f} microvolts on "
        f"{int(correct.sum())} correct trials, {np.mean(mean[~correct]) * 1e6:+.1f} on "
        f"{int((~correct).sum())} errors"
    )
    for values in (mean, trough):
        assert np.mean(values[~correct]) < np.mean(values[correct])
        assert ttest_ind(values[~correct], values[correct], alternative="less").pvalue < 1e-6
    # Microvolts: the published ERN is several microvolts more negative than correct trials.
    assert (np.mean(mean[correct]) - np.mean(mean[~correct])) > 3e-6


@pytest.mark.validates(
    "peak_amplitude",
    "peak_latency",
    "area_under_curve",
    kind="formula",
    claim="Peak amplitude, peak latency and area are their documented formulas",
    criterion="exact for peaks; relative error below 1e-9 for area",
)
def test_peak_and_area_measures_are_the_documented_formulas(
    stimulus: Recording, stimulus_signal: ef.Signal
) -> None:
    data = stimulus.epochs.get_data(picks="eeg")
    times = stimulus.epochs.times
    inside = (times >= N1.tmin) & (times <= N1.tmax)
    window = data[:, :, inside]

    negative = ef.peak_amplitude(
        [stimulus_signal], windows=[N1], polarity="negative", include_global=False
    ).values
    np.testing.assert_allclose(negative, window.min(axis=-1))
    positive = ef.peak_amplitude(
        [stimulus_signal], windows=[N1], polarity="positive", include_global=False
    ).values
    np.testing.assert_allclose(positive, window.max(axis=-1))
    latency = ef.peak_latency(
        [stimulus_signal], windows=[N1], polarity="negative", include_global=False
    ).values
    np.testing.assert_array_equal(latency, times[inside][np.argmin(window, axis=-1)])
    area = ef.area_under_curve([stimulus_signal], windows=[N1], include_global=False).values
    np.testing.assert_allclose(area, trapezoid(window, times[inside], axis=-1), rtol=1e-9)
