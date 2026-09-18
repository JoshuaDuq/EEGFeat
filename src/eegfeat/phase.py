from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from eegfeat._expand import expand_signal
from eegfeat.baseline import EPS
from eegfeat.signal import BandSignal
from eegfeat.spectra import Window
from eegfeat.table import FeatureTable


def itpc(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Inter-trial phase coherence.

    The length of the mean unit phase vector across trials, averaged over the
    window:

    .. math::

       \\mathrm{ITPC} = \\frac{1}{T} \\sum_t
       \\left| \\frac{1}{N} \\sum_n e^{i \\phi_n(t)} \\right|

    One means the phase is identical on every trial at that latency, zero that it
    is uniformly distributed. Trials are averaged first and time second; reversing
    the order measures something else.

    **This is estimated across trials, so the result has one row per trial group,
    not one per epoch.** The returned table carries ``row_labels`` and cannot be
    concatenated with per-epoch features. Broadcasting group-level estimates onto
    single epochs introduces pseudo-replication in downstream statistical models;
    models fitted on broadcasted tables incorrectly treat a single group estimate
    as N independent observations.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    trials : sequence of str, optional
        A label per epoch, giving the group each trial belongs to. One row is
        returned per distinct label, in sorted order. None estimates from all
        trials together and returns a single row labelled ``"all"``.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Coherence in ``[0, 1]``, with one row per trial group.
    """
    n_epochs = signals[0].n_epochs if signals else 0
    row_groups, labels = _resolve_rows(trials, n_epochs)

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal, times
        return {"itpc": _itpc(trace, row_groups, len(labels))}

    return expand_signal(
        signals,
        # The kernel needs phase, and the expander hands it whatever this returns.
        trace_of=lambda signal: signal.phase,
        kernel=kernel,
        units={"itpc": "a.u."},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        row_groups=row_groups,
        row_labels=labels,
    )


def pac(
    phase_signal: BandSignal,
    amplitude_signal: BandSignal,
    *,
    windows: Sequence[Window],
    normalize: bool = True,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
) -> FeatureTable:
    """Phase-amplitude coupling by mean vector length.

    The amplitude-weighted resultant of the slow band's phase:

    .. math::

       \\mathrm{MVL} = \\frac{\\left| \\sum_t A(t) e^{i \\phi(t)} \\right|}{\\sum_t A(t)}

    where :math:`\\phi` is the phase of ``phase_signal`` and :math:`A` the envelope
    of ``amplitude_signal``. Normalizing by the summed amplitude makes the value
    independent of overall power, so it is comparable across channels and trials;
    without it the result scales with amplitude and is not.

    Computed within each trial, so it carries no cross-trial leakage and the
    result has one row per epoch. No surrogate correction is applied: a raw
    coupling value is biased upward by amplitude and phase autocorrelation, so
    compare against a null you construct rather than reading it absolutely.

    Parameters
    ----------
    phase_signal : BandSignal
        The slower band, whose phase modulates. Conventionally theta or alpha.
    amplitude_signal : BandSignal
        The faster band, whose envelope is modulated. Conventionally gamma.
    windows : sequence of Window
        Analysis windows.
    normalize : bool, default True
        Divide by the summed amplitude. Leave True unless you specifically want
        the unnormalized resultant.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.

    Returns
    -------
    FeatureTable
        Coupling strength, dimensionless when normalized. The band recorded on
        each column is the amplitude band.
    """
    if phase_signal.ch_names != amplitude_signal.ch_names:
        raise ValueError("phase_signal and amplitude_signal must share the same channels.")
    if phase_signal.times.shape != amplitude_signal.times.shape or not np.allclose(
        phase_signal.times, amplitude_signal.times
    ):
        raise ValueError("phase_signal and amplitude_signal must share the same time axis.")
    slow, fast = phase_signal.band, amplitude_signal.band
    if slow is not None and fast is not None and slow.fmin >= fast.fmin:
        raise ValueError(
            f"phase_signal band {slow.name!r} must be slower than amplitude_signal "
            f"band {fast.name!r}; got {slow.fmin} Hz and {fast.fmin} Hz."
        )

    unit_phase = np.exp(1j * phase_signal.phase)

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal
        window = np.isin(amplitude_signal.times, times)
        return {"pac": _mean_vector_length(unit_phase[:, :, window], trace, normalize)}

    return expand_signal(
        [amplitude_signal],
        trace_of=lambda signal: signal.envelope,
        kernel=kernel,
        units={"pac": "a.u." if normalize else "V"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
    )


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


def _itpc(
    phase: npt.NDArray[np.float64],
    row_groups: npt.NDArray[np.int_],
    n_rows: int,
) -> npt.NDArray[np.float64]:
    unit_vectors = np.where(np.isfinite(phase), np.exp(1j * phase), np.nan)
    out = np.full((n_rows, phase.shape[1]), np.nan)
    for row in range(n_rows):
        member = unit_vectors[row_groups == row]
        if member.shape[0] == 0:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            # Trials first, then time: the coherence at each latency, averaged.
            coherence = np.abs(np.nanmean(member, axis=0))
            out[row] = np.nanmean(np.where(np.isfinite(coherence), coherence, np.nan), axis=1)
    return out


def _mean_vector_length(
    unit_phase: npt.NDArray[np.complex128],
    amplitude: npt.NDArray[np.float64],
    normalize: bool,
) -> npt.NDArray[np.float64]:
    finite = np.isfinite(amplitude) & np.isfinite(unit_phase)
    weighted = np.where(finite, amplitude * unit_phase, 0.0)
    resultant = np.abs(weighted.sum(axis=2))
    if not normalize:
        total = np.maximum(finite.sum(axis=2), 1)
        return np.asarray(resultant / total, dtype=float)
    weight = np.where(finite, amplitude, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > EPS, resultant / weight, np.nan)
