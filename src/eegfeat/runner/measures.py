"""The measures a recipe may name, and how their parameters are checked.

A recipe entry's parameters are the keyword parameters of the library function
it names, checked against that function's own annotations. Adding a parameter to
a measure therefore makes it settable from a recipe with no change here.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import eegfeat as ef
from eegfeat.table import FeatureTable

Kind = Literal["spectra", "series", "signals", "pac", "wpli", "microstates"]
"""What a measure is computed from, named after the input the runner builds."""

SUPPLIED = frozenset(
    {
        "spectra",
        "series",
        "signals",
        "signal",
        "phase_signal",
        "amplitude_signal",
        "segmentation",
        "bands",
        "band",
        "windows",
        "baseline",
        "groups",
        "include_global",
        "trials",
    }
)
"""Parameters the runner fills in itself; a recipe reaches them through its own keys."""

_KINDS: dict[str, Kind] = {
    "spectra": "spectra",
    "series": "series",
    "signals": "signals",
    "phase_signal": "pac",
    "signal": "wpli",
    "segmentation": "microstates",
}

_NOUNS: dict[object, str] = {bool: "booleans", int: "integers", float: "numbers", str: "strings"}


@dataclass(frozen=True, eq=False)
class Measure:
    """A library function a recipe can name."""

    name: str
    function: Callable[..., Any]

    @property
    def parameters(self) -> Mapping[str, inspect.Parameter]:
        """The function's parameters, in order."""
        return inspect.signature(self.function).parameters

    @property
    def kind(self) -> Kind:
        """The input the measure is computed from."""
        return _KINDS[next(iter(self.parameters))]

    def takes(self, parameter: str) -> bool:
        """Whether the function has a parameter of this name."""
        return parameter in self.parameters

    @property
    def baseline_required(self) -> bool:
        """Whether the function has a baseline parameter without a default."""
        baseline = self.parameters.get("baseline")
        return baseline is not None and baseline.default is inspect.Parameter.empty

    @property
    def settable(self) -> dict[str, Any]:
        """Keyword parameters a recipe can set, mapped to their annotations.

        Excludes what the runner supplies and anything a recipe cannot express,
        such as an array.
        """
        hints = typing.get_type_hints(self.function)
        return {
            name: hints[name]
            for name, parameter in self.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and name not in SUPPLIED
            and describe(hints[name]) is not None
        }


def _measures(*names: str) -> dict[str, Measure]:
    return {name: Measure(name, getattr(ef, name)) for name in names}


MEASURES: dict[str, Measure] = _measures(
    "integrated_band_power",
    "mean_psd",
    "mean_tfr_power",
    "aperiodic",
    "peak_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "spectral_edge",
    "variance",
    "peak_to_peak",
    "mean_amplitude",
    "area_under_curve",
    "peak_amplitude",
    "peak_latency",
    "sample_entropy",
    "multiscale_entropy",
    "burst_count",
    "burst_rate",
    "burst_duration",
    "burst_amplitude",
    "fraction_above_threshold",
    "erds_mean",
    "erds_slope",
    "erd_magnitude",
    "erd_duration",
    "ers_magnitude",
    "ers_duration",
    "erds_peak_latency",
    "erds_onset_latency",
    "erds_rebound_latency",
    "itpc",
    "ppc",
    "pac",
    "envelope_correlation",
    "wpli",
    "microstate_coverage",
    "microstate_duration",
    "microstate_occurrence",
    "microstate_transitions",
)
"""Every measure a recipe entry can name."""

SEGMENTATION = Measure("segment", ef.segment)
"""Microstate segmentation, configured once per recipe and shared by its measures."""

GRAPH: dict[str, Callable[..., FeatureTable]] = {
    "global_efficiency": ef.global_efficiency,
    "clustering_coefficient": ef.clustering_coefficient,
}
"""Network summaries a connectivity entry can add."""

SPECTRAL_INPUT: dict[str, tuple[str, tuple[str, ...]]] = {
    "mean_psd": ("a power spectral density", ("welch", "multitaper")),
    "integrated_band_power": ("a power spectral density", ("welch", "multitaper")),
    "mean_tfr_power": ("time-frequency power", ("morlet",)),
}
"""Measures that read only one kind of spectrum: what they read, and the methods giving it.

A spectral measure not named here reads either kind. The two are dimensionally
different quantities, so no recipe can feed a measure the wrong one.
"""

REQUIRES: dict[str, tuple[str, str]] = {
    "wpli": ("mne_connectivity", "connectivity"),
    "microstate_coverage": ("sklearn", "microstates"),
    "microstate_duration": ("sklearn", "microstates"),
    "microstate_occurrence": ("sklearn", "microstates"),
    "microstate_transitions": ("sklearn", "microstates"),
}
"""Measures that need an optional dependency: the module, and the extra providing it."""


class Mismatch(ValueError):
    """A recipe value does not fit the annotation it was checked against."""


def convert(value: object, annotation: Any) -> object:
    """Turn a TOML value into what an annotation asks for.

    Integers are accepted where a float is expected, and lists become the tuples
    the library takes.

    Raises
    ------
    Mismatch
        When the value cannot be read as the annotation.
    """
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise Mismatch
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise Mismatch
    if annotation is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        raise Mismatch
    if annotation is str:
        if isinstance(value, str):
            return value
        raise Mismatch

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is Literal:
        if any(value == option and type(value) is type(option) for option in args):
            return value
        raise Mismatch
    if origin in (typing.Union, types.UnionType):
        for option in args:
            if option is type(None) or describe(option) is None:
                continue
            try:
                return convert(value, option)
            except Mismatch:
                continue
        raise Mismatch
    if origin is tuple:
        if not isinstance(value, list):
            raise Mismatch
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(convert(item, args[0]) for item in value)
        if len(value) != len(args):
            raise Mismatch
        return tuple(convert(item, arg) for item, arg in zip(value, args, strict=True))
    if origin in (collections.abc.Sequence, list):
        if not isinstance(value, list):
            raise Mismatch
        return tuple(convert(item, args[0]) for item in value)
    raise Mismatch


def describe(annotation: Any) -> str | None:
    """Plain-language description of what an annotation accepts from a recipe.

    None when a recipe cannot express the annotation at all.
    """
    simple = {bool: "true or false", int: "an integer", float: "a number", str: "a string"}
    if annotation in simple:
        return simple[annotation]

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is Literal:
        return "one of " + ", ".join(repr(option) for option in args)
    if origin in (typing.Union, types.UnionType):
        options = [describe(option) for option in args if option is not type(None)]
        present = [option for option in options if option is not None]
        return " or ".join(present) if present else None
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return _list_of(args[0])
        if len(set(args)) == 1 and args[0] in _NOUNS:
            return f"a list of {len(args)} {_NOUNS[args[0]]}"
        return None
    if origin in (collections.abc.Sequence, list):
        return _list_of(args[0])
    return None


def _list_of(item: Any) -> str | None:
    return f"a list of {_NOUNS[item]}" if item in _NOUNS else None
