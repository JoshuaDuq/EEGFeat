from __future__ import annotations

_DOMAIN = "eeg"
_NO_BAND = "broadband"
_NO_WINDOW = "all"


def feature_name(
    *,
    measure: str,
    band: str | None,
    space: str,
    window: str | None,
    normalization: str,
) -> str:
    """Render the canonical feature name for one column.

    The name is ``{domain}_{measure}_{band}_{space}_{window}_{normalization}``.

    Parameters
    ----------
    measure : str
        Measure label, e.g. ``"power"`` or ``"peak_freq"``.
    band : str or None
        Band name, or None for a measure that spans the fitted range.
    space : str
        Channel name, ROI name, or ``"global"``.
    window : str or None
        Time window name, or None when the spectrum spans the whole segment.
    normalization : str
        One of ``"raw"``, ``"log10"``, ``"log_ratio"``, ``"db"``.

    Returns
    -------
    str
        The canonical name. It always splits into exactly six underscore-
        separated fields, because field values have their own underscores and
        spaces replaced by hyphens.
    """
    for label, value in (("measure", measure), ("space", space), ("normalization", normalization)):
        if not value:
            raise ValueError(f"feature_name requires a non-empty {label}.")
    fields = (
        _DOMAIN,
        measure,
        band if band else _NO_BAND,
        space,
        window if window else _NO_WINDOW,
        normalization,
    )
    return "_".join(_slug(field) for field in fields)


def _slug(value: str) -> str:
    # The underscore is the field separator, so it must not survive inside a field.
    return value.strip().lower().replace("_", "-").replace(" ", "-")
