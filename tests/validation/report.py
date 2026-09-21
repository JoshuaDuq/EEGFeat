"""Write the validation results the docs render.

Every validation test declares what it validates with the ``validates`` marker,
and may record an observed value through the ``record`` fixture. At the end of a
run the rows are merged into ``docs/validation/results.json`` and rendered to
three reStructuredText fragments the validation guide includes, so the numbers on
the page are the numbers the tests saw and cannot drift from them.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import eegfeat

KINDS = ("formula", "estimator", "physiology", "decoding", "behaviour")
KIND_TITLES = {
    "formula": "Formula",
    "estimator": "Estimator",
    "physiology": "Physiology",
    "decoding": "Decoding",
    "behaviour": "Behaviour",
}
KIND_MEANINGS = {
    "formula": "equals an independent NumPy, SciPy or closed-form computation on the same data",
    "estimator": "agrees with a different estimator or library of the same quantity",
    "physiology": "recovers an effect the literature describes, at the level it is known to hold",
    "decoding": "a leakage-safe model finds a contrast strong enough that a correct pipeline must",
    "behaviour": "a documented refusal, identity, null rate or command-line behaviour holds",
}

# The modeling and runner entry points the scorecard reports on; the rest of
# eegfeat.model are helpers reached through these.
MODEL_API = (
    "build_design",
    "within_subject_folds",
    "loso_folds",
    "cross_fit_classification",
    "cross_fit_regression",
    "classification_metrics",
    "regression_metrics",
    "subject_level_r",
    "permutation_test",
    "permutation_importance_over_folds",
    "prediction_intervals",
)
TOOLING = (
    "Spectra.from_spectrum",
    "Spectra.from_tfr",
    "Signal.from_epochs",
    "BandSignal.from_epochs",
    "stack_rows",
    "runner",
    "eegfeat command",
)

DATASET_TITLES = {
    "eegbci": "PhysioNet motor movement",
    "ssvep": "MNE SSVEP",
    "erp_core": "ERP CORE Flankers",
    "sleep": "Sleep-EDF",
}


@dataclass(frozen=True)
class Row:
    nodeid: str
    measures: tuple[str, ...]
    kind: str
    dataset: str
    claim: str
    criterion: str
    observed: str
    passed: bool
    run: str
    cases: int = 1


def feature_api() -> tuple[str, ...]:
    return tuple(
        name
        for name in eegfeat.__all__
        if name[0].islower() and callable(getattr(eegfeat, name)) and name != "concat"
    )


def _versions() -> dict[str, str]:
    packages = ("eegfeat", "mne", "numpy", "scipy", "scikit-learn", "mne-connectivity")
    out = {"python": platform.python_version()}
    for package in packages:
        try:
            out[package] = version(package)
        except Exception:  # noqa: BLE001 - an optional package may be absent
            out[package] = "not installed"
    return out


def write(rows: Iterable[Row], root: Path) -> None:
    """Merge ``rows`` into the stored results and regenerate the fragments."""
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results.json"
    fresh = _collapse(rows)
    # A module that ran replaces everything previously stored for it, so a renamed or
    # removed test does not linger; modules that did not run keep their last result.
    ran = {row.nodeid.split("::")[0] for row in fresh}
    stored: dict[str, dict[str, Any]] = {}
    if results.exists():
        stored = {
            r["nodeid"]: r
            for r in json.loads(results.read_text())["rows"]
            if r["nodeid"].split("::")[0] not in ran
        }
    for row in fresh:
        stored[row.nodeid] = asdict(row)
    merged = sorted(stored.values(), key=lambda r: (r["dataset"], r["nodeid"]))
    payload = {
        "generated": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "versions": _versions(),
        "rows": merged,
    }
    results.write_text(json.dumps(payload, indent=1) + "\n")
    (root / "summary.inc").write_text(_summary(payload))
    (root / "scorecard.inc").write_text(_scorecard(merged))
    (root / "results.inc").write_text(_results(merged))


def _collapse(rows: Iterable[Row]) -> list[Row]:
    """One row per test function: parametrized cases share a claim and pool their values."""
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.nodeid.split("[")[0], []).append(row)
    out = []
    for nodeid, cases in grouped.items():
        first = cases[0]
        observed = "; ".join(case.observed for case in cases if case.observed)
        out.append(
            Row(
                nodeid=nodeid,
                measures=first.measures,
                kind=first.kind,
                dataset=first.dataset,
                claim=first.claim,
                criterion=first.criterion,
                observed=observed,
                passed=all(case.passed for case in cases),
                run=max(case.run for case in cases),
                cases=len(cases),
            )
        )
    return out


def _summary(payload: Mapping[str, Any]) -> str:
    rows = payload["rows"]
    checks = sum(r.get("cases", 1) for r in rows)
    passed_checks = sum(r.get("cases", 1) for r in rows if r["passed"])
    passed = sum(r["passed"] for r in rows)
    datasets = sorted({r["dataset"] for r in rows if r["dataset"]})
    measures = {m for r in rows for m in r["measures"]}
    v = payload["versions"]
    return (
        f"**{passed} of {len(rows)} claims hold, {passed_checks} of {checks} checks pass** "
        f"across {len(datasets)} public datasets, covering {len(measures)} public functions. "
        f"Last run {payload['generated'][:10]} on "
        f"Python {v['python']}, MNE {v['mne']}, NumPy {v['numpy']}, SciPy {v['scipy']}, "
        f"scikit-learn {v['scikit-learn']}, eegfeat {v['eegfeat']}.\n"
    )


def _cell(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "–"
    failed = [r for r in rows if not r["passed"]]
    if failed:
        return f"✗ {len(failed)} of {len(rows)}"
    return f"✓ {len(rows)}"


def _scorecard(rows: list[dict[str, Any]]) -> str:
    by_measure: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for measure in row["measures"]:
            by_measure.setdefault(measure, []).append(row)

    groups = (
        ("Features", feature_api()),
        ("Modeling", MODEL_API),
        ("Containers, runner and command", TOOLING),
    )
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 35 13 13 13 13 13",
        "",
        "   * - Function",
        *(f"     - {KIND_TITLES[kind]}" for kind in KINDS),
    ]
    untested: list[str] = []
    for title, names in groups:
        lines += [f"   * - **{title}**", *("     - " for _ in KINDS)]
        for name in names:
            tested = by_measure.get(name, [])
            if not tested:
                untested.append(name)
                continue
            lines.append(f"   * - ``{name}``")
            lines += [f"     - {_cell([r for r in tested if r['kind'] == kind])}" for kind in KINDS]
    text = "\n".join(lines) + "\n"
    if untested:
        listed = ", ".join(f"``{name}``" for name in untested)
        text += "\n**Covered by unit tests only, not yet by a real-data check:** " f"{listed}.\n"
    return text


def _escape(text: str) -> str:
    return text.replace("*", "\\*").replace("|", "\\|")


def _results(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        title = DATASET_TITLES.get(dataset, dataset or "Several datasets")
        lines += [title, "^" * len(title), ""]
        lines += [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 30 26 30 14",
            "",
            "   * - Claim",
            "     - Criterion",
            "     - Observed",
            "     - Result",
        ]
        for row in [r for r in rows if r["dataset"] == dataset]:
            functions = ", ".join(f"``{m}``" for m in row["measures"])
            verdict = "pass" if row["passed"] else "**fail**"
            lines += [
                f"   * - {_escape(row['claim'])} ({functions})",
                f"     - {_escape(row['criterion'])}",
                f"     - {_escape(row['observed']) or '–'}",
                f"     - {verdict}",
            ]
        lines.append("")
    return "\n".join(lines)
