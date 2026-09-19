"""The ``eegfeat`` command: start a recipe, check it, run it.

Exit status is 0 when everything succeeded, 1 when a run finished but some
recordings failed (or ``check``'s trial recording failed, or it found recordings
that lack channels the recipe names), and 2 when nothing could start: an invalid
recipe, missing inputs, or earlier results in the way.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from eegfeat.runner.batch import CheckReport, RunError, TrialError, check, run
from eegfeat.runner.progress import CHECK, CROSS, JsonReporter, TextReporter
from eegfeat.runner.recipe import RecipeError, load_recipe

_LABEL_WIDTH = 15


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    args = _parser().parse_args(argv)
    status: int = args.handler(args)
    return status


def _parser() -> argparse.ArgumentParser:
    from eegfeat import __version__

    parser = argparse.ArgumentParser(
        prog="eegfeat",
        description="Compute EEG features from preprocessed MNE epochs files, "
        "as a recipe describes.",
    )
    parser.add_argument("--version", action="version", version=f"eegfeat {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    run_parser = commands.add_parser(
        "run", help="compute features for every recording a recipe selects"
    )
    run_parser.add_argument("recipe", type=Path, help="the recipe file")
    run_parser.add_argument(
        "--overwrite", action="store_true", help="replace results from an earlier run"
    )
    _add_n_jobs(run_parser)
    run_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="report progress as one JSON event per line, for a front end",
    )
    run_parser.set_defaults(handler=_run)

    check_parser = commands.add_parser(
        "check", help="validate a recipe and try it on the first recording, writing nothing"
    )
    check_parser.add_argument("recipe", type=Path, help="the recipe file")
    _add_n_jobs(check_parser)
    check_parser.set_defaults(handler=_check)

    init_parser = commands.add_parser("init", help="write a commented recipe to start from")
    init_parser.add_argument(
        "path", type=Path, nargs="?", default=Path("recipe.toml"), help="default: recipe.toml"
    )
    init_parser.set_defaults(handler=_init)
    return parser


def _add_n_jobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        metavar="N",
        help="parallel jobs for MNE filtering and spectral estimation (default 1)",
    )


def _run(args: argparse.Namespace) -> int:
    reporter = JsonReporter() if args.progress_json else None
    try:
        recipe = load_recipe(args.recipe)
        result = run(
            recipe,
            overwrite=args.overwrite,
            n_jobs=args.n_jobs,
            reporter=reporter if reporter is not None else TextReporter(),
        )
    except (RecipeError, RunError, OSError) as exc:
        return _fail(exc, reporter)
    return 0 if result.ok else 1


def _check(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
        report = check(recipe, n_jobs=args.n_jobs)
    except (RecipeError, RunError, OSError) as exc:
        return _fail(exc, None)
    except TrialError as exc:
        print(f"eegfeat: the trial recording failed: {exc}", file=sys.stderr)
        return 1
    for line in _check_lines(recipe.path, len(recipe.features), recipe.output.root, report):
        print(line)
    return 1 if report.missing_channels else 0


def _init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists():
        print(f"eegfeat: error: {path} already exists; choose another path.", file=sys.stderr)
        return 2
    path.write_text(resources.files("eegfeat.runner").joinpath("template.toml").read_text())
    print(f"Wrote {path}. Set inputs.root and output.root, then run: eegfeat check {path}")
    return 0


def _fail(exc: Exception, reporter: JsonReporter | None) -> int:
    code = "recipe" if isinstance(exc, RecipeError) else "setup"
    if reporter is not None:
        reporter.error(code, str(exc))
    print(f"eegfeat: error: {exc}", file=sys.stderr)
    return 2


def _check_lines(
    recipe_path: Path, n_entries: int, output_root: Path, report: CheckReport
) -> list[str]:
    def row(label: str, value: str) -> str:
        return f"  {label:<{_LABEL_WIDTH}}{value}"

    n = len(report.recordings)
    existing = (
        f"{len(report.existing)} result files already there; run needs --overwrite"
        if report.existing
        else "no earlier results"
    )
    lines = [
        f"eegfeat check · {recipe_path}",
        row("Recipe", f"valid · {n_entries} feature entries"),
        row("Inputs", f"{n} recording{'s' if n != 1 else ''}"),
        row("Output", f"{output_root} · {existing}"),
        row(
            "Trial",
            f"{report.trial.label} · {report.n_epochs} epochs · {len(report.channels)} channels",
        ),
    ]
    if report.features.epochs is not None:
        table = report.features.epochs
        lines.append(row("Per epoch", f"{len(table.meta)} features × {table.n_rows} epochs"))
    if report.features.crosstrial is not None:
        table = report.features.crosstrial
        groups = ", ".join(table.row_labels or ())
        lines.append(row("Across trials", f"{len(table.meta)} features × groups {groups}"))
    if report.missing_channels:
        n_failing = len(report.missing_channels)
        counted = "1 recording lacks" if n_failing == 1 else f"{n_failing} recordings lack"
        them = "it" if n_failing == 1 else "them"
        lines.append(f"{CROSS} {counted} channels the recipe names, so a run would fail {them}:")
        for label, problems in report.missing_channels.items():
            lines.extend(f"  {label}: {problem}" for problem in problems)
        return lines
    lines.append(f"{CHECK} Ready: eegfeat run {recipe_path}")
    return lines
