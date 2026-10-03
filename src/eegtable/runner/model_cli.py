"""Dependency-free registration for the optional modeling command family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _handle(args: argparse.Namespace) -> int:
    from eegtable.runner.model_command import handle

    return handle(args)


def register(commands: Any) -> None:
    parser = commands.add_parser("model", help="check and run grouped predictive modeling recipes")
    subcommands = parser.add_subparsers(dest="model_command", required=True, metavar="command")
    for name, help_text in (
        ("init", "write a strict YAML modeling recipe"),
        ("check", "validate inputs, feature quality and nested group splits"),
        ("run", "fit nested group-disjoint models and save an auditable result bundle"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("recipe", type=Path, help="the YAML modeling recipe")
        command.set_defaults(handler=_handle)
