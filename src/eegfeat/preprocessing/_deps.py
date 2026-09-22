"""Lazy optional imports with errors limited to the requested missing package."""

from importlib import import_module
from types import ModuleType


def require(package: str, extra: str) -> ModuleType:
    try:
        return import_module(package)
    except ModuleNotFoundError as exc:
        if exc.name != package:
            raise
        raise ModuleNotFoundError(f"Install {package}: pip install 'eegfeat[{extra}]'") from exc
