from __future__ import annotations

from importlib.util import find_spec

if find_spec("sklearn") is None:
    raise ModuleNotFoundError(
        "eegfeat.model requires scikit-learn. Install it with: pip install 'eegfeat[model]'"
    )
