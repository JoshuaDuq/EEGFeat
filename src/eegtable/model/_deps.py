from __future__ import annotations

from importlib.util import find_spec

if find_spec("sklearn") is None:
    raise ModuleNotFoundError(
        "eegtable.model requires scikit-learn. Install it with: pip install 'eegtable[model]'"
    )


def has_shap() -> bool:
    return find_spec("shap") is not None


def require_shap() -> None:
    if not has_shap():
        raise ModuleNotFoundError(
            "SHAP is required for SHAP-based feature importance. "
            "Install it with: pip install shap"
        )
