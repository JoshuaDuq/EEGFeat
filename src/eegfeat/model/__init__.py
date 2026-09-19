from __future__ import annotations

# Imported for its side effect: it reports the missing extra by name, before a submodule
# fails on `import sklearn` with a message that does not say what to install.
from eegfeat.model import _deps as _deps

__all__: list[str] = []
