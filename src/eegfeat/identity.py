from __future__ import annotations

from typing import Any

import numpy as np

from eegfeat.table import RowId


def epoch_row_ids(source: Any, recording: str, n_rows: int) -> tuple[RowId, ...]:
    """Build stable row identities from an MNE epoch-derived object."""
    if not recording:
        raise ValueError("recording must be a non-empty stable identifier.")
    events = getattr(source, "events", None)
    selection = np.asarray(getattr(source, "selection", np.arange(n_rows)), dtype=int)
    if events is None:
        if n_rows != 1:
            raise ValueError("epoch-derived input must expose events for every row.")
        return ((recording, int(selection[0]), "continuous"),)
    event_array = np.asarray(events)
    if event_array.shape != (n_rows, 3) or selection.shape != (n_rows,):
        raise ValueError("events and selection must identify every epoch row.")
    names = {int(code): str(name) for name, code in source.event_id.items()}
    return tuple(
        (recording, int(epoch), names[int(code)])
        for epoch, code in zip(selection, event_array[:, 2], strict=True)
    )
