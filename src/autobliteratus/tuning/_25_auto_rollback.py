"""Tuning type 25: auto rollback.

Revert to the last good marker on guarantee failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


TOGGLE = {"id": "tuning.auto_rollback", "default": True}


@dataclass
class RollbackResult:
    rolled_back: bool
    from_stage: str
    to_marker: str
    reason: str


class AutoRollback:
    def __init__(self, restore_marker: Callable, marker_dir: str, enabled: bool = True) -> None:
        self.restore_marker = restore_marker
        self.marker_dir = Path(marker_dir)
        self.enabled = enabled

    def on_failure(self, from_stage: str, reason: str) -> RollbackResult:
        if not self.enabled:
            return RollbackResult(False, from_stage, "", reason)
        marker = self._last_good_marker_before(from_stage)
        if marker is None:
            return RollbackResult(False, from_stage, "", "no_marker")
        self.restore_marker(marker)
        return RollbackResult(True, from_stage, str(marker), reason)

    def _last_good_marker_before(self, stage: str):
        if not self.marker_dir.exists():
            return None
        markers = sorted(self.marker_dir.glob(f"*-before-{stage}.marker"), reverse=True)
        return markers[0] if markers else None
