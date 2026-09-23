# abliterate/tuning/25_auto_rollback.py
#
# Tuning Type 25 of 103:
# Auto rollback.
#
# Runtime toggle: --enable tuning.auto_rollback  (default: on)
#
# Effect:
#   Automatic revert to the last good marker on guarantee failure. When the
#   pipeline's guarantee enforcement detects that a guarantee cannot be met,
#   the tool rolls back to the marker at the stage before the failure.
#
# This file owns:
#   - the guarantee check
#   - the rollback execution
#   - the report of what failed and where

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RollbackResult:
    rolled_back: bool
    from_stage: str
    to_marker: str
    reason: str


class AutoRollback:

    def __init__(self, restore_marker: Callable, marker_dir: str,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.auto_rollback", "default": True}
