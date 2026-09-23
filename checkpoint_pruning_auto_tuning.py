# abliterate/tuning/96_checkpoint_pruning_auto_tuning.py
#
# Tuning Type 96 of 103:
# Checkpoint pruning auto tuning.
#
# Runtime toggle: --enable tuning.checkpoint_pruning_auto  (default: on)
#
# Effect:
#   Remove intermediate checkpoints that are no longer needed.
#
# This file owns:
#   - checkpoint inventory
#   - retention policy
#   - deletion

from __future__ import annotations

from pathlib import Path


class CheckpointPruningAutoTuning:

    def __init__(self, work_dir: str = "./markers", enabled: bool = True,
                 keep_last: int = 5):
        self.work_dir = Path(work_dir)
        self.enabled = enabled
        self.keep_last = keep_last

    def run(self) -> list:
        if not self.enabled:
            return []
        markers = sorted(self.work_dir.glob("*.marker"))
        deleted = []
        for m in markers[:-self.keep_last]:
            m.unlink(missing_ok=True)
            deleted.append(str(m))
        return deleted


TOGGLE = {"id": "tuning.checkpoint_pruning_auto", "default": True}
