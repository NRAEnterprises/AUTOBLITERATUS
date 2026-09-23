# abliterate/tuning/26_important_marker_creation_for_rollbacks.py
#
# Tuning Type 26 of 103:
# Important marker creation for rollbacks.
#
# Runtime toggle: --enable tuning.marker_creation  (default: on)
#
# Effect:
#   Write checkpoints at every pipeline stage (pre-ablation baseline,
#   post-measurement, post-EXCISE, post-correction-iteration-N,
#   post-quantization) so rollback can target any specific stage.
#
# This file owns:
#   - the marker write at each stage boundary
#   - the marker naming scheme (stage-before-X, stage-after-X)
#   - the pruning of stale markers

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Callable


class ImportantMarkerCreation:

    def __init__(self, snapshot_model: Callable, marker_dir: str = "./markers",
                 enabled: bool = True, max_markers: int = 50):
        self.snapshot_model = snapshot_model
        self.marker_dir = Path(marker_dir)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.max_markers = max_markers

    def write(self, stage: str, position: str, model) -> Path:
        if not self.enabled:
            return Path()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        marker = self.marker_dir / f"{ts}-{position}-{stage}.marker"
        self.snapshot_model(model, marker)
        self._prune()
        return marker

    def _prune(self):
        markers = sorted(self.marker_dir.glob("*.marker"))
        for old in markers[:-self.max_markers]:
            old.unlink(missing_ok=True)


TOGGLE = {"id": "tuning.marker_creation", "default": True}
