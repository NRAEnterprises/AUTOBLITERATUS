"""Tuning type 26: important marker creation for rollbacks.

Write checkpoints at every pipeline stage boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


TOGGLE = {"id": "tuning.marker_creation", "default": True}


class ImportantMarkerCreation:
    def __init__(
        self,
        snapshot_model: Callable,
        marker_dir: str = "./markers",
        enabled: bool = True,
        max_markers: int = 50,
    ) -> None:
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

    def _prune(self) -> None:
        markers = sorted(self.marker_dir.glob("*.marker"))
        for old in markers[:-self.max_markers]:
            old.unlink(missing_ok=True)
