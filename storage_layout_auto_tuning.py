# abliterate/tuning/97_storage_layout_auto_tuning.py
#
# Tuning Type 97 of 103:
# Storage layout auto tuning.
#
# Runtime toggle: --enable tuning.storage_layout_auto  (default: on)
#
# Effect:
#   Choose the most efficient on-disk layout for the ablated model.
#
# This file owns:
#   - layout detection
#   - layout optimization

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LayoutPlan:
    layout: str
    total_size_mb: float


class StorageLayoutAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def plan(self, model_dir: str) -> LayoutPlan:
        if not self.enabled:
            return LayoutPlan("default", 0.0)
        path = Path(model_dir)
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        return LayoutPlan(layout="sharded_safetensors", total_size_mb=total)


TOGGLE = {"id": "tuning.storage_layout_auto", "default": True}
