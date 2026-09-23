"""Tuning type 95: delta-weight export auto tuning.

Produce only the weight delta between base and ablated model.
"""

from __future__ import annotations

from pathlib import Path


TOGGLE = {"id": "tuning.delta_weight_export_auto", "default": True}


class DeltaWeightExportAutoTuning:
    def __init__(self, work_dir: str = "./ablated_model/delta", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def run(self, base_model, ablated_model) -> Path:
        if not self.enabled:
            return Path()
        return self.work_dir / "delta.safetensors"
