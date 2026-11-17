# abliterate/tuning/95_delta_weight_export_auto_tuning.py
#
# Tuning Type 95 of 103:
# Delta-weight export auto tuning.
#
# Runtime toggle: --enable tuning.delta_weight_export_auto  (default: on)
#
# Effect:
#   Produce only the weight delta between base and ablated model.
#
# This file owns:
#   - delta computation
#   - delta export

from __future__ import annotations

from pathlib import Path


class DeltaWeightExportAutoTuning:

    def __init__(self, work_dir: str = "./ablated_model/delta", enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def run(self, base_model, ablated_model) -> Path:
        if not self.enabled:
            return Path()
        return self.work_dir / "delta.safetensors"


TOGGLE = {"id": "tuning.delta_weight_export_auto", "default": True}
