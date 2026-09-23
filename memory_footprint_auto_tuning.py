# abliterate/tuning/44_memory_footprint_auto_tuning.py
#
# Tuning Type 44 of 103:
# Memory footprint auto tuning.
#
# Runtime toggle: --enable tuning.memory_footprint_auto  (default: on)
#
# Effect:
#   VRAM and RAM profile measured and optimized after ablation.
#
# This file owns:
#   - memory measurement
#   - memory optimization

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryPlan:
    vram_mb: float
    ram_mb: float
    optimized: bool


class MemoryFootprintAutoTuning:

    def __init__(self, measure, optimize, enabled: bool = True):
        self.measure = measure
        self.optimize = optimize
        self.enabled = enabled

    def run(self, model) -> MemoryPlan:
        if not self.enabled:
            return MemoryPlan(0.0, 0.0, False)
        before = self.measure(model)
        self.optimize(model)
        after = self.measure(model)
        return MemoryPlan(vram_mb=after.get("vram", 0.0),
                          ram_mb=after.get("ram", 0.0), optimized=True)


TOGGLE = {"id": "tuning.memory_footprint_auto", "default": True}
