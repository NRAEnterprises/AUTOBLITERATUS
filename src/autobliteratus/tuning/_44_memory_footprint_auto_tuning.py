"""Tuning type 44: memory footprint auto tuning.

VRAM and RAM profile measured and optimized after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.memory_footprint_auto", "default": True}


@dataclass
class MemoryPlan:
    vram_mb: float
    ram_mb: float
    optimized: bool


class MemoryFootprintAutoTuning:
    def __init__(self, measure, optimize, enabled: bool = True) -> None:
        self.measure = measure
        self.optimize = optimize
        self.enabled = enabled

    def run(self, model) -> MemoryPlan:
        if not self.enabled:
            return MemoryPlan(0.0, 0.0, False)
        self.measure(model)
        self.optimize(model)
        after = self.measure(model)
        return MemoryPlan(
            vram_mb=after.get("vram", 0.0),
            ram_mb=after.get("ram", 0.0),
            optimized=True,
        )
