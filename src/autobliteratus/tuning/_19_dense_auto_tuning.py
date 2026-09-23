"""Tuning type 19: dense auto tuning.

Layer selection, band width, and method variant for dense transformer models.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.dense_auto", "default": True}


@dataclass
class DensePlan:
    peak_layer: int
    band_start: int
    band_end: int
    method: str


class DenseAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def derive(self, refusal_probe: dict) -> DensePlan:
        if not self.enabled:
            return DensePlan(0, 0, 0, "basic")
        norms = refusal_probe.get("refusal_norms") or []
        peak = max(range(len(norms)), key=lambda i: norms[i]) if norms else 0
        return DensePlan(
            peak_layer=peak,
            band_start=max(0, peak - 2),
            band_end=peak + 2,
            method="advanced",
        )
