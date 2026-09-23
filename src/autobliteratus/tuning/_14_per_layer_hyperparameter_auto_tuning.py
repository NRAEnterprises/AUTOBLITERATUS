"""Tuning type 14: per layer hyperparameter auto tuning.

Direction index, projection strength, band width — per layer.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.per_layer_hyperparameter_auto", "default": True}


@dataclass
class LayerHyperparams:
    index: int
    direction_index: float
    projection_strength: float
    band_width: int


class PerLayerHyperparameterAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def derive(self, refusal_probe: dict) -> list[LayerHyperparams]:
        if not self.enabled:
            return []
        norms = refusal_probe.get("refusal_norms") or []
        out = []
        for i, n in enumerate(norms):
            peak = float(n)
            out.append(LayerHyperparams(
                index=i,
                direction_index=1.0,
                projection_strength=min(1.0, peak),
                band_width=1,
            ))
        return out
