# abliterate/tuning/14_per_layer_hyperparameter_auto_tuning.py
#
# Tuning Type 14 of 103:
# Per layer hyperparameter auto tuning.
#
# Runtime toggle: --enable tuning.per_layer_hyperparameter_auto  (default: on)
#
# Effect:
#   direction index, projection strength, band width — per layer. Not a global
#   value. Every layer has its own.
#
# This file owns:
#   - the per-layer hyperparameter vector
#   - the derivation of that vector from the model's geometry
#
# This file consumes:
#   - the refusal probe (from PROBE, passed in)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerHyperparams:
    index: int
    direction_index: float
    projection_strength: float
    band_width: int


class PerLayerHyperparameterAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def derive(self, refusal_probe: dict) -> list:
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


TOGGLE = {"id": "tuning.per_layer_hyperparameter_auto", "default": True}
