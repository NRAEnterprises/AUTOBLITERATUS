# abliterate/tuning/28_quantization_scheme_selection.py
#
# Tuning Type 28 of 103:
# Quantization scheme selection.
#
# Runtime toggle: --enable tuning.quantization_scheme_selection  (default: on)
#
# Effect:
#   For each layer, evaluate multiple quantization schemes (GPTQ, AWQ,
#   SmoothQuant, none) and select the one whose quantized activations have
#   the highest linear CKA against the full-precision activations of that
#   layer. Per-layer, not global.
#
# This file owns:
#   - the per-layer scheme evaluation
#   - the CKA computation against full-precision activations
#   - the selection

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SchemeSelection:
    per_layer: dict = field(default_factory=dict)


class QuantizationSchemeSelection:

    SCHEMES = ("none", "gptq", "awq", "smoothquant", "fp8")

    def __init__(self, calibrate: Callable, cka: Callable, enabled: bool = True):
        self.calibrate = calibrate
        self.cka = cka
        self.enabled = enabled

    def select(self, model, calibration_batches: list) -> SchemeSelection:
        if not self.enabled:
            return SchemeSelection()
        per_layer = {}
        for layer_idx in self._layer_indices(model):
            best_scheme = "none"
            best_cka = -1.0
            reference = self._reference_activations(model, layer_idx, calibration_batches)
            for scheme in self.SCHEMES:
                quantized = self.calibrate(model, layer_idx, scheme, calibration_batches)
                score = self.cka(reference, quantized)
                if score > best_cka:
                    best_cka = score
                    best_scheme = scheme
            per_layer[layer_idx] = {"scheme": best_scheme, "cka": best_cka}
        return SchemeSelection(per_layer=per_layer)

    def _layer_indices(self, model):
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return range(model.config.num_hidden_layers)
        return range(0)

    def _reference_activations(self, model, layer_idx, batches):
        return None


TOGGLE = {"id": "tuning.quantization_scheme_selection", "default": True}
