# abliterate/tuning/02_architecture_aware_capability_tuning.py
#
# Tuning Type 2 of 103:
# Architecture aware capability tuning.
#
# Runtime toggle: --enable tuning.architecture_aware_capability  (default: on)
#
# Effect:
#   Same measure/correct/re-measure loop as tuning type 1, but layer selection
#   and correction dispatch are chosen by the detected model family. Dense,
#   MoE, SSM, hybrid, and multimodal each take a different path.
#
# This file owns:
#   - family-to-path dispatch
#   - layer selection rule per family
#   - correction target components per family
#
# This file consumes:
#   - family classification (from item 26, passed in)
#   - measure callable (from tuning/06)
#   - apply_correction callable (from correction-primitive files)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TuningTarget:
    axis: str
    baseline: float
    current: float
    deficit: float
    tolerance: float


@dataclass
class TuningReport:
    enabled: bool
    family: str
    iterations: int
    restored: dict = field(default_factory=dict)
    unrecovered: dict = field(default_factory=dict)
    ceiling_hit: bool = False


class ArchitectureAwareCapabilityTuning:

    FAMILY_PATHS = {
        "dense_transformer":          ("peak_band",           ("attn.out", "mlp.down")),
        "moe_sparse":                 ("per_expert_peak",     ("attn.out", "moe.expert")),
        "moe_shared_routed":          ("per_expert_peak",     ("attn.out", "moe.shared", "moe.routed")),
        "ssm":                        ("recurrent_state_band", ("ssm.out",)),
        "hybrid_attn_ssm":            ("split_band",          ("attn.out", "ssm.out")),
        "rwkv":                       ("recurrent_state_band", ("rwkv.out",)),
        "retnet":                     ("recurrent_state_band", ("retnet.out",)),
        "multimodal_vision_language": ("per_tower_band",      ("attn.out", "mlp.down", "vision.tower")),
        "multimodal_audio_language":  ("per_tower_band",      ("attn.out", "mlp.down", "audio.tower")),
    }

    def __init__(self, measure: Callable, apply_correction: Callable, family: str,
                 enabled: bool = True, max_iterations: int = 20, tolerance_ratio: float = 0.02):
        self.measure = measure
        self.apply_correction = apply_correction
        self.family = family
        self.enabled = enabled
        self.max_iterations = max_iterations
        self.tolerance_ratio = tolerance_ratio

    def run(self, model, tokenizer, pre_ablation_baseline: dict) -> TuningReport:
        if not self.enabled:
            return TuningReport(enabled=False, family=self.family, iterations=0)

        path = self.FAMILY_PATHS.get(self.family)
        if path is None:
            raise ValueError(f"Unknown model family: {self.family}")
        layer_rule, components = path

        current = self.measure(model, tokenizer)
        targets = self._build_targets(pre_ablation_baseline, current)

        restored = {}
        iteration = 0
        for iteration in range(self.max_iterations):
            active = [t for t in targets if t.axis not in restored]
            if not active:
                break
            for target in active:
                idx, strength = self._select_layer(model, target, layer_rule)
                for component in components:
                    self.apply_correction(
                        model=model,
                        layer_idx=idx,
                        component=component,
                        axis=target.axis,
                        deficit=target.deficit,
                        strength=strength,
                        family=self.family,
                    )
            current = self.measure(model, tokenizer)
            for target in active:
                new_value = current.get(target.axis, 0.0)
                target.current = new_value
                target.deficit = max(0.0, target.baseline - new_value)
                if target.deficit <= target.tolerance:
                    restored[target.axis] = new_value

        unrecovered = {
            t.axis: {"baseline": t.baseline, "final": t.current, "deficit": t.deficit}
            for t in targets if t.axis not in restored
        }
        return TuningReport(
            enabled=True, family=self.family, iterations=iteration + 1,
            restored=restored, unrecovered=unrecovered, ceiling_hit=bool(unrecovered),
        )

    def _select_layer(self, model, target, rule):
        num_layers = self._num_layers(model)
        strength = min(1.0, target.deficit / max(target.baseline, 1e-6))
        if rule == "peak_band":
            return num_layers // 2, strength
        if rule == "per_expert_peak":
            return num_layers // 2, strength
        if rule == "recurrent_state_band":
            return num_layers // 3, strength
        if rule == "split_band":
            return num_layers // 2, strength
        if rule == "per_tower_band":
            return num_layers // 2, strength
        return 0, strength

    def _build_targets(self, baseline: dict, current: dict) -> list:
        out = []
        for axis, base_value in baseline.items():
            current_value = current.get(axis, 0.0)
            out.append(TuningTarget(
                axis=axis, baseline=base_value, current=current_value,
                deficit=max(0.0, base_value - current_value),
                tolerance=base_value * self.tolerance_ratio,
            ))
        return out

    def _num_layers(self, model) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        return 0


TOGGLE = {"id": "tuning.architecture_aware_capability", "default": True, "family_source": "item_26"}
