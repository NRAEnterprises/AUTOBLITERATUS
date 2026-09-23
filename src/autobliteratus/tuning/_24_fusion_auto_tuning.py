"""Tuning type 24: fusion auto tuning.

Routing weights, layer mixing, attention sharing when two or more ablated
models are combined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.fusion_auto", "default": True}


@dataclass
class FusionConfig:
    routing_weights: dict = field(default_factory=dict)
    layer_mixing: dict = field(default_factory=dict)
    attention_sharing: dict = field(default_factory=dict)


class FusionAutoTuning:
    def __init__(self, evaluate: Callable, enabled: bool = True) -> None:
        self.evaluate = evaluate
        self.enabled = enabled

    def tune(self, source_models: list, target_model) -> FusionConfig:
        if not self.enabled or len(source_models) < 2:
            return FusionConfig()
        return FusionConfig()
