"""Tuning type 71: KV quantization auto tuning.

Quantize the KV cache separately from the weights and verify capability
retention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.kv_quantization_auto", "default": True}


@dataclass
class KVQuantPlan:
    scheme: str
    capability_retained: bool


class KVQuantizationAutoTuning:
    SCHEMES = ("none", "fp8", "int8", "int4")

    def __init__(self, verify: Callable, enabled: bool = True) -> None:
        self.verify = verify
        self.enabled = enabled

    def choose(self, model, tokenizer) -> KVQuantPlan:
        if not self.enabled:
            return KVQuantPlan("none", True)
        best_scheme = "none"
        for scheme in self.SCHEMES:
            if self.verify(model, tokenizer, scheme):
                best_scheme = scheme
        return KVQuantPlan(best_scheme, self.verify(model, tokenizer, best_scheme))
