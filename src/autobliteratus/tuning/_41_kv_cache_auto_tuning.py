"""Tuning type 41: KV cache auto tuning.

The ablation changes attention patterns, so cache size, quantization, and
eviction policies may need adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.kv_cache_auto", "default": True}


@dataclass
class KVCachePlan:
    size: int
    quantization: str
    eviction_policy: str


class KVCacheAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def plan(self, model) -> KVCachePlan:
        if not self.enabled:
            return KVCachePlan(0, "none", "none")
        return KVCachePlan(size=2048, quantization="fp8", eviction_policy="h2o")
