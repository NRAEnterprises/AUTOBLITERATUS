# abliterate/tuning/41_kv_cache_auto_tuning.py
#
# Tuning Type 41 of 103:
# KV cache auto tuning.
#
# Runtime toggle: --enable tuning.kv_cache_auto  (default: on)
#
# Effect:
#   The ablation changes attention patterns, so cache size, quantization, and
#   eviction policies may need adjustment.
#
# This file owns:
#   - cache size measurement against ablation
#   - cache quantization choice
#   - eviction policy choice

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCachePlan:
    size: int
    quantization: str
    eviction_policy: str


class KVCacheAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def plan(self, model) -> KVCachePlan:
        if not self.enabled:
            return KVCachePlan(0, "none", "none")
        return KVCachePlan(size=2048, quantization="fp8", eviction_policy="h2o")


TOGGLE = {"id": "tuning.kv_cache_auto", "default": True}
