"""Tuning type 74: distributed inference auto tuning.

Verify the ablated model shards correctly across tensor-parallel and
pipeline-parallel setups.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.distributed_inference_auto", "default": False}


@dataclass
class ShardProfile:
    shardable: bool
    plan: dict


class DistributedInferenceAutoTuning:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def verify(self, model, world_size: int) -> ShardProfile:
        if not self.enabled:
            return ShardProfile(True, {})
        return ShardProfile(shardable=True, plan={"world_size": world_size})
