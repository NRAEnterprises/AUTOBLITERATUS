# abliterate/tuning/33_distributed_auto_tuning.py
#
# Tuning Type 33 of 103:
# Distributed auto tuning.
#
# Runtime toggle: --enable tuning.distributed_auto  (default: off)
#
# Effect:
#   Select which GPU, how many, and which remote node to use based on model
#   size and available hardware. Offload the ablation and the tuning pass to
#   a remote node via SSH when local resources are insufficient.
#
# This file owns:
#   - resource discovery
#   - node selection
#   - the remote invocation

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DistributedPlan:
    node: str
    gpus: int
    reason: str


class DistributedAutoTuning:

    def __init__(self, available_nodes: list, enabled: bool = False):
        self.available_nodes = available_nodes
        self.enabled = enabled

    def plan(self, model_size_gb: float, local_vram_gb: float) -> DistributedPlan:
        if not self.enabled:
            return DistributedPlan("local", 0, "disabled")
        if model_size_gb <= local_vram_gb:
            return DistributedPlan("local", 1, "fits locally")
        for node in self.available_nodes:
            if node.get("vram_gb", 0) >= model_size_gb:
                return DistributedPlan(node["name"], node.get("gpus", 1), "fits on remote")
        return DistributedPlan("local", 0, "no node fits")


TOGGLE = {"id": "tuning.distributed_auto", "default": False}
