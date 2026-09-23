"""Tuning type 33: distributed auto tuning.

Select GPU, count, and remote node for the ablation and tuning pass.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.distributed_auto", "default": False}


@dataclass
class DistributedPlan:
    node: str
    gpus: int
    reason: str


class DistributedAutoTuning:
    def __init__(self, available_nodes: list, enabled: bool = False) -> None:
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
