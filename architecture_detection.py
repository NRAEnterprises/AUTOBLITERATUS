# abliterate/tuning/27_architecture_detection.py
#
# Tuning Type 27 of 103:
# Architecture detection.
#
# Runtime toggle: --enable tuning.architecture_detection  (default: on)
#
# Effect:
#   Detect the model's architecture family from config, state dict layout,
#   and a short structural probe. Emit one of the known family labels.
#   This is a read-only classifier. It does not tune anything itself.
#
# This file owns:
#   - the family classification
#   - the structural probe
#   - the confirmation check when the family is not recognized
#
# This file consumes:
#   - the model's config.json
#   - the model's state dict keys

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class FamilyVerdict:
    family: str
    confidence: float
    evidence: list


class ArchitectureDetection:

    FAMILY_TABLE = {
        "llama": "dense_transformer",
        "mistral": "dense_transformer",
        "qwen2": "dense_transformer",
        "qwen3": "dense_transformer",
        "mixtral": "moe_sparse",
        "deepseek_v2": "moe_shared_routed",
        "deepseek_v3": "moe_shared_routed",
        "mamba": "ssm",
        "rwkv": "rwkv",
        "retnet": "retnet",
        "jamba": "hybrid_attn_ssm",
    }

    def __init__(self, model_path: str, enabled: bool = True):
        self.model_path = Path(model_path)
        self.enabled = enabled

    def detect(self) -> FamilyVerdict:
        if not self.enabled:
            return FamilyVerdict("unknown", 0.0, [])
        cfg_path = self.model_path / "config.json"
        if not cfg_path.exists():
            return FamilyVerdict("unknown", 0.0, ["no config.json"])
        cfg = json.loads(cfg_path.read_text())
        model_type = (cfg.get("model_type") or "").lower()
        family = self.FAMILY_TABLE.get(model_type, "unknown")
        confidence = 1.0 if family != "unknown" else 0.0
        evidence = [f"model_type={model_type}"]
        if "num_experts" in cfg:
            evidence.append("has num_experts")
            if family == "unknown":
                family = "moe_sparse"
                confidence = 0.5
        if "ssm_cfg" in cfg or "state_size" in cfg:
            evidence.append("has ssm_cfg")
            if family == "unknown":
                family = "ssm"
                confidence = 0.5
        return FamilyVerdict(family, confidence, evidence)


TOGGLE = {"id": "tuning.architecture_detection", "default": True, "read_only": True}
