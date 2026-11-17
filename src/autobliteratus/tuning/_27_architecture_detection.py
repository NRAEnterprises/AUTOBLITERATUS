"""Tuning type 27: architecture detection.

Read-only classifier. Detects the model's family from config and state dict
layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

TOGGLE = {
    "id": "tuning.architecture_detection",
    "default": True,
    "desc": "Inspect live Transformers model config, decoder blocks, and eligible linear modules.",
}


@dataclass
class FamilyVerdict:
    family: str
    confidence: float
    evidence: list


class ArchitectureDetection:
    FAMILY_TABLE: ClassVar[dict[str, str]] = {
        "llama":       "dense_transformer",
        "gpt2":        "dense_transformer",
        "gpt_neo":     "dense_transformer",
        "gpt_neox":    "dense_transformer",
        "opt":         "dense_transformer",
        "bloom":       "dense_transformer",
        "falcon":      "dense_transformer",
        "mpt":         "dense_transformer",
        "mistral":     "dense_transformer",
        "gemma":       "dense_transformer",
        "gemma2":      "dense_transformer",
        "phi":         "dense_transformer",
        "phi3":        "dense_transformer",
        "olmo":        "dense_transformer",
        "olmo2":       "dense_transformer",
        "stablelm":    "dense_transformer",
        "starcoder2":  "dense_transformer",
        "qwen2":       "dense_transformer",
        "qwen3":       "dense_transformer",
        "mixtral":     "moe_sparse",
        "deepseek_v2": "moe_shared_routed",
        "deepseek_v3": "moe_shared_routed",
        "mamba":       "ssm",
        "rwkv":        "rwkv",
        "retnet":      "retnet",
        "jamba":       "hybrid_attn_ssm",
    }

    def __init__(self, model_path: str, enabled: bool = True) -> None:
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


def run(adapter, context):
    """Inspect the actual loaded model without mutating it or promising support."""
    import torch

    handle = adapter.handle
    model = handle.model
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "unknown"))
    layers = adapter.modules.count()
    experts = int(
        getattr(config, "num_local_experts", None)
        or getattr(config, "num_experts", None)
        or 0
    )
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    has_multimodal_config = any(
        getattr(config, name, None) is not None
        for name in ("vision_config", "audio_config", "video_config")
    )
    if has_multimodal_config:
        family = "multimodal-causal-lm"
    elif experts:
        family = "mixture-of-experts-causal-lm"
    elif is_encoder_decoder:
        family = "encoder-decoder-generation-model"
    elif layers:
        family = "decoder-only-causal-lm"
    else:
        family = "unrecognized-transformers-causal-lm-block-layout"

    first_parameter = next(model.parameters())
    report = {
        "model_type": model_type,
        "architectures": list(getattr(config, "architectures", None) or []),
        "family": family,
        "block_count": layers,
        "hidden_size": int(
            getattr(config, "hidden_size", None)
            or getattr(config, "n_embd", None)
            or getattr(config, "d_model", None)
            or 0
        ),
        "linear_module_count": sum(
            isinstance(module, torch.nn.Linear) for module in model.modules()
        ),
        "expert_count_from_config": experts,
        "is_encoder_decoder": is_encoder_decoder,
        "has_multimodal_config": has_multimodal_config,
        "model_class": type(model).__name__,
        "device": str(first_parameter.device),
        "dtype": str(first_parameter.dtype),
        "quantization_support": "not-probed; conversion and reload must pass for the selected backend",
        "ablation_support": "requires a recognized transformer block and compatible projection modules",
    }
    context["architecture_report"] = report
    return report
