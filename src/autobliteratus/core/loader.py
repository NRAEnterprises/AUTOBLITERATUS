"""Model loading and handle construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


FAMILY_TABLE = {
    "llama": "dense_transformer",
    "mistral": "dense_transformer",
    "qwen2": "dense_transformer",
    "qwen3": "dense_transformer",
    "phi": "dense_transformer",
    "gemma": "dense_transformer",
    "gemma2": "dense_transformer",
    "gpt2": "dense_transformer",
    "mixtral": "moe_sparse",
    "deepseek_v2": "moe_shared_routed",
    "deepseek_v3": "moe_shared_routed",
    "mamba": "ssm",
    "rwkv": "rwkv",
    "retnet": "retnet",
    "jamba": "hybrid_attn_ssm",
}


@dataclass
class ModelHandle:
    model: object
    tokenizer: object
    path: str
    architecture: str = "unknown"
    architecture_class: str = "unknown"
    num_layers: int = 0
    hidden_size: int = 0
    num_experts: int = 0
    is_moe: bool = False
    is_multimodal: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"ModelHandle(path={self.path}, arch={self.architecture}, "
            f"layers={self.num_layers}, hidden={self.hidden_size}, "
            f"moe={self.is_moe}, device={self.device}, dtype={self.dtype})"
        )


def load_model(path, device="auto", dtype="auto", trust_remote_code=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)

    if dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    if device == "auto":
        device_map = "auto" if torch.cuda.is_available() else None
    else:
        device_map = device

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    model_type = (getattr(cfg, "model_type", "") or "").lower()
    architecture_class = FAMILY_TABLE.get(model_type, "unknown")

    num_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
        or 0
    )
    hidden_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or getattr(cfg, "d_model", None)
        or 0
    )
    num_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None) or 0
    is_moe = num_experts > 0 or architecture_class in ("moe_sparse", "moe_shared_routed")

    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        path=path,
        architecture=model_type,
        architecture_class=architecture_class,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_experts=num_experts,
        is_moe=is_moe,
        device=str(next(model.parameters()).device),
        dtype=str(torch_dtype),
        config=cfg.to_dict() if hasattr(cfg, "to_dict") else {},
    )
