"""Model loading and handle construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FAMILY_TABLE = {
    "gpt2": "dense_transformer",
    "gpt_neo": "dense_transformer",
    "gpt_neox": "dense_transformer",
    "opt": "dense_transformer",
    "bloom": "dense_transformer",
    "falcon": "dense_transformer",
    "mpt": "dense_transformer",
    "llama": "dense_transformer",
    "mistral": "dense_transformer",
    "qwen2": "dense_transformer",
    "qwen3": "dense_transformer",
    "phi": "dense_transformer",
    "gemma": "dense_transformer",
    "gemma2": "dense_transformer",
    "phi3": "dense_transformer",
    "olmo": "dense_transformer",
    "olmo2": "dense_transformer",
    "stablelm": "dense_transformer",
    "starcoder2": "dense_transformer",
    "mixtral": "moe_sparse",
    "deepseek_v2": "moe_shared_routed",
    "deepseek_v3": "moe_shared_routed",
    "mamba": "ssm",
    "rwkv": "rwkv",
    "retnet": "retnet",
    "jamba": "hybrid_attn_ssm",
}


def _backend_available(torch, device_type: str) -> bool:
    if device_type == "cpu":
        return True
    if device_type == "mps":
        return bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
    backend = getattr(torch, device_type, None)
    available = getattr(backend, "is_available", None)
    return bool(available()) if callable(available) else False


def _resolve_device(torch, requested: str) -> tuple[str, bool]:
    """Resolve a concrete PyTorch device and whether CUDA auto-map is safe."""
    if requested == "auto":
        for candidate in ("cuda", "xpu", "mps", "hpu", "npu"):
            if _backend_available(torch, candidate):
                if candidate == "cuda":
                    return "cuda:0", True
                return candidate, False
        return "cpu", False

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch device {requested!r}.") from exc
    if device.type not in {"cpu", "cuda", "mps", "xpu", "hpu", "npu"}:
        raise ValueError(
            f"Unsupported device type {device.type!r}; supported types are cpu, cuda, mps, xpu, hpu, and npu."
        )
    if not _backend_available(torch, device.type):
        raise RuntimeError(f"Requested {requested!r}, but the {device.type} backend is unavailable.")
    if device.index is not None and device.type in {"cuda", "xpu"}:
        count = getattr(torch, device.type).device_count()
        if device.index >= count:
            raise ValueError(f"Requested {requested!r}, but only {count} {device.type} device(s) are available.")
    concrete = str(device)
    if device.type == "cuda" and device.index is None:
        concrete = "cuda:0"
    return concrete, False


def _model_device_summary(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        devices = sorted({str(value) for value in device_map.values()})
        return devices[0] if len(devices) == 1 else "mixed:" + ",".join(devices)
    return str(next(model.parameters()).device)


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
    revision: str | None = None
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"ModelHandle(path={self.path}, arch={self.architecture}, "
            f"layers={self.num_layers}, hidden={self.hidden_size}, "
            f"moe={self.is_moe}, device={self.device}, dtype={self.dtype})"
        )


def load_model(
    path,
    device="auto",
    dtype="auto",
    trust_remote_code=False,
    device_map=None,
    max_memory=None,
    offload_folder=None,
    revision=None,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    selected_device, use_cuda_auto_map = _resolve_device(torch, device)
    if dtype == "auto":
        torch_dtype = torch.float16 if selected_device.startswith("cuda") else torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose auto, float32, float16, or bfloat16.")
        torch_dtype = dtype_map[dtype]

    local_path = Path(path).expanduser()
    is_local = local_path.exists()
    if not is_local and not revision:
        raise ValueError("Remote Hugging Face model loads require an immutable --model-revision (commit SHA).")
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code, revision=revision)
    if getattr(cfg, "is_encoder_decoder", False):
        raise ValueError(
            "Encoder-decoder models are not supported by the current causal-LM probing pipeline."
        )
    multimodal_fields = [
        name for name in ("vision_config", "audio_config", "video_config")
        if getattr(cfg, name, None) is not None
    ]
    if multimodal_fields:
        raise ValueError(
            "Multimodal model inputs are not supported by the current text-only tokenizer/probe path; "
            f"detected config fields: {multimodal_fields}."
        )

    if device_map is None and (max_memory is not None or offload_folder is not None):
        device_map = "auto"
    if device_map is not None:
        if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            raise ValueError(f"Unsupported device_map strategy {device_map!r}.")
        try:
            import accelerate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Sharded/model-map loading requires Accelerate; install autobliteratus dependencies.") from exc
        resolved_device_map = device_map
    elif use_cuda_auto_map:
        resolved_device_map = "auto"
    elif selected_device.startswith("cuda"):
        resolved_device_map = {"": selected_device}
    else:
        resolved_device_map = None

    if offload_folder is not None and resolved_device_map is None:
        raise ValueError("offload_folder requires an Accelerate device map.")
    if offload_folder is not None:
        Path(offload_folder).expanduser().mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": resolved_device_map,
        "trust_remote_code": trust_remote_code,
    }
    if max_memory is not None:
        if resolved_device_map is None:
            raise ValueError("max_memory requires an Accelerate device map.")
        load_kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        load_kwargs["offload_folder"] = str(Path(offload_folder).expanduser().resolve())

    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs, revision=revision)
    if resolved_device_map is None and selected_device != "cpu":
        model.to(torch.device(selected_device))
    if resolved_device_map is not None and not getattr(model, "hf_device_map", None):
        first_parameter_device = str(next(model.parameters()).device)
        if first_parameter_device != "meta":
            model.hf_device_map = {"": first_parameter_device}
    model.eval()

    model_type = (getattr(cfg, "model_type", "") or "").lower()
    architecture_class = FAMILY_TABLE.get(model_type, "unknown")
    if architecture_class == "unknown":
        if getattr(cfg, "is_encoder_decoder", False):
            architecture_class = "encoder_decoder"
        elif any(getattr(cfg, name, None) is not None for name in ("vision_config", "audio_config")):
            architecture_class = "multimodal"
        elif any(
            getattr(cfg, name, None) is not None
            for name in ("ssm_cfg", "state_size", "mamba_d_state")
        ):
            architecture_class = "state_space_or_hybrid"
        elif any(hasattr(model, name) for name in ("layers", "model", "transformer", "gpt_neox")):
            architecture_class = "transformers_causal_lm_generic"

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
        is_multimodal=any(
            getattr(cfg, name, None) is not None
            for name in ("vision_config", "audio_config", "video_config")
        ),
        device=_model_device_summary(model),
        dtype=str(torch_dtype),
        revision=revision,
        config=cfg.to_dict() if hasattr(cfg, "to_dict") else {},
    )
