"""Read-only diagnostics for low-utilization MoE routers.

This stage measures routing probabilities on bounded representative prompts. It
never changes router weights or expert assignments; automatic rebalancing would
require architecture-specific training and validation and is deliberately not
claimed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TOGGLE = {
    "id": "tuning.anti_router_collapse",
    "default": False,
    "desc": "Measure MoE expert utilization on representative prompts; diagnostic only, no rebalancing.",
}


def _read_probe_texts(context: dict[str, Any]) -> list[str]:
    provided = context.get("routing_probe_texts")
    if provided is not None:
        if not isinstance(provided, (list, tuple)) or not all(isinstance(item, str) for item in provided):
            raise TypeError("routing_probe_texts must be a list of strings.")
        return [item for item in provided if item.strip()]

    data_path = context.get("calibration_data")
    if not data_path:
        return []
    path = Path(data_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Routing calibration data does not exist: {path}")
    raw_text = path.read_text(encoding="utf-8")

    def collect(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, dict):
            text = value.get("text", value.get("prompt"))
            return [text] if isinstance(text, str) and text.strip() else []
        if isinstance(value, list):
            return [text for item in value for text in collect(item)]
        return []

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None
    if payload is not None:
        return collect(payload)

    texts: list[str] = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            texts.append(line)
            continue
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            text = value.get("text", value.get("prompt"))
            if isinstance(text, str):
                texts.append(text)
    return texts


def _extract_tensors(value: Any, torch: Any) -> list[Any]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        result: list[Any] = []
        for child in value.values():
            result.extend(_extract_tensors(child, torch))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            result.extend(_extract_tensors(child, torch))
        return result
    return []


def _entropy_report(observations: dict[str, Any], num_experts: int, threshold: float) -> dict[str, Any]:
    import torch

    if num_experts < 2:
        raise ValueError("Routing analysis requires at least two experts.")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("routing_entropy_threshold must be finite and in [0, 1].")

    layers: dict[str, dict[str, Any]] = {}
    collapsed: list[str] = []
    for layer_name, raw in observations.items():
        values = raw if torch.is_tensor(raw) else torch.as_tensor(raw)
        if values.ndim < 1 or values.shape[-1] != num_experts:
            raise ValueError(f"Router observation {layer_name!r} must end in {num_experts} experts.")
        values = values.detach().to(dtype=torch.float64).reshape(-1, num_experts)
        values = values[torch.isfinite(values).all(dim=-1)]
        if values.numel() == 0:
            continue
        if bool((values < 0).any()):
            probabilities = torch.softmax(values, dim=-1)
        else:
            row_sums = values.sum(dim=-1, keepdim=True)
            if bool((row_sums <= 0).any()):
                probabilities = torch.softmax(values, dim=-1)
            else:
                probabilities = values / row_sums
        load = probabilities.mean(dim=0)
        load = load / load.sum()
        entropy = float((-(load * load.clamp_min(1e-12).log()).sum()).item())
        normalized = entropy / math.log(num_experts)
        layer_result = {
            "routing_entropy_nats": entropy,
            "normalized_entropy": normalized,
            "expert_load": [float(x) for x in load.tolist()],
            "token_observations": int(values.shape[0]),
            "collapsed": normalized < threshold,
        }
        layers[str(layer_name)] = layer_result
        if layer_result["collapsed"]:
            collapsed.append(str(layer_name))

    return {
        "status": "measured" if layers else "no_valid_router_observations",
        "num_experts": num_experts,
        "entropy_threshold": threshold,
        "layers": layers,
        "collapsed_layers": collapsed,
        "model_modified": False,
        "interpretation": "Routing utilization diagnostic only; not a security or quality verdict.",
    }


def _model_device(model: Any, torch: Any) -> Any:
    embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    weight = getattr(embeddings, "weight", None)
    if weight is not None and weight.device.type != "meta":
        return weight.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not find a materialized model input device for router probing.")


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Collect router probabilities on representative texts without changing weights."""
    handle = adapter.handle
    model = handle.model
    num_experts = int(getattr(handle, "num_experts", 0) or 0)
    if num_experts < 2:
        report = {
            "status": "not_applicable",
            "reason": "Model metadata does not declare at least two experts.",
            "model_modified": False,
        }
        context["router_collapse_report"] = report
        return report

    texts = _read_probe_texts(context)
    max_samples = int(context.get("routing_max_samples", 8))
    max_length = int(context.get("routing_max_length", 256))
    if max_samples < 1 or max_length < 2:
        raise ValueError("routing_max_samples must be >=1 and routing_max_length must be >=2.")
    texts = [text[:100_000] for text in texts[:max_samples]]
    if not texts:
        report = {
            "status": "not_evaluated",
            "reason": "Provide routing_probe_texts or calibration_data to measure observed expert utilization.",
            "model_modified": False,
        }
        context["router_collapse_report"] = report
        return report

    import torch

    observations: dict[str, list[Any]] = {}
    hooks = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1].lower()
        if not ("router" in leaf or leaf in {"gate", "router_gate"}):
            continue

        def capture(_module: Any, _inputs: Any, output: Any, module_name: str = name) -> None:
            tensors = [tensor for tensor in _extract_tensors(output, torch) if tensor.ndim >= 2 and tensor.shape[-1] == num_experts]
            if tensors:
                observations.setdefault(module_name, []).append(tensors[0].detach().float().cpu())

        hooks.append(module.register_forward_hook(capture))

    if not hooks:
        report = {
            "status": "unsupported_router_layout",
            "reason": "No router/gate modules with recognized names were found; no weights were changed.",
            "model_modified": False,
        }
        context["router_collapse_report"] = report
        return report

    was_training = bool(model.training)
    model.eval()
    try:
        device = _model_device(model, torch)
        tokenizer = handle.tokenizer
        with torch.inference_mode():
            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
                if "input_ids" not in encoded or encoded["input_ids"].numel() == 0:
                    continue
                model(**encoded)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    combined = {
        name: torch.cat(tensors, dim=0)
        for name, tensors in observations.items()
        if tensors
    }
    report = _entropy_report(
        combined,
        num_experts,
        float(context.get("routing_entropy_threshold", 0.5)),
    )
    report["probe_samples"] = len(texts)
    context["router_collapse_report"] = report
    return report
