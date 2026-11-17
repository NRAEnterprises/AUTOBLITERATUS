"""Record execution environment details for reproducible artifact comparisons."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from typing import Any

TOGGLE = {
    "id": "tuning.reproducibility_auto",
    "default": False,
    "desc": "Record package/runtime/device and deterministic-algorithm settings; does not set seeds or promise bitwise reproducibility.",
}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    import torch

    model = adapter.handle.model
    config = getattr(model, "config", None)
    deterministic = bool(torch.are_deterministic_algorithms_enabled())
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "optimum-quanto", "safetensors", "accelerate")
        },
        "selected_device": context.get("device", "auto"),
        "loaded_device_summary": getattr(adapter.handle, "device", None),
        "dtype": context.get("dtype", "auto"),
        "architecture": getattr(config, "model_type", None),
        "model_config_class": type(config).__name__ if config is not None else None,
        "deterministic_algorithms_enabled": deterministic,
        "random_seed": None,
        "bitwise_reproducibility_claimed": False,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA", "HIP", "TORCH", "PYTORCH", "CUBLAS", "CUDNN", "NCCL", "OMP", "MKL"))
        },
    }
    report = {"runtime": runtime}
    context["reproducibility_report"] = report
    return report
