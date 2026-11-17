"""Representative-data calibration and post-training Quanto quantization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TOGGLE = {
    "id": "tuning.calibrated_quantization",
    "default": False,
    "desc": "Calibrate activation ranges and quantize model weights with Optimum Quanto.",
}


def _read_calibration_texts(path: Path, limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("calibration_samples must be at least 1.")
    if not path.is_file():
        raise FileNotFoundError(f"Calibration data file does not exist: {path}")

    texts: list[str] = []
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as stream:
        if suffix == ".json":
            payload = json.load(stream)
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            for row in rows:
                if isinstance(row, str):
                    text = row
                elif isinstance(row, dict):
                    text = row.get("text") or row.get("prompt")
                else:
                    text = None
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    if len(texts) >= limit:
                        break
        else:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                if suffix == ".jsonl":
                    row = json.loads(line)
                    text = row if isinstance(row, str) else row.get("text") or row.get("prompt") if isinstance(row, dict) else None
                    if not isinstance(text, str):
                        continue
                    line = text.strip()
                texts.append(line)
                if len(texts) >= limit:
                    break

    if not texts:
        raise ValueError(f"No usable text samples found in calibration file: {path}")
    return texts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Quantize the in-memory model after measuring activation ranges.

    Modes ending in ``weight-only`` omit activation quantization. Other modes
    use calibrated int8 activations and require representative text samples.
    """
    mode = str(context.get("quantization", "none"))
    weight_types = {
        "quanto-int8": "qint8",
        "quanto-int4": "qint4",
        "quanto-int8-weight-only": "qint8",
        "quanto-int4-weight-only": "qint4",
    }
    if mode not in weight_types:
        raise ValueError(f"Unsupported calibrated quantization mode: {mode!r}")

    activation_quantized = not mode.endswith("weight-only")
    calibration_path = Path(str(context.get("calibration_data", ""))).expanduser()
    sample_count = int(context.get("calibration_samples", 128))
    max_length = int(context.get("calibration_max_length", 256))
    momentum = float(context.get("calibration_momentum", 0.9))
    streamline = bool(context.get("calibration_streamline", True))
    if activation_quantized and max_length < 2:
        raise ValueError("calibration_max_length must be at least 2.")
    if activation_quantized and not 0.0 <= momentum <= 1.0:
        raise ValueError("calibration_momentum must be in [0, 1].")
    texts = _read_calibration_texts(calibration_path, sample_count) if activation_quantized else []

    try:
        import torch
        from optimum.quanto import (
            Calibration,
            QuantizedModelForCausalLM,
            freeze,
            qint4,
            qint8,
            quantization_map,
            quantize,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Calibrated quantization requires the 'quantization' extra: "
            "install with `pip install -e '.[dev,quantization]'`."
        ) from exc

    model = adapter.handle.model
    tokenizer = adapter.handle.tokenizer
    if "int4" in mode:
        weight_type = qint4
    else:
        weight_type = qint8

    embedding = model.get_input_embeddings()
    device = embedding.weight.device if embedding is not None else model.device
    quantize(
        model,
        weights=weight_type,
        activations=qint8 if activation_quantized else None,
        exclude=["lm_head"],
    )
    model.eval()

    valid_samples = 0
    # Quanto's low-bit tensor subclasses may need normal version counters;
    # no_grad avoids creating inference tensors during activation calibration.
    if activation_quantized:
        with torch.no_grad(), Calibration(momentum=momentum, streamline=streamline):
            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                if encoded["input_ids"].shape[-1] < 2:
                    continue
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
                valid_samples += 1

    if activation_quantized and valid_samples == 0:
        raise ValueError("Calibration samples contained no sequences with at least two tokens.")

    freeze(model)
    qmap = quantization_map(model)
    if not qmap:
        raise RuntimeError(
            "Quanto did not quantize any supported modules; this architecture is not supported by this stage."
        )

    def marked_quantized(value: Any) -> bool:
        return str(value).lower() not in {"none", "null", "false", ""}

    weight_modules = [
        name for name, details in qmap.items()
        if isinstance(details, dict) and marked_quantized(details.get("weights"))
    ]
    activation_modules = [
        name for name, details in qmap.items()
        if isinstance(details, dict) and marked_quantized(details.get("activations"))
    ]
    if not weight_modules:
        raise RuntimeError(
            "Quanto found no supported model weights to quantize for this architecture; "
            "the requested weight-quantization mode cannot be honored."
        )
    expected_weight = "qint4" if "int4" in mode else "qint8"
    wrong_weight_types = [
        name for name in weight_modules
        if str(qmap[name].get("weights")).lower() != expected_weight
    ]
    if wrong_weight_types:
        raise RuntimeError(
            f"Quanto did not apply the requested {expected_weight} weight type to all quantized modules: "
            f"{wrong_weight_types[:8]}"
        )
    if activation_quantized and not activation_modules:
        raise RuntimeError(
            "Quanto found no supported activations to quantize; the requested activation-quantized mode cannot be honored."
        )

    wrapped_model = QuantizedModelForCausalLM(model)
    adapter.handle.model = wrapped_model
    adapter.modules.model = wrapped_model

    report = {
        "backend": "optimum-quanto",
        "mode": mode,
        "weight_type": expected_weight,
        "activation_type": "qint8" if activation_quantized else None,
        "activation_calibration": (
            "representative-text-activation-ranges" if activation_quantized else "not-used-weight-only"
        ),
        "calibration_samples": valid_samples,
        "calibration_max_length": max_length,
        "calibration_momentum": momentum,
        "calibration_streamline": streamline if activation_quantized else None,
        "calibration_file": calibration_path.name if activation_quantized else None,
        "calibration_file_sha256": _file_sha256(calibration_path) if activation_quantized else None,
        "quantized_module_count": len(qmap),
        "weight_quantized_module_count": len(weight_modules),
        "activation_quantized_module_count": len(activation_modules),
        "weight_quantized_modules": weight_modules,
        "activation_quantized_modules": activation_modules,
        "excluded_modules": ["lm_head"],
    }
    context["model_modified"] = True
    context["quantization_report"] = report
    return report
