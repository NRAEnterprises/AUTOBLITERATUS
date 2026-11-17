"""Tuning type 88: provenance recording auto tuning.

Record hashes of inputs, outputs, and weights at every stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

TOGGLE = {
    "id": "tuning.provenance_recording_auto",
    "default": False,
    "desc": "Hash the final model state and record source, config, calibration, and active stages in the export.",
}


@dataclass
class ProvenanceEntry:
    stage: str
    inputs_hash: str
    outputs_hash: str
    weights_hash: str


class ProvenanceRecordingAutoTuning:
    def __init__(self, work_dir: str = "./provenance", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.entries: list[ProvenanceEntry] = []

    def record(self, stage: str, inputs, outputs, model) -> None:
        if not self.enabled:
            return
        entry = ProvenanceEntry(
            stage=stage,
            inputs_hash=self._hash(inputs),
            outputs_hash=self._hash(outputs),
            weights_hash=self._hash_model(model),
        )
        self.entries.append(entry)
        with (self.work_dir / "provenance.jsonl").open("a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")

    def _hash(self, value) -> str:
        return hashlib.sha256(repr(value).encode()).hexdigest()

    def _hash_model(self, model) -> str:
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()


def fingerprint_model(model) -> tuple[str, int]:
    """Hash state tensors in bounded chunks, dequantizing custom tensors if needed."""
    import torch

    digest = hashlib.sha256()
    tensor_count = 0
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        value = tensor.detach()
        if value.is_quantized:
            value = value.dequantize()
        dequantize = getattr(value, "dequantize", None)
        if callable(dequantize):
            value = dequantize()
        raw = value.to(device="cpu").contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0, raw.numel(), 4 * 1024 * 1024):
            digest.update(raw[offset : offset + 4 * 1024 * 1024].numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise RuntimeError("Cannot fingerprint a model with an empty state dictionary.")
    return digest.hexdigest(), tensor_count


def run(adapter, context):
    import hashlib
    import json

    model_hash, tensor_count = fingerprint_model(adapter.handle.model)
    config_payload = json.dumps(adapter.handle.config, sort_keys=True, default=str).encode("utf-8")
    calibration_path = context.get("calibration_data")
    calibration_hash = None
    if calibration_path:
        calibration_hash = hashlib.sha256()
        with open(calibration_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                calibration_hash.update(chunk)
        calibration_hash = calibration_hash.hexdigest()
    report = {
        "source_model": str(context.get("model_path")),
        "source_model_revision": context.get("model_revision"),
        "model_class": type(adapter.handle.model).__name__,
        "model_type": adapter.handle.architecture,
        "model_config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "model_state_sha256": model_hash,
        "state_tensor_count": tensor_count,
        "calibration_sha256": calibration_hash,
        "calibration_samples": context.get("calibration_samples"),
        "calibration_max_length": context.get("calibration_max_length"),
        "calibration_momentum": context.get("calibration_momentum"),
        "calibration_streamline": context.get("calibration_streamline"),
        "quantization": context.get("quantization", "none"),
        "target_frameworks": list(context.get("target_frameworks", ["transformers"])),
        "deployment_target": context.get("deployment_target", "server"),
        "device": context.get("device", "auto"),
        "dtype": context.get("dtype", "auto"),
        "enabled_stages": sorted(
            toggle_id
            for toggle_id, enabled in context.get("_resolved_toggles", {}).items()
            if enabled
        ),
    }
    context["provenance_report"] = report
    return report
