"""Bounded batch-size throughput profiling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

TOGGLE = {
    "id": "tuning.batch_size_throughput_auto",
    "default": False,
    "desc": "Profile bounded batch-size throughput without mutating the model or selecting a deployment policy.",
}


@dataclass
class ThroughputPlan:
    batch_size: int
    sequence_packing: bool
    tokens_per_second: float


class BatchSizeThroughputAutoTuning:
    def __init__(self, measure, enabled: bool = True) -> None:
        self.measure = measure
        self.enabled = enabled

    def plan(self, model, tokenizer) -> ThroughputPlan:
        if not self.enabled:
            return ThroughputPlan(1, False, 0.0)
        rate = self.measure(model, tokenizer)
        return ThroughputPlan(batch_size=8, sequence_packing=True, tokens_per_second=rate)


def _sync(torch, device: Any) -> None:
    backend = getattr(torch, str(device).split(":", 1)[0], None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Measure a bounded batch-size throughput sweep on the active model/device."""
    import torch

    prompts = context.get("throughput_prompts", ["Summarize this throughput profiling workload."])
    candidates = context.get("throughput_batch_sizes", [1, 2, 4])
    max_new_tokens = context.get("throughput_max_new_tokens", 16)
    warmups = context.get("throughput_warmups", 0)
    timeout_seconds = context.get("throughput_timeout_seconds")
    cancelled = context.get("throughput_cancelled", False)
    if not isinstance(prompts, (list, tuple)) or not prompts or not all(isinstance(item, str) for item in prompts):
        raise TypeError("throughput_prompts must be a non-empty list of strings.")
    if len(prompts) > 8 or any(len(item) > 10_000 for item in prompts):
        raise ValueError("throughput_prompts is limited to 8 prompts of 10000 characters each.")
    if not isinstance(candidates, (list, tuple)) or not candidates or len(candidates) > 4:
        raise ValueError("throughput_batch_sizes must contain between 1 and 4 candidates.")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8 for value in candidates):
        raise ValueError("throughput_batch_sizes values must be integers in [1, 8].")
    if len(set(candidates)) != len(candidates):
        raise ValueError("throughput_batch_sizes cannot contain duplicates.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
        raise ValueError("throughput_max_new_tokens must be an integer in [1, 128].")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or not 0 <= warmups <= 1:
        raise ValueError("throughput_warmups must be an integer in [0, 1].")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 3600.0
    ):
        raise ValueError("throughput_timeout_seconds must be finite and in (0, 3600].")
    if not isinstance(cancelled, bool):
        raise TypeError("throughput_cancelled must be a boolean.")
    model = adapter.handle.model
    tokenizer = adapter.handle.tokenizer
    device = next((parameter.device for parameter in model.parameters() if parameter.device.type != "meta"), torch.device("cpu"))
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    def encode(batch: list[str]) -> dict[str, Any]:
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        return {name: value.to(device) for name, value in encoded.items()}

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch_size in candidates:
            if cancelled:
                rows.append({"batch_size": batch_size, "status": "cancelled"})
                continue
            batch = [prompts[index % len(prompts)] for index in range(batch_size)]
            try:
                for _ in range(warmups):
                    model.generate(**encode(batch), max_new_tokens=max_new_tokens, do_sample=False)
                inputs = encode(batch)
                _sync(torch, device)
                started = time.perf_counter()
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                _sync(torch, device)
                elapsed = time.perf_counter() - started
                if timeout_seconds is not None and elapsed > float(timeout_seconds):
                    rows.append({
                        "batch_size": batch_size,
                        "status": "timed_out",
                        "sequence_length": int(inputs["input_ids"].shape[-1]),
                        "elapsed_seconds": elapsed,
                    })
                    warnings.append(f"Batch size {batch_size} exceeded its throughput timeout.")
                    continue
                generated_tokens = max(0, int(generated.shape[-1]) - int(inputs["input_ids"].shape[-1])) * batch_size
                if elapsed <= 0 or not math.isfinite(elapsed):
                    raise ValueError("Throughput measurement produced an invalid elapsed time.")
                rows.append({
                    "batch_size": batch_size,
                    "status": "measured",
                    "sequence_length": int(inputs["input_ids"].shape[-1]),
                    "elapsed_seconds": elapsed,
                    "generated_tokens": generated_tokens,
                    "tokens_per_second": generated_tokens / elapsed,
                })
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                backend = getattr(torch, str(device).split(":", 1)[0], None)
                empty_cache = getattr(backend, "empty_cache", None)
                if not callable(empty_cache):
                    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
                rows.append({"batch_size": batch_size, "status": "oom"})
                warnings.append(f"Batch size {batch_size} was stopped after an out-of-memory failure.")
    measured = [row for row in rows if row.get("status") == "measured"]
    best = max(measured, key=lambda row: row["tokens_per_second"], default=None)
    status = "measured" if best is not None else ("cancelled" if cancelled else "no_successful_candidates")
    report = {
        "stage_id": "tuning.batch_size_throughput_auto",
        "schema_version": 1,
        "status": status,
        "device": str(device),
        "candidates": rows,
        "selected_batch_size": best["batch_size"] if best else None,
        "selected_tokens_per_second": best["tokens_per_second"] if best else None,
        "optimization_performed": False,
        "model_modified": False,
        "artifacts": [],
        "warnings": [
            "Throughput is runner-, workload-, model-, and thermal-state-specific.",
            *warnings,
        ],
    }
    context["throughput_report"] = report
    return report
