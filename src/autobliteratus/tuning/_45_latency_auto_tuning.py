"""Bounded, diagnostic generation-latency benchmark."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

TOGGLE = {
    "id": "tuning.latency_auto",
    "default": False,
    "desc": "Measure bounded first-pass generation latency and throughput without mutating the model.",
}


@dataclass
class LatencyPlan:
    first_token_ms: float
    per_token_ms: float
    optimized: bool


class LatencyAutoTuning:
    def __init__(self, measure, optimize, enabled: bool = True) -> None:
        self.measure = measure
        self.optimize = optimize
        self.enabled = enabled

    def run(self, model, tokenizer) -> LatencyPlan:
        if not self.enabled:
            return LatencyPlan(0.0, 0.0, False)
        self.measure(model, tokenizer)
        self.optimize(model)
        after = self.measure(model, tokenizer)
        return LatencyPlan(
            first_token_ms=after.get("first_token_ms", 0.0),
            per_token_ms=after.get("per_token_ms", 0.0),
            optimized=True,
        )


def _sync(torch, device: Any) -> None:
    backend = getattr(torch, str(device).split(":", 1)[0], None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Measure deterministic generation latency on the active model/device."""
    import torch

    prompts = context.get("latency_prompts", ["Explain the purpose of this benchmark in one sentence."])
    if not isinstance(prompts, (list, tuple)) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise TypeError("latency_prompts must be a non-empty list of strings.")
    if len(prompts) > 8 or any(len(prompt) > 10_000 for prompt in prompts):
        raise ValueError("latency_prompts is limited to 8 prompts of 10000 characters each.")
    max_new_tokens = context.get("latency_max_new_tokens", 32)
    warmups = context.get("latency_warmups", 1)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 256:
        raise ValueError("latency_max_new_tokens must be an integer in [1, 256].")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or not 0 <= warmups <= 2:
        raise ValueError("latency_warmups must be an integer in [0, 2].")
    model = adapter.handle.model
    tokenizer = adapter.handle.tokenizer
    device = next((parameter.device for parameter in model.parameters() if parameter.device.type != "meta"), torch.device("cpu"))

    def encode(prompt: str) -> dict[str, Any]:
        encoded = tokenizer(prompt, return_tensors="pt")
        return {name: value.to(device) for name, value in encoded.items()}

    model.eval()
    with torch.inference_mode():
        for prompt in list(prompts)[:warmups]:
            model.generate(**encode(prompt), max_new_tokens=max_new_tokens, do_sample=False)
        samples: list[dict[str, float]] = []
        for prompt in prompts:
            inputs = encode(prompt)
            input_tokens = int(inputs["input_ids"].shape[-1])
            _sync(torch, device)
            started = time.perf_counter()
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            _sync(torch, device)
            elapsed = time.perf_counter() - started
            generated_tokens = max(0, int(generated.shape[-1]) - input_tokens)
            if elapsed <= 0 or not math.isfinite(elapsed):
                raise ValueError("Latency measurement produced an invalid elapsed time.")
            samples.append(
                {
                    "elapsed_seconds": elapsed,
                    "generated_tokens": float(generated_tokens),
                    "batch_size": 1,
                    "sequence_length": input_tokens,
                }
            )
    median_seconds = statistics.median(sample["elapsed_seconds"] for sample in samples)
    median_tokens = statistics.median(sample["generated_tokens"] for sample in samples)
    tokens_per_second = median_tokens / median_seconds if median_seconds else 0.0
    report = {
        "stage_id": "tuning.latency_auto",
        "schema_version": 1,
        "status": "measured",
        "device": str(device),
        "sample_count": len(samples),
        "median_latency_ms": median_seconds * 1000.0,
        "median_generated_tokens": median_tokens,
        "median_tokens_per_second": tokens_per_second,
        "sequence_length": max(int(sample["sequence_length"]) for sample in samples),
        "optimization_performed": False,
        "model_modified": False,
        "artifacts": [],
        "warnings": ["Latency is runner-, model-, prompt-, and thermal-state-specific."],
    }
    context["latency_report"] = report
    return report
