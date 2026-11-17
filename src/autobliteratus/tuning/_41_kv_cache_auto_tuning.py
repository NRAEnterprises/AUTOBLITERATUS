"""Bounded KV-cache and sequence-length profiling.

This stage measures cache behavior only. It never changes model weights, cache
configuration, deployment policy, or tokenizer state beyond a temporary pad
value required by the benchmark call.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

TOGGLE = {
    "id": "tuning.kv_cache_auto",
    "default": False,
    "desc": "Profile bounded KV-cache modes and sequence lengths without mutating the model or selecting a policy.",
}


@dataclass
class KVCachePlan:
    size: int
    quantization: str
    eviction_policy: str


class KVCacheAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def plan(self, model) -> KVCachePlan:
        del model
        if not self.enabled:
            return KVCachePlan(0, "none", "none")
        return KVCachePlan(size=0, quantization="unmeasured", eviction_policy="unmeasured")


def _sync(torch, device: Any) -> None:
    backend = getattr(torch, str(device).split(":", 1)[0], None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _resident_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _accelerator_bytes(torch, device: Any) -> int | None:
    device_name = str(device)
    device_type = device_name.split(":", 1)[0]
    if device_type in {"cpu", "meta"}:
        return None
    backend = getattr(torch, device_type, None)
    allocated = getattr(backend, "memory_allocated", None)
    if not callable(allocated) and device_type == "mps":
        allocated = getattr(backend, "current_allocated_memory", None)
        if callable(allocated):
            try:
                return int(allocated())
            except (RuntimeError, TypeError, ValueError):
                return None
    if callable(allocated):
        try:
            return int(allocated(device_name))
        except (RuntimeError, TypeError, ValueError):
            return None
    return None


def _validate_sequence_lengths(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 4:
        raise ValueError("kv_cache_sequence_lengths must contain between 1 and 4 candidates.")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 2048 for item in value):
        raise ValueError("kv_cache_sequence_lengths values must be integers in [1, 2048].")
    if len(set(value)) != len(value):
        raise ValueError("kv_cache_sequence_lengths cannot contain duplicates.")
    return [int(item) for item in value]


def _validate_batch_sizes(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 4:
        raise ValueError("kv_cache_batch_sizes must contain between 1 and 4 candidates.")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 8 for item in value):
        raise ValueError("kv_cache_batch_sizes values must be integers in [1, 8].")
    if len(set(value)) != len(value):
        raise ValueError("kv_cache_batch_sizes cannot contain duplicates.")
    return [int(item) for item in value]


def _validate_modes(value: Any) -> list[bool]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 2:
        raise ValueError("kv_cache_modes must contain one or two boolean modes.")
    if any(not isinstance(item, bool) for item in value):
        raise TypeError("kv_cache_modes values must be booleans.")
    if len(set(value)) != len(value):
        raise ValueError("kv_cache_modes cannot contain duplicates.")
    return [bool(item) for item in value]


def _tensor_from_output(output: Any, torch):
    if torch.is_tensor(output):
        return output
    sequences = getattr(output, "sequences", None)
    if torch.is_tensor(sequences):
        return sequences
    raise TypeError("KV-cache profiling requires model.generate to return a tensor or an object with tensor sequences.")


def _output_drift(reference, candidate, torch) -> float:
    if tuple(reference.shape) != tuple(candidate.shape):
        return math.inf
    reference_cpu = reference.detach().to(device="cpu")
    candidate_cpu = candidate.detach().to(device="cpu")
    if reference_cpu.numel() == 0:
        return 0.0
    if not (reference_cpu.is_floating_point() or reference_cpu.is_complex()):
        return float(torch.max(torch.abs(reference_cpu.to(dtype=torch.int64) - candidate_cpu.to(dtype=torch.int64))).item())
    return float(torch.max(torch.abs(reference_cpu - candidate_cpu)).item())


def _pad_or_trim(encoded: dict[str, Any], sequence_length: int, torch) -> dict[str, Any]:
    input_ids = encoded.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise TypeError("Tokenizer output must contain a rank-2 input_ids tensor.")
    current = int(input_ids.shape[-1])
    if current > sequence_length:
        input_ids = input_ids[:, :sequence_length]
    elif current < sequence_length:
        pad_id = int(encoded.get("pad_token_id", 0)) if not torch.is_tensor(encoded.get("pad_token_id")) else 0
        padding = torch.full(
            (input_ids.shape[0], sequence_length - current),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat((input_ids, padding), dim=-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _validate_prompts(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 8:
        raise ValueError("kv_cache_prompts must contain between 1 and 8 prompts.")
    if not all(isinstance(prompt, str) for prompt in value):
        raise TypeError("kv_cache_prompts must contain only strings.")
    if any(len(prompt) > 10_000 for prompt in value):
        raise ValueError("kv_cache_prompts entries must be at most 10000 characters.")
    return list(value)


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Measure bounded cache modes and sequence lengths on the active model."""
    import torch

    prompts = _validate_prompts(context.get("kv_cache_prompts", ["KV cache profiling fixture prompt."]))
    sequence_lengths = _validate_sequence_lengths(context.get("kv_cache_sequence_lengths", [8, 16, 32]))
    batch_sizes = _validate_batch_sizes(context.get("kv_cache_batch_sizes", [1]))
    modes = _validate_modes(context.get("kv_cache_modes", [False, True]))
    max_new_tokens = context.get("kv_cache_max_new_tokens", 4)
    timeout_seconds = context.get("kv_cache_timeout_seconds")
    cancelled = context.get("kv_cache_cancelled", False)
    tolerance = context.get("kv_cache_output_tolerance", 0.0)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 64:
        raise ValueError("kv_cache_max_new_tokens must be an integer in [1, 64].")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 3600.0
    ):
        raise ValueError("kv_cache_timeout_seconds must be finite and in (0, 3600].")
    if not isinstance(cancelled, bool):
        raise TypeError("kv_cache_cancelled must be a boolean.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or not 0.0 <= float(tolerance) <= 1.0:
        raise ValueError("kv_cache_output_tolerance must be finite and in [0, 1].")
    if len(sequence_lengths) * len(batch_sizes) * len(modes) > 16:
        raise ValueError("KV-cache profiling is limited to 16 candidate combinations.")

    model = adapter.handle.model
    tokenizer = adapter.handle.tokenizer
    device = next((parameter.device for parameter in model.parameters() if parameter.device.type != "meta"), torch.device("cpu"))
    model.eval()
    original_pad = getattr(tokenizer, "pad_token", None)
    if original_pad is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    def encode(batch_size: int, sequence_length: int) -> dict[str, Any]:
        batch = [prompts[index % len(prompts)] for index in range(batch_size)]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
        encoded = {name: value.to(device) for name, value in encoded.items() if torch.is_tensor(value)}
        return _pad_or_trim(encoded, sequence_length, torch)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    quality_failures = 0
    try:
        with torch.inference_mode():
            for sequence_length in sequence_lengths:
                for batch_size in batch_sizes:
                    if cancelled:
                        for use_cache in modes:
                            rows.append(
                                {
                                    "sequence_length": sequence_length,
                                    "batch_size": batch_size,
                                    "use_cache": use_cache,
                                    "status": "cancelled",
                                }
                            )
                        continue
                    inputs = encode(batch_size, sequence_length)
                    baseline = None
                    baseline_started = time.perf_counter()
                    baseline_memory = _resident_bytes()
                    baseline_accelerator = _accelerator_bytes(torch, device)
                    try:
                        baseline_output = _tensor_from_output(
                            model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                use_cache=False,
                            ),
                            torch,
                        )
                        _sync(torch, device)
                        baseline_elapsed = time.perf_counter() - baseline_started
                        baseline = (baseline_output, baseline_elapsed, baseline_memory, baseline_accelerator)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        rows.append(
                            {
                                "sequence_length": sequence_length,
                                "batch_size": batch_size,
                                "use_cache": False,
                                "status": "oom",
                            }
                        )
                        warnings.append(f"Batch size {batch_size}, sequence length {sequence_length} baseline stopped after OOM.")
                        continue
                    for use_cache in modes:
                        if use_cache is False:
                            output, elapsed, memory_before, accelerator_before = baseline
                        else:
                            started = time.perf_counter()
                            memory_before = _resident_bytes()
                            accelerator_before = _accelerator_bytes(torch, device)
                            try:
                                output = _tensor_from_output(
                                    model.generate(
                                        **inputs,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        use_cache=True,
                                    ),
                                    torch,
                                )
                                _sync(torch, device)
                                elapsed = time.perf_counter() - started
                            except RuntimeError as exc:
                                if "out of memory" not in str(exc).lower():
                                    raise
                                empty_cache = getattr(getattr(torch, str(device).split(":", 1)[0], None), "empty_cache", None)
                                if callable(empty_cache):
                                    empty_cache()
                                rows.append(
                                    {
                                        "sequence_length": sequence_length,
                                        "batch_size": batch_size,
                                        "use_cache": True,
                                        "status": "oom",
                                    }
                                )
                                warnings.append(f"Batch size {batch_size}, sequence length {sequence_length} cache mode stopped after OOM.")
                                continue
                        output_drift = _output_drift(baseline[0], output, torch)
                        output_stable = output_drift <= float(tolerance)
                        if not output_stable:
                            quality_failures += 1
                        if timeout_seconds is not None and elapsed > float(timeout_seconds):
                            rows.append(
                                {
                                    "sequence_length": sequence_length,
                                    "batch_size": batch_size,
                                    "use_cache": use_cache,
                                    "status": "timed_out",
                                    "elapsed_seconds": elapsed,
                                }
                            )
                            warnings.append(f"Batch size {batch_size}, sequence length {sequence_length} exceeded its cache timeout.")
                            continue
                        memory_after = _resident_bytes()
                        accelerator_after = _accelerator_bytes(torch, device)
                        generated_tokens = max(0, int(output.shape[-1]) - sequence_length) * batch_size
                        if elapsed <= 0 or not math.isfinite(elapsed):
                            raise ValueError("KV-cache measurement produced an invalid elapsed time.")
                        rows.append(
                            {
                                "sequence_length": sequence_length,
                                "batch_size": batch_size,
                                "use_cache": use_cache,
                                "status": "measured",
                                "elapsed_seconds": elapsed,
                                "latency_ms": elapsed * 1000.0,
                                "generated_tokens": generated_tokens,
                                "tokens_per_second": generated_tokens / elapsed,
                                "host_rss_before_bytes": memory_before,
                                "host_rss_after_bytes": memory_after,
                                "host_rss_delta_bytes": (
                                    memory_after - memory_before
                                    if memory_after is not None and memory_before is not None
                                    else None
                                ),
                                "accelerator_before_bytes": accelerator_before,
                                "accelerator_after_bytes": accelerator_after,
                                "accelerator_delta_bytes": (
                                    accelerator_after - accelerator_before
                                    if accelerator_after is not None and accelerator_before is not None
                                    else None
                                ),
                                "output_drift": output_drift,
                                "output_stable": output_stable,
                            }
                        )
    finally:
        if original_pad is None:
            tokenizer.pad_token = original_pad

    measured = [row for row in rows if row.get("status") == "measured"]
    if not measured:
        status = "cancelled" if cancelled else "no_successful_candidates"
    elif quality_failures:
        status = "quality_regression"
    else:
        status = "measured"
    report = {
        "stage_id": "tuning.kv_cache_auto",
        "schema_version": 1,
        "status": status,
        "device": str(device),
        "sequence_lengths": sequence_lengths,
        "batch_sizes": batch_sizes,
        "cache_modes": modes,
        "candidates": rows,
        "quality_gate": {
            "status": "failed" if quality_failures else ("passed" if measured else "not_evaluated"),
            "output_tolerance": float(tolerance),
            "failures": quality_failures,
        },
        "optimization_performed": False,
        "model_modified": False,
        "artifacts": [],
        "warnings": [
            "KV-cache measurements are runner-, model-, tokenizer-, and thermal-state-specific.",
            *warnings,
        ],
    }
    context["kv_cache_report"] = report
    return report
