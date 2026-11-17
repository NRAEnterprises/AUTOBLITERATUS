"""Small, explicit causal-LM evaluation battery shared by acceptance stages."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


class InvalidEvaluationMetric(ValueError):
    """Raised when evaluation has no scored tokens or produces non-finite values."""


def load_evaluation_texts(context: dict[str, Any]) -> list[str]:
    """Resolve explicit evaluation texts or a local txt/JSON/JSONL file."""
    supplied = context.get("evaluation_texts")
    if supplied is not None:
        if not isinstance(supplied, (list, tuple)) or not all(isinstance(x, str) for x in supplied):
            raise TypeError("evaluation_texts must be a list of strings.")
        texts = [value for value in supplied if value.strip()]
    else:
        data_path = context.get("evaluation_data")
        if not data_path:
            raise ValueError(
                "Baseline/regression stages require explicit evaluation_texts or evaluation_data; "
                "calibration samples are not silently reused as an evaluation set."
            )
        path = Path(data_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation data file does not exist: {path}")
        raw = path.read_text(encoding="utf-8")

        def collect(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, dict):
                candidate = value.get("text", value.get("prompt"))
                return [candidate] if isinstance(candidate, str) and candidate.strip() else []
            if isinstance(value, list):
                return [text for item in value for text in collect(item)]
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            texts = collect(payload)
        else:
            texts = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    texts.extend(collect(json.loads(line)))
                except json.JSONDecodeError:
                    texts.append(line)

    max_samples = int(context.get("evaluation_max_samples", 64))
    if max_samples < 1:
        raise ValueError("evaluation_max_samples must be at least 1.")
    texts = [text[:100_000] for text in texts[:max_samples]]
    if not texts:
        raise ValueError("The evaluation battery contains no usable text samples.")
    return texts


def evaluate_causal_lm(model: Any, tokenizer: Any, texts: list[str], max_length: int = 256) -> dict[str, Any]:
    """Measure token-weighted causal NLL, capped perplexity, and forward throughput."""
    import torch
    import torch.nn.functional as F

    if not isinstance(texts, list) or not texts or not all(isinstance(text, str) for text in texts):
        raise TypeError("texts must be a non-empty list of strings.")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("evaluation_max_length must be an integer of at least 2.")

    input_embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    input_weight = getattr(input_embeddings, "weight", None)
    if input_weight is not None and input_weight.device.type != "meta":
        device = input_weight.device
    else:
        device = next((p.device for p in model.parameters() if p.device.type != "meta"), None)
        if device is None:
            raise RuntimeError("No materialized model device is available for evaluation.")

    was_training = bool(model.training)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    forward_seconds = 0.0
    sample_count = 0
    backend = getattr(torch, device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    try:
        with torch.inference_mode():
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = encoded["input_ids"].to(device)
                if input_ids.ndim != 2 or input_ids.shape[-1] < 2:
                    continue
                attention = encoded.get("attention_mask")
                if attention is not None:
                    attention = attention.to(device)
                if callable(synchronize):
                    synchronize()
                started = time.perf_counter()
                outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                if callable(synchronize):
                    synchronize()
                forward_seconds += time.perf_counter() - started
                logits = getattr(outputs, "logits", None)
                if logits is None or logits.ndim != 3:
                    raise TypeError("Evaluation model must return rank-3 causal-LM logits.")
                if not bool(torch.isfinite(logits).all()):
                    raise InvalidEvaluationMetric("Evaluation model produced non-finite logits.")
                shifted_logits = logits[:, :-1, :].contiguous().float()
                labels = input_ids[:, 1:].contiguous()
                if attention is not None:
                    label_mask = attention[:, 1:].contiguous().bool()
                    if not bool(label_mask.any()):
                        continue
                    shifted_logits = shifted_logits[label_mask]
                    labels = labels[label_mask]
                sample_nll = F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                )
                nll_value = float(sample_nll.item())
                if not math.isfinite(nll_value):
                    raise InvalidEvaluationMetric("Evaluation produced a non-finite negative log-likelihood.")
                total_nll += nll_value
                total_tokens += int(labels.numel())
                sample_count += 1
    finally:
        model.train(was_training)

    if total_tokens == 0 or not math.isfinite(total_nll):
        raise InvalidEvaluationMetric("Evaluation produced no finite scored tokens.")
    if forward_seconds <= 0 or not math.isfinite(forward_seconds):
        raise InvalidEvaluationMetric("Evaluation produced an invalid forward-time measurement.")
    mean_nll = total_nll / total_tokens
    perplexity = math.exp(min(mean_nll, 80.0))
    throughput = total_tokens / forward_seconds
    metrics = {
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "forward_tokens_per_second": throughput,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise InvalidEvaluationMetric("Evaluation metrics contain a non-finite value.")
    return {
        "metrics": metrics,
        "sample_count": sample_count,
        "scored_tokens": total_tokens,
        "forward_seconds": forward_seconds,
        "perplexity_cap_nll": 80.0,
    }
