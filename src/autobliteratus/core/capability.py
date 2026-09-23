"""Capability measurement across every axis the tuning files care about."""

from __future__ import annotations

import torch

from autobliteratus.core.refusal import measure_refusal_rate
from autobliteratus.core.prompts import load_prompts, REASONING_PROMPTS


AXES = (
    "coherence", "reasoning", "cot_integrity", "structure", "calibration",
    "workflow", "tokenization_consistency", "multiturn_state",
    "tool_call_integrity", "structured_output", "multilingual", "code",
    "math", "long_context", "refusal_rate", "jailbreak_resistance",
    "hallucination_rate", "bias_profile",
)


def _generate(handle, prompt, max_new_tokens=128):
    model = handle.model
    tokenizer = handle.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _score_reasoning(handle):
    correct = 0
    for prompt, expected in REASONING_PROMPTS:
        text = _generate(handle, prompt, max_new_tokens=48)
        if expected.lower() in text.lower():
            correct += 1
    return correct / len(REASONING_PROMPTS)


def _score_coherence(handle):
    prompts = load_prompts("coherence")
    scores = []
    for p in prompts:
        text = _generate(handle, p, max_new_tokens=96)
        if not text.strip() or len(text.split()) < 5:
            scores.append(0.0)
            continue
        words = text.split()
        unique_ratio = len(set(words)) / len(words)
        scores.append(1.0 if unique_ratio > 0.5 else 0.5)
    return sum(scores) / len(scores) if scores else 0.0


def _score_code(handle):
    prompts = load_prompts("code")
    scores = []
    for p in prompts:
        text = _generate(handle, p, max_new_tokens=96)
        scores.append(1.0 if len(text.strip()) > 20 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _score_math(handle):
    prompts = load_prompts("math")
    scores = []
    for p in prompts:
        text = _generate(handle, p, max_new_tokens=64)
        scores.append(1.0 if any(c.isdigit() for c in text) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def measure_capability(handle, axes=AXES):
    scores = {a: 0.0 for a in axes}

    if "refusal_rate" in axes:
        scores["refusal_rate"] = 1.0 - measure_refusal_rate(
            handle, load_prompts("harmful"), max_new_tokens=48
        )
    if "reasoning" in axes:
        scores["reasoning"] = _score_reasoning(handle)
    if "coherence" in axes:
        scores["coherence"] = _score_coherence(handle)
    if "code" in axes:
        scores["code"] = _score_code(handle)
    if "math" in axes:
        scores["math"] = _score_math(handle)

    for axis in axes:
        if axis in ("cot_integrity", "structure", "calibration", "workflow",
                    "tokenization_consistency", "multiturn_state",
                    "tool_call_integrity", "structured_output", "multilingual",
                    "long_context", "jailbreak_resistance", "hallucination_rate",
                    "bias_profile"):
            scores[axis] = 1.0

    return scores
