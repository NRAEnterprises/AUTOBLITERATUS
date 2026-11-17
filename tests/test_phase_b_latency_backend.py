from __future__ import annotations

import pytest
import torch

from autobliteratus.tuning._45_latency_auto_tuning import run
from autobliteratus.tuning.toggle_registry import TOGGLES


class _Tokenizer:
    def __call__(self, prompt: str, return_tensors: str):
        return {"input_ids": torch.ones((1, max(2, len(prompt.split()))), dtype=torch.long)}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, max_new_tokens, do_sample):
        assert do_sample is False
        return torch.cat((input_ids, torch.zeros((1, min(3, max_new_tokens)), dtype=torch.long)), dim=-1)


class _Adapter:
    handle = type("Handle", (), {"model": _Model(), "tokenizer": _Tokenizer()})()


def test_phase_b_latency_backend_is_callable_and_opt_in() -> None:
    assert TOGGLES["tuning.latency_auto"]["available"] is True
    assert TOGGLES["tuning.latency_auto"]["default"] is False


def test_latency_backend_returns_bounded_measurement_without_mutation() -> None:
    report = run(_Adapter(), {"latency_prompts": ["hello world"], "latency_warmups": 0})
    assert report["status"] == "measured"
    assert report["sample_count"] == 1
    assert report["model_modified"] is False
    assert report["optimization_performed"] is False


def test_latency_backend_rejects_unbounded_prompt_batch() -> None:
    with pytest.raises(ValueError, match="8 prompts"):
        run(_Adapter(), {"latency_prompts": ["x"] * 9})
