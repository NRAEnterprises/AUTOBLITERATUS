from __future__ import annotations

import pytest
import torch

from autobliteratus.tuning._43_batch_size_throughput_auto_tuning import run
from autobliteratus.tuning.toggle_registry import TOGGLES


class _Tokenizer:
    pad_token = "[PAD]"
    eos_token = "[EOS]"

    def __call__(self, prompts, return_tensors, padding, truncation):
        if isinstance(prompts, str):
            prompts = [prompts]
        return {"input_ids": torch.ones((len(prompts), 3), dtype=torch.long)}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, max_new_tokens, do_sample):
        assert do_sample is False
        return torch.cat((input_ids, torch.zeros((input_ids.shape[0], 2), dtype=torch.long)), dim=-1)


class _Adapter:
    handle = type("Handle", (), {"model": _Model(), "tokenizer": _Tokenizer()})()


def test_phase_c_throughput_backend_is_callable_and_opt_in() -> None:
    assert TOGGLES["tuning.batch_size_throughput_auto"]["available"] is True
    assert TOGGLES["tuning.batch_size_throughput_auto"]["default"] is False


def test_throughput_backend_reports_bounded_candidates_without_mutation() -> None:
    context = {"throughput_batch_sizes": [1, 2], "throughput_warmups": 0}
    report = run(_Adapter(), context)
    assert report["status"] == "measured"
    assert len(report["candidates"]) == 2
    assert report["selected_batch_size"] in {1, 2}
    assert report["optimization_performed"] is False
    assert report["model_modified"] is False
    assert context["throughput_report"] is report


def test_throughput_backend_rejects_unbounded_candidates() -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        run(_Adapter(), {"throughput_batch_sizes": [1, 2, 3, 4, 5]})


def test_throughput_backend_records_cancellation_without_running_candidates() -> None:
    report = run(_Adapter(), {"throughput_batch_sizes": [1, 2], "throughput_cancelled": True})

    assert report["status"] == "cancelled"
    assert [row["status"] for row in report["candidates"]] == ["cancelled", "cancelled"]
    assert report["selected_batch_size"] is None
    assert report["model_modified"] is False


def test_throughput_backend_records_timeout_without_selecting_candidate(monkeypatch) -> None:
    clock = iter((0.0, 0.1))
    monkeypatch.setattr(
        "autobliteratus.tuning._43_batch_size_throughput_auto_tuning.time.perf_counter",
        lambda: next(clock),
    )
    report = run(_Adapter(), {"throughput_batch_sizes": [1], "throughput_timeout_seconds": 0.001})

    assert report["status"] == "no_successful_candidates"
    assert report["candidates"][0]["status"] == "timed_out"
    assert report["selected_batch_size"] is None


def test_throughput_backend_recovers_from_out_of_memory(monkeypatch) -> None:
    class _OOMModel(_Model):
        def generate(self, input_ids, max_new_tokens, do_sample):
            raise RuntimeError("CUDA out of memory")

    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    adapter = type("Adapter", (), {"handle": type("Handle", (), {"model": _OOMModel(), "tokenizer": _Tokenizer()})()})()
    report = run(adapter, {"throughput_batch_sizes": [1]})

    assert report["status"] == "no_successful_candidates"
    assert report["candidates"] == [{"batch_size": 1, "status": "oom"}]
    assert empty_cache_calls == [True]
