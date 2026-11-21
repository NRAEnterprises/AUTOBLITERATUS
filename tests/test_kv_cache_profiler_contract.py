from __future__ import annotations

import importlib

import pytest
import torch

from tests.fixtures.kv_cache_fixtures import (
    make_kv_cache_fixture_adapter,
    projected_kv_cache_bytes,
)

_KV_MODULE = importlib.import_module("autobliteratus.tuning._41_kv_cache_auto_tuning")
_KV_RUN_AVAILABLE = callable(getattr(_KV_MODULE, "run", None))


@pytest.mark.parametrize("use_cache", [False, True])
def test_deterministic_fixture_preserves_output_across_cache_modes(use_cache: bool) -> None:
    adapter = make_kv_cache_fixture_adapter()
    encoded = adapter.handle.tokenizer(
        ["one two", "three four five"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    output = adapter.handle.model.generate(
        **encoded,
        max_new_tokens=2,
        do_sample=False,
        use_cache=use_cache,
    )

    assert output.shape == (2, 5)
    assert torch.equal(output[:, -2:], torch.full((2, 2), 7, dtype=torch.long))
    assert adapter.handle.model.generate_calls == [use_cache]


def test_deterministic_fixture_models_cache_memory_growth() -> None:
    short = projected_kv_cache_bytes(batch_size=1, sequence_length=4, use_cache=True)
    long = projected_kv_cache_bytes(batch_size=1, sequence_length=8, use_cache=True)
    wider = projected_kv_cache_bytes(batch_size=2, sequence_length=4, use_cache=True)

    assert short == 512
    assert long == short * 2
    assert wider == short * 2
    assert projected_kv_cache_bytes(batch_size=2, sequence_length=4, use_cache=False) == 0


def test_kv_cache_profiler_is_callable_but_remains_opt_in() -> None:
    from autobliteratus.tuning.toggle_registry import TOGGLES

    assert TOGGLES["tuning.kv_cache_auto"]["available"] is True
    assert TOGGLES["tuning.kv_cache_auto"]["default"] is False
    assert _KV_RUN_AVAILABLE is True


def test_kv_cache_profiler_reports_bounded_matrix_and_quality_gate() -> None:
    adapter = make_kv_cache_fixture_adapter()
    context = {
        "kv_cache_sequence_lengths": [2, 4],
        "kv_cache_modes": [False, True],
        "kv_cache_batch_sizes": [1],
        "kv_cache_max_new_tokens": 2,
        "kv_cache_output_tolerance": 0.0,
    }

    report = _KV_MODULE.run(adapter, context)

    assert report["stage_id"] == "tuning.kv_cache_auto"
    assert report["schema_version"] == 1
    assert report["status"] in {"measured", "violations", "incomplete"}
    assert len(report["candidates"]) == 4
    assert all("sequence_length" in row for row in report["candidates"])
    assert all("use_cache" in row for row in report["candidates"])
    assert report["model_modified"] is False
    assert report["optimization_performed"] is False


def test_kv_cache_profiler_preserves_cancellation_metadata() -> None:
    adapter = make_kv_cache_fixture_adapter()
    report = _KV_MODULE.run(
        adapter,
        {
            "kv_cache_sequence_lengths": [2],
            "kv_cache_modes": [False, True],
            "kv_cache_batch_sizes": [1, 2],
            "kv_cache_cancelled": True,
        },
    )

    assert report["status"] == "cancelled"
    assert len(report["candidates"]) == 4
    assert all(
        row["failure_metadata"]
        == {"kind": "cancelled", "backend": "cpu", "reason": "requested"}
        for row in report["candidates"]
    )


def test_kv_cache_profiler_rejects_unbounded_or_invalid_inputs() -> None:
    adapter = make_kv_cache_fixture_adapter()

    with pytest.raises(ValueError, match="sequence"):
        _KV_MODULE.run(adapter, {"kv_cache_sequence_lengths": list(range(1, 100))})
    with pytest.raises(ValueError, match="tolerance"):
        _KV_MODULE.run(adapter, {"kv_cache_output_tolerance": -1})
