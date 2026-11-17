from __future__ import annotations

import pytest

from autobliteratus.tuning.resource_budget_gate import run
from autobliteratus.tuning.toggle_registry import TOGGLES


def _context() -> dict:
    return {
        "memory_report": {
            "host_rss_bytes": 1000,
            "accelerator_allocated_bytes": {"cuda:0": 2000},
        },
        "latency_report": {
            "status": "measured",
            "median_latency_ms": 12.5,
            "median_tokens_per_second": 80.0,
            "sequence_length": 6,
        },
        "throughput_report": {
            "status": "measured",
            "candidates": [
                {
                    "candidate_index": 0,
                    "status": "measured",
                    "batch_size": 1,
                    "sequence_length": 6,
                    "tokens_per_second": 80.0,
                },
                {
                    "candidate_index": 1,
                    "status": "measured",
                    "batch_size": 4,
                    "sequence_length": 6,
                    "tokens_per_second": 120.0,
                },
            ],
        },
    }


def test_resource_budget_gate_is_callable_and_opt_in() -> None:
    assert TOGGLES["tuning.resource_budget_gate"]["available"] is True
    assert TOGGLES["tuning.resource_budget_gate"]["default"] is False


def test_resource_budget_gate_reports_within_budget_and_correlates_dimensions() -> None:
    context = _context()
    context["resource_budget"] = {
        "max_host_rss_bytes": 1000,
        "max_accelerator_allocated_bytes": 2000,
        "max_latency_ms": 13,
        "min_tokens_per_second": 80,
        "max_batch_size": 4,
        "max_sequence_length": 6,
    }

    report = run(None, context)

    assert report["status"] == "within_budget"
    assert len(report["observations"]) == 2
    assert report["observations"][1]["batch_size"] == 4
    assert report["observations"][1]["sequence_length"] == 6
    assert report["violations"] == []
    assert report["model_modified"] is False
    assert context["resource_budget_report"] is report


def test_resource_budget_gate_emits_machine_readable_violations() -> None:
    context = _context()
    context["resource_budget"] = {
        "max_host_rss_bytes": 900,
        "max_latency_ms": 10,
        "min_tokens_per_second": 130,
        "max_batch_size": 2,
        "max_sequence_length": 5,
    }

    report = run(None, context)

    assert report["status"] == "violations"
    assert {item["budget"] for item in report["violations"]} == {
        "max_host_rss_bytes",
        "max_latency_ms",
        "min_tokens_per_second",
        "max_batch_size",
        "max_sequence_length",
    }
    violation = report["violations"][0]
    assert {"metric", "observed", "limit", "operator", "unit", "batch_size", "sequence_length"} <= violation.keys()


def test_resource_budget_gate_reports_missing_metrics_without_claiming_pass() -> None:
    context = {"resource_budget": {"max_host_rss_bytes": 10, "max_sequence_length": 10}}
    report = run(None, context)

    assert report["status"] == "not_evaluated"
    assert report["violations"] == []
    assert {item["report"] for item in report["missing_metrics"]} == {
        "memory_report",
        "throughput_report_or_latency_report",
    }


def test_resource_budget_gate_reports_incomplete_metric_rows() -> None:
    context = _context()
    context["resource_budget"] = {"max_accelerator_allocated_bytes": 3000}
    context["memory_report"]["accelerator_allocated_bytes"] = {"cuda:0": None}

    report = run(None, context)

    assert report["status"] == "incomplete"
    assert report["missing_metrics"][0]["metric"] == "accelerator_allocated_bytes"


def test_resource_budget_gate_validates_budget_schema() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        run(None, {"resource_budget": {"max_power_watts": 20}})
    with pytest.raises(ValueError, match="finite and non-negative"):
        run(None, {"resource_budget": {"max_latency_ms": float("inf")}})


def test_resource_budget_gate_without_budget_is_not_configured() -> None:
    report = run(None, {})

    assert report["status"] == "not_configured"
    assert report["budget"] == {}
    assert report["optimization_performed"] is False
