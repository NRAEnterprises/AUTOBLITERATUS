from __future__ import annotations

from autobliteratus.tuning.promotion_evidence_coordinator import (
    REQUIRED_REPORTS,
    coordinate_promotion_evidence,
)


def _reports() -> dict[str, dict[str, str]]:
    return {
        "capability_report": {"status": "measured"},
        "safety_report": {"status": "passed"},
        "latency_report": {"status": "measured"},
        "throughput_report": {"status": "measured"},
        "memory_report": {"status": "measured"},
        "kv_cache_report": {"status": "measured"},
        "regression_report": {"status": "passed"},
        "resource_budget_report": {"status": "within_budget"},
    }


def test_complete_evidence_is_accepted_without_granting_mutation_authority() -> None:
    report = coordinate_promotion_evidence(_reports())

    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["blocking_reports"] == []
    assert report["required_reports"] == list(REQUIRED_REPORTS)
    assert report["mutation_authorized"] is False
    assert report["optimization_performed"] is False
    assert report["evidence_digest"]


def test_missing_or_non_passing_evidence_is_rejected() -> None:
    reports = _reports()
    reports.pop("kv_cache_report")
    reports["resource_budget_report"] = {"status": "violations"}

    report = coordinate_promotion_evidence(reports)

    assert report["status"] == "rejected"
    assert "kv_cache_report" in report["blocking_reports"]
    assert "resource_budget_report" in report["blocking_reports"]
    assert any("kv_cache_report is required" in error for error in report["errors"])
    assert any("resource_budget_report.status" in error for error in report["errors"])
    assert report["mutation_authorized"] is False


def test_non_mapping_input_is_rejected() -> None:
    report = coordinate_promotion_evidence(None)

    assert report["status"] == "rejected"
    assert report["mutation_authorized"] is False
    assert report["blocking_reports"] == list(REQUIRED_REPORTS)
