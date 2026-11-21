"""Coordinate measured quality and resource reports before runtime promotion."""

from __future__ import annotations

import hashlib
import json
from typing import Any

REQUIRED_REPORTS = (
    "capability_report",
    "safety_report",
    "latency_report",
    "throughput_report",
    "memory_report",
    "kv_cache_report",
    "regression_report",
    "resource_budget_report",
)

_ALLOWED_STATUSES = {
    "capability_report": {"measured", "passed"},
    "safety_report": {"measured", "passed"},
    "latency_report": {"measured"},
    "throughput_report": {"measured"},
    "memory_report": {"measured"},
    "kv_cache_report": {"measured", "passed"},
    "regression_report": {"passed"},
    "resource_budget_report": {"within_budget"},
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def coordinate_promotion_evidence(reports: Any) -> dict[str, Any]:
    """Return a deterministic report-only decision for promotion evidence.

    This function never grants mutation authority. It requires every evidence
    class and rejects missing, malformed, incomplete, failed, or budget-violating
    reports so callers cannot mistake partial measurements for qualification.
    """
    errors: list[str] = []
    statuses: dict[str, str | None] = {}
    if not isinstance(reports, dict):
        reports = {}
        errors.append("reports must be a mapping of required evidence reports")

    for name in REQUIRED_REPORTS:
        report = reports.get(name)
        if not isinstance(report, dict):
            errors.append(f"{name} is required and must be an object")
            statuses[name] = None
            continue
        status = report.get("status")
        statuses[name] = status if isinstance(status, str) else None
        if status not in _ALLOWED_STATUSES[name]:
            expected = sorted(_ALLOWED_STATUSES[name])
            errors.append(f"{name}.status must be one of {expected}")

    blocking_reports = [
        name for name in REQUIRED_REPORTS if statuses.get(name) not in _ALLOWED_STATUSES[name]
    ]
    payload = {
        "schema_version": 1,
        "required_reports": list(REQUIRED_REPORTS),
        "report_statuses": statuses,
        "blocking_reports": blocking_reports,
    }
    report = {
        **payload,
        "status": "passed" if not errors else "rejected",
        "errors": errors,
        "mutation_authorized": False,
        "optimization_performed": False,
        "evidence_digest": _digest(payload),
    }
    return report
