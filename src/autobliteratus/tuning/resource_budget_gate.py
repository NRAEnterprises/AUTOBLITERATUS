"""Report-only resource budget checks for measured runtime observations."""

from __future__ import annotations

import math
from typing import Any

TOGGLE = {
    "id": "tuning.resource_budget_gate",
    "default": False,
    "desc": "Compare measured memory, throughput, latency, batch size, and sequence length against explicit budgets.",
}

_BUDGET_FIELDS = {
    "max_host_rss_bytes": ("host_rss_bytes", "<=", "bytes"),
    "max_accelerator_allocated_bytes": ("accelerator_allocated_bytes", "<=", "bytes"),
    "max_latency_ms": ("latency_ms", "<=", "milliseconds"),
    "min_tokens_per_second": ("tokens_per_second", ">=", "tokens_per_second"),
    "max_batch_size": ("batch_size", "<=", "items"),
    "max_sequence_length": ("sequence_length", "<=", "tokens"),
}


def _validate_budget(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError("resource_budget must be a mapping of metric budgets.")
    unknown = sorted(set(value) - set(_BUDGET_FIELDS))
    if unknown:
        raise ValueError(f"resource_budget contains unsupported keys: {unknown}")
    budget: dict[str, float] = {}
    for name, limit in value.items():
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            raise TypeError(f"resource_budget.{name} must be a finite number.")
        if not math.isfinite(float(limit)) or float(limit) < 0:
            raise ValueError(f"resource_budget.{name} must be finite and non-negative.")
        budget[name] = float(limit)
    return budget


def _max_accelerator_allocation(memory_report: dict[str, Any] | None) -> int | None:
    if not isinstance(memory_report, dict):
        return None
    values = memory_report.get("accelerator_allocated_bytes")
    if not isinstance(values, dict):
        return None
    numeric = [int(value) for value in values.values() if isinstance(value, int) and not isinstance(value, bool)]
    return max(numeric, default=None)


def _observations(context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    memory_report = context.get("memory_report")
    throughput_report = context.get("throughput_report")
    latency_report = context.get("latency_report")
    host_rss = memory_report.get("host_rss_bytes") if isinstance(memory_report, dict) else None
    accelerator = _max_accelerator_allocation(memory_report)
    latency_ms = latency_report.get("median_latency_ms") if isinstance(latency_report, dict) else None
    rows = throughput_report.get("candidates") if isinstance(throughput_report, dict) else None
    observations: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("status") != "measured":
                continue
            observations.append(
                {
                    "candidate_index": index,
                    "batch_size": row.get("batch_size"),
                    "sequence_length": row.get("sequence_length"),
                    "tokens_per_second": row.get("tokens_per_second"),
                    "latency_ms": latency_ms,
                    "host_rss_bytes": host_rss,
                    "accelerator_allocated_bytes": accelerator,
                }
            )
    if not observations and isinstance(latency_report, dict) and latency_report.get("status") == "measured":
        observations.append(
            {
                "candidate_index": 0,
                "batch_size": 1,
                "sequence_length": latency_report.get("sequence_length"),
                "tokens_per_second": latency_report.get("median_tokens_per_second"),
                "latency_ms": latency_ms,
                "host_rss_bytes": host_rss,
                "accelerator_allocated_bytes": accelerator,
            }
        )

    missing_reports = []
    if not isinstance(memory_report, dict):
        missing_reports.append("memory_report")
    if not isinstance(throughput_report, dict) and not isinstance(latency_report, dict):
        missing_reports.append("throughput_report_or_latency_report")
    return observations, missing_reports


def _candidate_outcomes(context: dict[str, Any]) -> list[dict[str, Any]]:
    throughput_report = context.get("throughput_report")
    rows = throughput_report.get("candidates") if isinstance(throughput_report, dict) else None
    if not isinstance(rows, list):
        return []
    outcomes: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("status") == "measured":
            continue
        status = row.get("status")
        outcome = {
            "candidate_index": index,
            "status": status if isinstance(status, str) and status else "unknown",
            "batch_size": row.get("batch_size"),
            "sequence_length": row.get("sequence_length"),
        }
        failure_metadata = row.get("failure_metadata")
        if isinstance(failure_metadata, dict):
            outcome["failure_metadata"] = dict(failure_metadata)
        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(float(elapsed)):
            outcome["elapsed_seconds"] = elapsed
        outcomes.append(outcome)
    return outcomes


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Compare existing diagnostic measurements with caller-supplied budgets."""
    del adapter
    budget = _validate_budget(context.get("resource_budget", {}))
    observations, missing_reports = _observations(context)
    candidate_outcomes = _candidate_outcomes(context)
    violations: list[dict[str, Any]] = []
    missing_metrics: list[dict[str, Any]] = []

    for index, observation in enumerate(observations):
        for budget_name, limit in budget.items():
            metric_name, operator, unit = _BUDGET_FIELDS[budget_name]
            observed = observation.get(metric_name)
            if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isfinite(float(observed)):
                missing_metrics.append(
                    {
                        "observation_index": index,
                        "metric": metric_name,
                        "budget": budget_name,
                        "reason": "measurement_unavailable",
                    }
                )
                continue
            exceeded = float(observed) > limit if operator == "<=" else float(observed) < limit
            if exceeded:
                violations.append(
                    {
                        "observation_index": index,
                        "candidate_index": observation.get("candidate_index"),
                        "metric": metric_name,
                        "budget": budget_name,
                        "observed": observed,
                        "limit": limit,
                        "operator": operator,
                        "unit": unit,
                        "batch_size": observation.get("batch_size"),
                        "sequence_length": observation.get("sequence_length"),
                    }
                )

    for report_name in missing_reports:
        missing_metrics.append({"report": report_name, "reason": "report_unavailable"})

    if not budget:
        status = "not_configured"
    elif not observations:
        status = "not_evaluated"
    elif violations:
        status = "violations"
    elif missing_metrics or candidate_outcomes:
        status = "incomplete"
    else:
        status = "within_budget"

    report = {
        "stage_id": "tuning.resource_budget_gate",
        "schema_version": 1,
        "status": status,
        "budget": budget,
        "observations": observations,
        "candidate_outcomes": candidate_outcomes,
        "violations": violations,
        "missing_metrics": missing_metrics,
        "optimization_performed": False,
        "model_modified": False,
        "artifacts": [],
        "warnings": [
            "Resource budgets evaluate measured runner-specific reports; they do not optimize or certify deployment behavior."
        ],
    }
    context["resource_budget_report"] = report
    return report
