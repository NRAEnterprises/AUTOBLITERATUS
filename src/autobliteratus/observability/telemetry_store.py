"""Opt-in local telemetry warehouse for AUTOBLITERATUS test evidence.

The first increment imports the existing aggregate CSV ledger. It does not
capture live test output or enable any runtime pipeline stage; those integrations
must use the versioned schema and redacted event contract documented separately.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema-v1.sql"
_FILE_MODE_LOCK = threading.RLock()
_DATABASE_WRITER_THREAD_LOCK = threading.RLock()
_DATABASE_WRITER_LOCAL = threading.local()
_REQUIRED_COLUMNS = {
    "date",
    "commit",
    "platform",
    "command",
    "passed",
    "failed",
    "skipped",
    "notes",
}
_SECRET_KEY = (
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|token|"
    r"password|passwd|secret|client[_-]?secret|private[_-]?key|"
    r"secret[_-]?access[_-]?key"
)
_SECRET_IDENTIFIER = rf"(?:[A-Za-z][A-Za-z0-9]*[_-])*(?:{_SECRET_KEY})"
_SECRET_VALUE = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}&]+)'''
_AUTHORIZATION_HEADER = re.compile(
    rf'''(?ix)(?P<prefix>(?<![A-Za-z0-9_])["']?(?:[A-Za-z][A-Za-z0-9]*[_-])*authorization["']?\s*[:=]\s*)'''
    rf'''(?P<value>(?:Bearer|Basic)\s+[^\s,;]+|{_SECRET_VALUE})'''
)
_SECRET_ASSIGNMENT = re.compile(
    rf'''(?ix)(?P<prefix>(?<![A-Za-z0-9_])["']?(?:--)?(?:{_SECRET_IDENTIFIER})["']?\s*[:=]\s*)'''
    rf'''(?P<scheme>(?:Bearer|Basic)\s+)?(?P<value>{_SECRET_VALUE})'''
)
_CLI_SECRET_ARGUMENT = re.compile(
    rf'''(?ix)(?P<prefix>(?<![A-Za-z0-9_])--(?:{_SECRET_IDENTIFIER})\s+)'''
    rf'''(?P<scheme>(?:Bearer|Basic)\s+)?(?P<value>{_SECRET_VALUE})'''
)
_BEARER_TOKEN = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_PROVIDER_TOKEN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|"
    r"glpat-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9-]+|ya29\.[A-Za-z0-9_-]+)"
)


@dataclass(frozen=True)
class ImportResult:
    rows_seen: int
    rows_inserted: int
    source_sha256: str


@dataclass(frozen=True)
class ResourceBudgetImportResult:
    report_id: str
    inserted: bool
    observations_seen: int
    findings_seen: int
    source_sha256: str


@dataclass(frozen=True)
class BenchmarkImportResult:
    report_id: str
    report_kind: str
    inserted: bool
    records_seen: int
    rows_inserted: int
    source_sha256: str


_RESULTS_CSV_MAX_BYTES = 64 * 1024 * 1024
_RESOURCE_REPORT_MAX_BYTES = 4 * 1024 * 1024
_BENCHMARK_MAX_BYTES = 16 * 1024 * 1024
_BENCHMARK_MAX_RECORDS = 100_000
_BENCHMARK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_BENCHMARK_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_BENCHMARK_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BENCHMARK_STATISTICS = {"observed", "median", "p95", "first_after_parity"}
_BENCHMARK_CACHE_STATES = {"cold", "warm", "after_parity", "unknown", "not_applicable"}
_BENCHMARK_CSV_REQUIRED = {
    "schema_version", "workload_id", "engine", "query_id", "latency_statistic",
    "correct", "latency_ms", "rows_returned",
}
_BENCHMARK_CSV_OPTIONAL = {
    "engine_version", "run_id", "input_tokens", "output_tokens", "context_bytes",
    "result_sha256", "cache_state", "hardware_fingerprint", "measured_at_utc",
}
_RESOURCE_STATUSES = {
    "not_configured",
    "not_evaluated",
    "violations",
    "incomplete",
    "within_budget",
}
_RESOURCE_CANDIDATE_STATUSES = {
    "oom",
    "timed_out",
    "cancelled",
    "failed",
    "unknown",
}
_RESOURCE_FAILURE_KINDS = {"timeout", "out_of_memory", "cancelled"}
_RESOURCE_BACKEND_RE = re.compile(r"^(?:cpu|mps|xpu|rocm|cuda(?::[0-9]{1,3})?)$", re.IGNORECASE)
_RESOURCE_METRICS = {
    "max_host_rss_bytes": ("host_rss_bytes", "<=", "bytes"),
    "max_accelerator_allocated_bytes": ("accelerator_allocated_bytes", "<=", "bytes"),
    "max_latency_ms": ("latency_ms", "<=", "milliseconds"),
    "min_tokens_per_second": ("tokens_per_second", ">=", "tokens_per_second"),
    "max_batch_size": ("batch_size", "<=", "items"),
    "max_sequence_length": ("sequence_length", "<=", "tokens"),
}
_RESOURCE_MISSING_REPORTS = {
    "memory_report",
    "throughput_report_or_latency_report",
}


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _check_json_nesting(raw: bytes, *, maximum_depth: int = 64) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > maximum_depth:
                raise ValueError("JSON nesting limit exceeded")
        elif byte in (0x7D, 0x5D):
            depth = max(0, depth - 1)


def _resource_integer(
    value: Any,
    field: str,
    *,
    nullable: bool = False,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int | None:
    if value is None and nullable:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"Resource-budget report has invalid {field}")
    return value


def _resource_number(value: Any, field: str, *, nullable: bool = False) -> int | float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Resource-budget report has invalid {field}")
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"Resource-budget report has invalid {field}")
    return value


def _resource_double(value: Any, field: str, *, nullable: bool = False) -> float | None:
    number = _resource_number(value, field, nullable=nullable)
    if number is None:
        return None
    try:
        converted = float(number)
    except OverflowError as exc:
        raise ValueError(f"Resource-budget report has out-of-range {field}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"Resource-budget report has out-of-range {field}")
    return converted


def _safe_resource_failure_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Resource-budget failure_metadata must be an object")
    safe: dict[str, Any] = {}
    kind = value.get("kind")
    if isinstance(kind, str) and kind in _RESOURCE_FAILURE_KINDS:
        safe["kind"] = kind
    backend = value.get("backend")
    if isinstance(backend, str) and _RESOURCE_BACKEND_RE.fullmatch(backend):
        safe["backend"] = backend.lower()
    reason = value.get("reason")
    if reason == "requested":
        safe["reason"] = reason
    for key in ("duration_seconds", "limit_seconds"):
        if key in value:
            safe[key] = _resource_double(value[key], f"failure_metadata.{key}")
    cleanup = value.get("cleanup")
    if isinstance(cleanup, dict):
        safe_cleanup = {
            key: cleanup[key]
            for key in ("attempted", "succeeded")
            if isinstance(cleanup.get(key), bool)
        }
        if cleanup.get("action") == "cuda.empty_cache":
            safe_cleanup["action"] = "cuda.empty_cache"
        if safe_cleanup:
            safe["cleanup"] = safe_cleanup
    return safe or None


def _validate_resource_budget_report(report: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate v1 and return only bounded, allowlisted fields safe to persist."""
    required = {
        "stage_id", "schema_version", "status", "budget", "observations",
        "candidate_outcomes", "violations", "missing_metrics",
        "optimization_performed", "model_modified", "artifacts", "warnings",
    }
    if not isinstance(report, dict) or not required.issubset(report):
        raise ValueError("Resource-budget report is missing required v1 fields")
    if report["stage_id"] != "tuning.resource_budget_gate":
        raise ValueError("Unsupported resource-budget stage")
    if (
        isinstance(report["schema_version"], bool)
        or not isinstance(report["schema_version"], int)
        or report["schema_version"] != 1
    ):
        raise ValueError("Unsupported resource-budget schema version")
    if not isinstance(report["status"], str) or report["status"] not in _RESOURCE_STATUSES:
        raise ValueError("Resource-budget report has an invalid status")
    if report["optimization_performed"] is not False or report["model_modified"] is not False:
        raise ValueError("Resource-budget reports must remain report-only")
    if report["artifacts"] != []:
        raise ValueError("Resource-budget v1 reports cannot publish artifacts")
    if not isinstance(report["warnings"], list) or any(not isinstance(item, str) for item in report["warnings"]):
        raise ValueError("Resource-budget report warnings must be strings")

    raw_budget = report["budget"]
    if not isinstance(raw_budget, dict) or set(raw_budget) - set(_RESOURCE_METRICS):
        raise ValueError("Resource-budget report has unknown budget keys")
    budget = {
        key: _resource_number(value, f"budget.{key}")
        for key, value in raw_budget.items()
    }
    if not all(isinstance(report[key], list) for key in (
        "observations", "candidate_outcomes", "violations", "missing_metrics"
    )):
        raise ValueError("Resource-budget report evidence fields must be arrays")

    observations: list[dict[str, Any]] = []
    observation_fields = (
        "tokens_per_second", "latency_ms", "host_rss_bytes", "accelerator_allocated_bytes"
    )
    for item in report["observations"]:
        if not isinstance(item, dict):
            raise TypeError("Resource-budget report contains an invalid observation")
        observation = {
            "candidate_index": _resource_integer(
                item.get("candidate_index"), "candidate_index", maximum=2**31 - 1
            ),
            "batch_size": _resource_integer(item.get("batch_size"), "batch_size", nullable=True, minimum=1),
            "sequence_length": _resource_integer(item.get("sequence_length"), "sequence_length", nullable=True, minimum=1),
        }
        for field in observation_fields:
            if field in {"host_rss_bytes", "accelerator_allocated_bytes"}:
                observation[field] = _resource_integer(item.get(field), field, nullable=True)
            else:
                observation[field] = _resource_double(item.get(field), field, nullable=True)
        observations.append(observation)

    candidate_outcomes: list[dict[str, Any]] = []
    for item in report["candidate_outcomes"]:
        if not isinstance(item, dict) or not isinstance(item.get("status"), str):
            raise TypeError("Resource-budget report contains an invalid candidate outcome")
        status = item["status"]
        if status not in _RESOURCE_CANDIDATE_STATUSES:
            raise ValueError("Resource-budget candidate status is unsupported")
        outcome = {
            "candidate_index": _resource_integer(
                item.get("candidate_index"), "candidate_index", maximum=2**31 - 1
            ),
            "status": status,
            "batch_size": _resource_integer(item.get("batch_size"), "batch_size", nullable=True, minimum=1),
            "sequence_length": _resource_integer(item.get("sequence_length"), "sequence_length", nullable=True, minimum=1),
        }
        if "elapsed_seconds" in item:
            outcome["elapsed_seconds"] = _resource_number(item["elapsed_seconds"], "elapsed_seconds")
        safe_failure = _safe_resource_failure_metadata(item.get("failure_metadata"))
        if safe_failure is not None:
            outcome["failure_metadata"] = safe_failure
        candidate_outcomes.append(outcome)

    violations: list[dict[str, Any]] = []
    for item in report["violations"]:
        if not isinstance(item, dict):
            raise TypeError("Resource-budget report contains an invalid violation")
        budget_key = item.get("budget")
        if not isinstance(budget_key, str) or budget_key not in _RESOURCE_METRICS or budget_key not in budget:
            raise ValueError("Resource-budget violation references an unknown budget")
        metric, operator, unit = _RESOURCE_METRICS[budget_key]
        if item.get("metric") != metric or item.get("operator") != operator or item.get("unit") != unit:
            raise ValueError("Resource-budget violation metadata does not match its budget")
        index = _resource_integer(
            item.get("observation_index"), "observation_index", maximum=2**31 - 1
        )
        if index >= len(observations):
            raise ValueError("Resource-budget violation references a missing observation")
        observation = observations[index]
        candidate_index = _resource_integer(
            item.get("candidate_index"),
            "candidate_index",
            nullable=True,
            maximum=2**31 - 1,
        )
        observed = _resource_number(item.get("observed"), "violation.observed")
        limit = _resource_number(item.get("limit"), "violation.limit")
        if limit != budget[budget_key] or candidate_index != observation["candidate_index"]:
            raise ValueError("Resource-budget violation does not match its source observation")
        if observed != observation[metric]:
            raise ValueError("Resource-budget violation value differs from its observation")
        if not (observed > limit if operator == "<=" else observed < limit):
            raise ValueError("Resource-budget violation does not cross its configured limit")
        violations.append({
            "kind": "violation", "code": metric, "metric": metric,
            "budget_key": budget_key, "observed": observed, "limit_value": limit,
            "unit": unit,
            "details": {
                "observation_index": index,
                "candidate_index": candidate_index,
                "batch_size": observation["batch_size"],
                "sequence_length": observation["sequence_length"],
            },
        })

    missing: list[dict[str, Any]] = []
    for item in report["missing_metrics"]:
        if not isinstance(item, dict) or not isinstance(item.get("reason"), str):
            raise TypeError("Resource-budget report contains an invalid missing-metric record")
        reason = item["reason"]
        report_name = item.get("report")
        if (
            reason == "report_unavailable"
            and isinstance(report_name, str)
            and report_name in _RESOURCE_MISSING_REPORTS
        ):
            details = {"report": report_name, "reason": reason}
            missing.append({"kind": "missing_metric", "code": reason, "metric": None, "budget_key": None, "observed": None, "limit_value": None, "unit": None, "details": details})
            continue
        budget_key = item.get("budget")
        metric = item.get("metric")
        if (
            reason != "measurement_unavailable"
            or not isinstance(budget_key, str)
            or budget_key not in _RESOURCE_METRICS
            or metric != _RESOURCE_METRICS[budget_key][0]
            or budget_key not in budget
        ):
            raise ValueError("Resource-budget report contains an unsupported missing-metric code")
        index = _resource_integer(
            item.get("observation_index"), "observation_index", maximum=2**31 - 1
        )
        if index >= len(observations) or observations[index][metric] is not None:
            raise ValueError("Resource-budget missing metric does not match its observation")
        missing.append({"kind": "missing_metric", "code": reason, "metric": metric, "budget_key": budget_key, "observed": None, "limit_value": None, "unit": _RESOURCE_METRICS[budget_key][2], "details": {"observation_index": index}})

    expected_violations: set[tuple[int, str]] = set()
    expected_missing: set[tuple[int, str]] = set()
    for observation_index, observation in enumerate(observations):
        for budget_key, limit in budget.items():
            metric, operator, _unit = _RESOURCE_METRICS[budget_key]
            observed = observation[metric]
            if observed is None:
                expected_missing.add((observation_index, budget_key))
            elif observed > limit if operator == "<=" else observed < limit:
                expected_violations.add((observation_index, budget_key))

    actual_violations = {
        (item["details"]["observation_index"], item["budget_key"])
        for item in violations
    }
    actual_missing = [
        (item["details"]["observation_index"], item["budget_key"])
        for item in missing
        if item["kind"] == "missing_metric" and "observation_index" in item["details"]
    ]
    if actual_violations != expected_violations or len(violations) != len(expected_violations):
        raise ValueError("Resource-budget violations are incomplete or inconsistent with observations")
    if set(actual_missing) != expected_missing or len(actual_missing) != len(expected_missing):
        raise ValueError("Resource-budget missing metrics are incomplete or inconsistent with observations")

    for outcome in candidate_outcomes:
        details = {key: value for key, value in outcome.items() if key != "status"}
        missing.append({"kind": "candidate_outcome", "code": outcome["status"], "metric": None, "budget_key": None, "observed": None, "limit_value": None, "unit": None, "details": details})

    expected_status = (
        "not_configured" if not budget
        else "not_evaluated" if not observations
        else "violations" if violations
        else "incomplete" if missing
        else "within_budget"
    )
    if report["status"] != expected_status:
        raise ValueError("Resource-budget report status conflicts with its evidence")
    return {"status": expected_status, "budget": budget}, observations, missing + violations


def _benchmark_text(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Benchmark {field} must be a non-empty string")
    if not _BENCHMARK_ID_RE.fullmatch(value):
        raise ValueError(f"Benchmark {field} contains unsupported characters")
    return value


def _benchmark_integer(value: Any, field: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"Benchmark {field} must be a non-negative integer")
    return value


def _benchmark_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Benchmark {field} must be a finite non-negative number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"Benchmark {field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Benchmark {field} must be a finite non-negative number")
    return number


def _benchmark_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Benchmark measured_at_utc must include a timezone")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        raise ValueError("Benchmark measured_at_utc must be an ISO-8601 string")  # noqa: TRY004
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Benchmark measured_at_utc must be an ISO-8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Benchmark measured_at_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _benchmark_record(
    *, workload_id: Any, engine: Any, engine_version: Any, query_id: Any,
    statistic: Any, correct: Any, latency_ms: Any, rows_returned: Any,
    result_sha256: Any = None, cache_state: Any = None, run_id: Any = None,
    input_tokens: Any = None, output_tokens: Any = None, context_bytes: Any = None,
    hardware_fingerprint: Any = None, measured_at: Any = None,
) -> dict[str, Any]:
    workload = _benchmark_text(workload_id, "workload_id")
    engine_name = _benchmark_text(engine, "engine")
    query = _benchmark_text(query_id, "query_id")
    if not isinstance(statistic, str) or statistic not in _BENCHMARK_STATISTICS:
        raise ValueError("Benchmark latency_statistic is unsupported")
    if not isinstance(correct, bool):
        raise ValueError("Benchmark correct must be a JSON boolean or CSV true/false")  # noqa: TRY004
    if result_sha256 not in (None, "") and (
        not isinstance(result_sha256, str) or not _BENCHMARK_SHA256_RE.fullmatch(result_sha256)
    ):
        raise ValueError("Benchmark result_sha256 must be a lowercase SHA-256 digest")
    if cache_state in (None, ""):
        cache = "after_parity" if statistic == "first_after_parity" else None
    elif isinstance(cache_state, str) and cache_state in _BENCHMARK_CACHE_STATES:
        cache = cache_state
    else:
        raise ValueError("Benchmark cache_state is unsupported")
    version = None if engine_version in (None, "") else _benchmark_text(engine_version, "engine_version")
    hardware = None if hardware_fingerprint in (None, "") else _benchmark_text(hardware_fingerprint, "hardware_fingerprint")
    run = None if run_id in (None, "") else _benchmark_text(run_id, "run_id")
    return {
        "workload_id": workload, "engine": engine_name, "engine_version": version,
        "query_id": query, "latency_statistic": statistic, "correct": correct,
        "latency_ms": _benchmark_number(latency_ms, "latency_ms"),
        "rows_returned": _benchmark_integer(rows_returned, "rows_returned"),
        "input_tokens": _benchmark_integer(input_tokens, "input_tokens", optional=True),
        "output_tokens": _benchmark_integer(output_tokens, "output_tokens", optional=True),
        "context_bytes": _benchmark_integer(context_bytes, "context_bytes", optional=True),
        "result_sha256": result_sha256 or None, "cache_state": cache,
        "hardware_fingerprint": hardware, "run_id": run,
        "measured_at": _benchmark_timestamp(measured_at),
    }


def _normalize_telemetry_benchmark_json(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("benchmark") != "autobliteratus_real_telemetry_ledger_parity_comparison":
        raise ValueError("Unsupported benchmark JSON report format")
    if payload.get("status") != "completed_parity_checked_smoke_scale":
        raise ValueError("Telemetry benchmark report is not a completed parity-checked report")
    parity = payload.get("parity")
    engines = payload.get("engines")
    if not isinstance(parity, dict) or parity.get("all_engines_match") is not True or parity.get("all_timed_executions_checked") is not True:
        raise ValueError("Telemetry benchmark parity validation failed")
    hashes = parity.get("result_hashes")
    if not isinstance(hashes, dict) or not hashes or not isinstance(engines, dict) or not engines:
        raise ValueError("Telemetry benchmark report is missing engines or parity hashes")
    measured_at = _benchmark_timestamp(payload.get("measured_at_utc"))
    workload_id = payload["benchmark"]
    records: list[dict[str, Any]] = []
    for engine_name, engine_data in engines.items():
        _benchmark_text(engine_name, "engine")
        if not isinstance(engine_data, dict) or not isinstance(engine_data.get("queries"), dict):
            raise ValueError("Telemetry benchmark engine has invalid query data")  # noqa: TRY004
        if set(engine_data["queries"]) != set(hashes):
            raise ValueError("Telemetry benchmark engine query set conflicts with parity evidence")
        for query_id, query_data in engine_data["queries"].items():
            _benchmark_text(query_id, "query_id")
            if not isinstance(query_data, dict) or query_data.get("correct") is not True:
                raise ValueError("Telemetry benchmark query failed parity")
            result_hash = query_data.get("result_sha256")
            if result_hash != hashes.get(query_id):
                raise ValueError("Telemetry benchmark query hash conflicts with parity evidence")
            for source_key, statistic in (
                ("median_ms", "median"), ("p95_ms", "p95"),
                ("first_measured_after_parity_ms", "first_after_parity"),
            ):
                if source_key not in query_data:
                    raise ValueError(f"Telemetry benchmark query is missing {source_key}")
                records.append(_benchmark_record(
                    workload_id=workload_id, engine=engine_name,
                    engine_version=engine_data.get("version"), query_id=query_id,
                    statistic=statistic, correct=True, latency_ms=query_data[source_key],
                    rows_returned=query_data.get("rows_returned"), result_sha256=result_hash,
                    cache_state="after_parity", measured_at=measured_at,
                ))
                if len(records) > _BENCHMARK_MAX_RECORDS:
                    raise ValueError("Benchmark report exceeds the 100000 measurement limit")
    if not records:
        raise ValueError("Telemetry benchmark report contains no measurements")
    return records


def _normalize_benchmark_csv(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Benchmark CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError("Benchmark CSV contains duplicate columns")
    missing = _BENCHMARK_CSV_REQUIRED - set(fields)
    unknown = set(fields) - _BENCHMARK_CSV_REQUIRED - _BENCHMARK_CSV_OPTIONAL
    if missing or unknown:
        raise ValueError(
            "Benchmark CSV columns invalid; missing=" + ",".join(sorted(missing))
            + "; unsupported=" + ",".join(sorted(unknown))
        )
    records: list[dict[str, Any]] = []
    try:
        for index, source_row in enumerate(reader):
            if index >= _BENCHMARK_MAX_RECORDS:
                raise ValueError("Benchmark CSV exceeds the 100000 measurement limit")
            if None in source_row or any(value is None for value in source_row.values()):
                raise ValueError(f"Benchmark CSV row {index + 2} has inconsistent columns")
            row = {key: value.strip() for key, value in source_row.items()}
            if row["schema_version"] != "1":
                raise ValueError("Benchmark CSV schema_version must be 1")
            correct = row["correct"].lower()
            if correct not in {"true", "false"}:
                raise ValueError("Benchmark CSV correct must be true or false")

            def integer_cell(name: str, source_row: dict[str, str] = row) -> int | None:
                value = source_row.get(name, "")
                if not value:
                    return None
                if not re.fullmatch(r"[0-9]+", value):
                    raise ValueError(f"Benchmark CSV {name} must be a non-negative integer")
                return int(value)

            try:
                latency = float(row["latency_ms"])
            except ValueError as exc:
                raise ValueError(f"Benchmark CSV row {index + 2} has invalid latency_ms") from exc
            records.append(_benchmark_record(
                workload_id=row["workload_id"], engine=row["engine"],
                engine_version=row.get("engine_version"), query_id=row["query_id"],
                statistic=row["latency_statistic"], correct=correct == "true",
                latency_ms=latency, rows_returned=integer_cell("rows_returned"),
                result_sha256=row.get("result_sha256"), cache_state=row.get("cache_state"),
                run_id=row.get("run_id"), input_tokens=integer_cell("input_tokens"),
                output_tokens=integer_cell("output_tokens"), context_bytes=integer_cell("context_bytes"),
                hardware_fingerprint=row.get("hardware_fingerprint"),
                measured_at=row.get("measured_at_utc"),
            ))
    except csv.Error as exc:
        raise ValueError("Benchmark CSV is malformed") from exc
    if not records:
        raise ValueError("Benchmark CSV contains no measurements")
    return records


def redact_text(value: str) -> str:
    """Redact common credential forms before storing free-text telemetry."""

    def replace_assignment(match: re.Match[str]) -> str:
        secret = match.group("value")
        if secret[:1] in ("'", '"') and secret[-1:] == secret[:1]:
            quote = secret[0]
            replacement = f"{quote}[REDACTED]{quote}"
        else:
            replacement = "[REDACTED]"
        return match.group("prefix") + (match.groupdict().get("scheme") or "") + replacement

    redacted = _AUTHORIZATION_HEADER.sub(
        lambda match: match.group("prefix") + "[REDACTED]", value
    )
    redacted = _SECRET_ASSIGNMENT.sub(replace_assignment, redacted)
    redacted = _CLI_SECRET_ARGUMENT.sub(replace_assignment, redacted)
    redacted = _BEARER_TOKEN.sub(r"\1[REDACTED]", redacted)
    return _PROVIDER_TOKEN.sub("[REDACTED]", redacted)


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise RuntimeError(
            "DuckDB telemetry support is optional. Install DuckDB with "
            "`pip install 'duckdb>=2.0,<3.0'`; in a fully provisioned project "
            "environment, the `observability` extra may be installed instead."
        ) from exc
    return duckdb


@contextmanager
def database_writer_lock(database_path: str | Path, timeout_seconds: float = 60.0):
    """Serialize every process writing one telemetry database, reentrantly per thread."""
    database = Path(database_path).expanduser().resolve()
    key = str(database)
    with _DATABASE_WRITER_THREAD_LOCK:
        held = getattr(_DATABASE_WRITER_LOCAL, "held", {})
        if held.get(key, 0):
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return

        database.parent.mkdir(parents=True, exist_ok=True)
        lock_path = database.with_name(database.name + ".collector.lock")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        if os.name == "posix":
            lock_path.chmod(0o600)
        locked = False
        deadline = time.monotonic() + timeout_seconds
        try:
            while not locked:
                try:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the telemetry database writer lock") from exc
                    time.sleep(0.05)
            held[key] = 1
            _DATABASE_WRITER_LOCAL.held = held
            try:
                yield
            finally:
                held.pop(key, None)
        finally:
            if locked:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


@contextmanager
def _private_file_creation():
    """Use a restrictive umask while DuckDB creates the database or sidecars."""
    if os.name != "posix":
        yield
        return
    # umask is process-global; serialize our own updates and only make files more
    # restrictive for the short duration of a database operation.
    with _FILE_MODE_LOCK:
        previous_umask = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous_umask)


def private_database_operation(database_argument_index: int):
    """Keep database and WAL files owner-only throughout a write operation."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            database_path = kwargs.get("database_path")
            if database_path is None:
                database_path = args[database_argument_index]
            path = Path(database_path).expanduser()
            with database_writer_lock(path), _private_file_creation():
                try:
                    return function(*args, **kwargs)
                finally:
                    if os.name == "posix":
                        for candidate in (path, Path(f"{path}.wal")):
                            if candidate.is_file() and not candidate.is_symlink():
                                candidate.chmod(0o600)
        return wrapped
    return decorate


@private_database_operation(0)
def initialize_database(database_path: str | Path) -> Path:
    """Create the local database and apply the checked-in v1 schema."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing to initialize telemetry through a symlink")
    if path.exists():
        if not path.is_file():
            raise IsADirectoryError(path)
        path.chmod(0o600)
    if not _SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Telemetry schema not found: {_SCHEMA_PATH}")
    connection = _duckdb().connect(str(path))
    try:
        connection.execute(_SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        connection.close()
    if os.name == "posix":
        path.chmod(0o600)
    return path


def _row_run_id(row: dict[str, str]) -> str:
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(("autobliteratus-csv-summary-v1\0" + canonical).encode("utf-8")).hexdigest()


@private_database_operation(1)
def import_results_csv(csv_path: str | Path, database_path: str | Path) -> ImportResult:
    """Import legacy aggregate test rows, deduplicating by canonical row hash."""
    source_path = Path(csv_path)
    with source_path.open("rb") as stream:
        raw = stream.read(_RESULTS_CSV_MAX_BYTES + 1)
    if len(raw) > _RESULTS_CSV_MAX_BYTES:
        raise ValueError("Test ledger CSV exceeds the 64 MiB import limit")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = set(reader.fieldnames or [])
    missing = _REQUIRED_COLUMNS - fields
    if missing:
        raise ValueError(f"Test ledger is missing required columns: {', '.join(sorted(missing))}")

    path = initialize_database(database_path)
    connection = _duckdb().connect(str(path))
    rows_seen = 0
    rows_inserted = 0
    connection.execute("BEGIN TRANSACTION")
    try:
        for source_row in reader:
            rows_seen += 1
            row = {key: (value or "") for key, value in source_row.items()}
            try:
                run_date = date.fromisoformat(row["date"])
                passed, failed, skipped = (int(row[key]) for key in ("passed", "failed", "skipped"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid date or test count in CSV row {rows_seen + 1}") from exc
            if min(passed, failed, skipped) < 0:
                raise ValueError(f"Negative test count in CSV row {rows_seen + 1}")

            status = (
                "failed"
                if failed
                else "passed"
                if passed and not skipped
                else "partial"
                if passed or skipped
                else "no_results"
            )
            run_id = _row_run_id(row)
            result = connection.execute(
                """INSERT INTO pipeline_runs
                   (run_id, run_date, commit_sha, platform, command, status,
                    source_name, source_sha256, notes_redacted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (run_id) DO NOTHING
                   RETURNING run_id""",
                [
                    run_id,
                    run_date,
                    row["commit"],
                    redact_text(row["platform"]),
                    redact_text(row["command"]),
                    status,
                    source_path.name,
                    source_sha256,
                    redact_text(row["notes"]),
                ],
            ).fetchone()
            if result is None:
                continue
            rows_inserted += 1
            connection.execute(
                """INSERT INTO test_summaries (run_id, passed, failed, skipped, total)
                   VALUES (?, ?, ?, ?, ?)""",
                [run_id, passed, failed, skipped, passed + failed + skipped],
            )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return ImportResult(rows_seen, rows_inserted, source_sha256)


def summarize_database(database_path: str | Path) -> dict[str, int]:
    """Return aggregate counts from the local telemetry database."""
    connection = _duckdb().connect(str(database_path), read_only=True)
    try:
        runs = connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        tests = connection.execute(
            "SELECT COALESCE(SUM(total), 0), COALESCE(SUM(failed), 0), "
            "COALESCE(SUM(skipped), 0) FROM test_summaries"
        ).fetchone()
        return {
            "runs": int(runs),
            "tests": int(tests[0]),
            "failed": int(tests[1]),
            "skipped": int(tests[2]),
        }
    finally:
        connection.close()


@private_database_operation(1)
def import_resource_budget_report(
    report_path: str | Path, database_path: str | Path
) -> ResourceBudgetImportResult:
    """Import a report-only resource-budget v1 JSON document idempotently.

    The v1 report contract has no source run ID or measurement timestamp, so this
    adapter stores it in dedicated report tables and records only the import
    time. Unknown fields, warning text, and raw JSON are not persisted.
    """
    source_path = Path(report_path).expanduser()
    with source_path.open("rb") as stream:
        raw = stream.read(_RESOURCE_REPORT_MAX_BYTES + 1)
    if len(raw) > _RESOURCE_REPORT_MAX_BYTES:
        raise ValueError("Resource-budget report exceeds the 4 MiB import limit")
    try:
        _check_json_nesting(raw)
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("Resource-budget report is not valid strict UTF-8 JSON") from exc

    report, observations, findings = _validate_resource_budget_report(payload)
    source_sha256 = hashlib.sha256(raw).hexdigest()
    report_id = hashlib.sha256(
        ("autobliteratus-resource-budget-report-v1\0" + source_sha256).encode("ascii")
    ).hexdigest()
    path = initialize_database(database_path)
    connection = _duckdb().connect(str(path))
    inserted = False
    connection.execute("BEGIN TRANSACTION")
    try:
        result = connection.execute(
            """INSERT INTO resource_budget_reports
               (report_id, stage_id, schema_version, status, source_sha256,
                budget, observation_count, finding_count)
               VALUES (?, ?, 1, ?, ?, ?::JSON, ?, ?)
               ON CONFLICT (report_id) DO NOTHING
               RETURNING report_id""",
            [
                report_id,
                "tuning.resource_budget_gate",
                report["status"],
                source_sha256,
                json.dumps(report["budget"], sort_keys=True, separators=(",", ":")),
                len(observations),
                len(findings),
            ],
        ).fetchone()
        inserted = result is not None
        if inserted:
            for index, item in enumerate(observations):
                connection.execute(
                    """INSERT INTO resource_budget_observations
                       (report_id, observation_index, candidate_index, batch_size,
                        sequence_length, tokens_per_second, latency_ms,
                        host_rss_bytes, accelerator_allocated_bytes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        report_id, index, item["candidate_index"], item["batch_size"],
                        item["sequence_length"], item["tokens_per_second"],
                        item["latency_ms"], item["host_rss_bytes"],
                        item["accelerator_allocated_bytes"],
                    ],
                )
            finding_indexes: dict[str, int] = {}
            for item in findings:
                kind = item["kind"]
                index = finding_indexes.get(kind, 0)
                finding_indexes[kind] = index + 1
                connection.execute(
                    """INSERT INTO resource_budget_findings
                       (report_id, finding_kind, finding_index, code, metric,
                        budget_key, observed, limit_value, unit, details)
                       VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?::JSON, ?, ?::JSON)""",
                    [
                        report_id, kind, index, item["code"], item["metric"],
                        item["budget_key"],
                        json.dumps(item["observed"], allow_nan=False) if item["observed"] is not None else None,
                        json.dumps(item["limit_value"], allow_nan=False) if item["limit_value"] is not None else None,
                        item["unit"], json.dumps(item["details"], sort_keys=True, separators=(",", ":")),
                    ],
                )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return ResourceBudgetImportResult(
        report_id=report_id,
        inserted=inserted,
        observations_seen=len(observations),
        findings_seen=len(findings),
        source_sha256=source_sha256,
    )


@private_database_operation(1)
def import_benchmark_report(
    report_path: str | Path, database_path: str | Path
) -> BenchmarkImportResult:
    """Import supported benchmark JSON or normalized CSV with source provenance.

    Report identity is content-addressed. Only source-provided run IDs that
    already exist in pipeline_runs are linked; absent or unknown IDs and absent
    timestamps remain NULL rather than being inferred from filenames/import time.
    """
    source_path = Path(report_path).expanduser()
    with source_path.open("rb") as stream:
        raw = stream.read(_BENCHMARK_MAX_BYTES + 1)
    if len(raw) > _BENCHMARK_MAX_BYTES:
        raise ValueError("Benchmark report exceeds the 16 MiB import limit")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        try:
            _check_json_nesting(raw)
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_strict_json_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ValueError("Benchmark report is not valid strict UTF-8 JSON") from exc
        records = _normalize_telemetry_benchmark_json(payload)
        report_kind = "telemetry_engine_json_v1"
    elif suffix == ".csv":
        records = _normalize_benchmark_csv(raw)
        report_kind = "query_benchmark_csv_v1"
    else:
        raise ValueError("Benchmark report must have a .json or .csv extension")
    report_id = hashlib.sha256(
        (f"autobliteratus-benchmark-{report_kind}\0" + source_sha256).encode("ascii")
    ).hexdigest()

    path = initialize_database(database_path)
    connection = _duckdb().connect(str(path))
    inserted = False
    rows_inserted = 0
    connection.execute("BEGIN TRANSACTION")
    try:
        result = connection.execute(
            """INSERT INTO benchmark_imports
               (report_id, report_kind, source_sha256, record_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (report_id) DO NOTHING
               RETURNING report_id""",
            [report_id, report_kind, source_sha256, len(records)],
        ).fetchone()
        inserted = result is not None
        if inserted:
            source_run_ids = sorted({record["run_id"] for record in records if record["run_id"]})
            existing_run_ids: set[str] = set()
            if source_run_ids:
                for (known_run_id,) in connection.execute(
                    "SELECT run_id FROM pipeline_runs WHERE run_id IN (SELECT UNNEST(?))",
                    [source_run_ids],
                ).fetchall():
                    existing_run_ids.add(known_run_id)
            for index, record in enumerate(records):
                run_id = record["run_id"] if record["run_id"] in existing_run_ids else None
                measurement_id = hashlib.sha256(
                    f"{report_id}:{index}".encode("ascii")
                ).hexdigest()
                connection.execute(
                    """INSERT INTO benchmark_measurements
                       (measurement_id, report_id, source_record_index, run_id,
                        workload_id, engine, engine_version, query_id,
                        latency_statistic, correct, latency_ms, rows_returned,
                        input_tokens, output_tokens, context_bytes, result_sha256,
                        cache_state, hardware_fingerprint, measured_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        measurement_id, report_id, index, run_id,
                        record["workload_id"], record["engine"], record["engine_version"],
                        record["query_id"], record["latency_statistic"], record["correct"],
                        record["latency_ms"], record["rows_returned"], record["input_tokens"],
                        record["output_tokens"], record["context_bytes"], record["result_sha256"],
                        record["cache_state"], record["hardware_fingerprint"], record["measured_at"],
                    ],
                )
                rows_inserted += 1
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return BenchmarkImportResult(
        report_id=report_id, report_kind=report_kind, inserted=inserted,
        records_seen=len(records), rows_inserted=rows_inserted,
        source_sha256=source_sha256,
    )
