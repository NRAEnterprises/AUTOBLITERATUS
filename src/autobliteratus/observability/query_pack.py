"""Parameterized, read-only reports over the versioned observability schema."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .telemetry_store import _duckdb

MAX_REPORT_ROWS = 1_000
_DEFAULT_LIMIT = 200
_DIMENSION_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class QueryReport:
    """Rows plus coverage and provenance for one bounded report query."""

    name: str
    rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL characters")
    return value


def _date_filter(value: Any, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a date or None")
    return value


def _date_range(start_date: Any, end_date: Any) -> tuple[date | None, date | None]:
    start = _date_filter(start_date, "start_date")
    end = _date_filter(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")
    return start, end


def _limit(value: Any) -> tuple[int, int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be a positive integer")
    return value, min(value, MAX_REPORT_ROWS)


def _where(filters: list[tuple[str, Any]]) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    return " WHERE " + " AND ".join(expression for expression, _ in filters), [
        value for _, value in filters
    ]


def _run_report(
    *,
    database_path: str | Path,
    name: str,
    source_tables: tuple[str, ...],
    sql: str,
    count_sql: str,
    provenance_sql: str,
    params: list[Any],
    count_params: list[Any] | None = None,
    provenance_params: list[Any] | None = None,
    filters: dict[str, Any],
    requested_limit: int,
    applied_limit: int,
) -> QueryReport:
    """Execute fixed report SQL with read-only access and bound values only."""
    connection = _duckdb().connect(str(Path(database_path).expanduser()), read_only=True)
    try:
        cursor = connection.execute(sql, [*params, applied_limit])
        columns = [item[0] for item in cursor.description]
        rows = tuple(dict(zip(columns, row, strict=True)) for row in cursor.fetchall())
        group_count = int(connection.execute(count_sql, params if count_params is None else count_params).fetchone()[0])
        source_count, minimum_date, maximum_date, commits = connection.execute(
            provenance_sql, params if provenance_params is None else provenance_params
        ).fetchone()
    finally:
        connection.close()

    if hasattr(commits, "to_list"):
        commits = commits.to_list()
    commit_list = sorted({item for item in (commits or []) if item is not None})
    returned_count = len(rows)
    metadata = {
        "source_tables": list(source_tables),
        "filters": filters,
        "coverage": {
            "source_rows": int(source_count or 0),
            "report_rows": group_count,
            "returned_rows": returned_count,
            "truncated": group_count > returned_count,
            "requested_limit": requested_limit,
            "applied_limit": applied_limit,
        },
        "provenance": {
            "date_start": minimum_date,
            "date_end": maximum_date,
            "commits": commit_list,
        },
    }
    return QueryReport(name=name, rows=rows, metadata=metadata)


def _provenance_query(source: str, where: str, date_expression: str, commit_expression: str) -> str:
    return (
        f"SELECT COUNT(*), MIN(CAST({date_expression} AS DATE)), "
        f"MAX(CAST({date_expression} AS DATE)), "
        f"LIST(DISTINCT {commit_expression}) FROM {source}{where}"
    )


def run_outcomes(
    database_path: str | Path,
    *,
    commit_sha: str | None = None,
    platform: str | None = None,
    status: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> QueryReport:
    """Group run outcomes by commit, platform, and status."""
    commit_sha = _optional_text(commit_sha, "commit_sha")
    platform = _optional_text(platform, "platform")
    status = _optional_text(status, "status")
    start_date, end_date = _date_range(start_date, end_date)
    requested, capped = _limit(limit)
    conditions: list[tuple[str, Any]] = []
    for field, value in (("r.commit_sha", commit_sha), ("r.platform", platform), ("r.status", status)):
        if value is not None:
            conditions.append((f"{field} = ?", value))
    if start_date is not None:
        conditions.append(("r.run_date >= ?", start_date))
    if end_date is not None:
        conditions.append(("r.run_date <= ?", end_date))
    where, params = _where(conditions)
    source = "pipeline_runs AS r LEFT JOIN test_summaries AS s USING (run_id)"
    grouped = (
        "SELECT r.commit_sha, r.platform, r.status, COUNT(*) AS run_count, "
        "SUM(s.passed) AS passed, SUM(s.failed) AS failed, SUM(s.skipped) AS skipped, "
        "SUM(s.total) AS total, MIN(r.run_date) AS date_start, MAX(r.run_date) AS date_end "
        f"FROM {source}{where} GROUP BY r.commit_sha, r.platform, r.status"
    )
    return _run_report(
        database_path=database_path,
        name="run_outcomes_by_commit_platform_status",
        source_tables=("pipeline_runs", "test_summaries"),
        sql=f"SELECT * FROM ({grouped}) AS report ORDER BY commit_sha, platform, status LIMIT ?",
        count_sql=f"SELECT COUNT(*) FROM ({grouped}) AS report",
        provenance_sql=_provenance_query(source, where, "r.run_date", "r.commit_sha"),
        params=params,
        filters={"commit_sha": commit_sha, "platform": platform, "status": status,
                 "start_date": start_date, "end_date": end_date},
        requested_limit=requested,
        applied_limit=capped,
    )


def repeated_test_failures(
    database_path: str | Path,
    *,
    test_id: str | None = None,
    failure_fingerprint: str | None = None,
    commit_sha: str | None = None,
    platform: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_runs: int = 2,
    limit: int = _DEFAULT_LIMIT,
) -> QueryReport:
    """Find fingerprints present in at least ``min_runs`` distinct runs."""
    test_id = _optional_text(test_id, "test_id")
    failure_fingerprint = _optional_text(failure_fingerprint, "failure_fingerprint")
    commit_sha = _optional_text(commit_sha, "commit_sha")
    platform = _optional_text(platform, "platform")
    start_date, end_date = _date_range(start_date, end_date)
    if isinstance(min_runs, bool) or not isinstance(min_runs, int) or min_runs < 2:
        raise ValueError("min_runs must be an integer of at least 2")
    requested, capped = _limit(limit)
    conditions: list[tuple[str, Any]] = [("t.failure_fingerprint IS NOT NULL", None)]
    if test_id is not None:
        conditions.append(("t.test_id = ?", test_id))
    if failure_fingerprint is not None:
        conditions.append(("t.failure_fingerprint = ?", failure_fingerprint))
    if commit_sha is not None:
        conditions.append(("r.commit_sha = ?", commit_sha))
    if platform is not None:
        conditions.append(("r.platform = ?", platform))
    if start_date is not None:
        conditions.append(("r.run_date >= ?", start_date))
    if end_date is not None:
        conditions.append(("r.run_date <= ?", end_date))
    # The fingerprint IS NOT NULL condition has no bind value.
    condition_values = [(clause, value) for clause, value in conditions if value is not None]
    where, params = _where(condition_values)
    if not where:
        where = " WHERE t.failure_fingerprint IS NOT NULL"
    else:
        where += " AND t.failure_fingerprint IS NOT NULL"
    source = "test_results AS t JOIN pipeline_runs AS r USING (run_id)"
    grouped = (
        "SELECT t.failure_fingerprint, COUNT(*) AS failure_occurrences, "
        "COUNT(DISTINCT t.run_id) AS run_count, LIST(DISTINCT t.test_id) AS test_ids, "
        "LIST(DISTINCT t.failure_type) AS failure_types, MIN(r.run_date) AS date_start, "
        "MAX(r.run_date) AS date_end "
        f"FROM {source}{where} GROUP BY t.failure_fingerprint "
        "HAVING COUNT(DISTINCT t.run_id) >= ?"
    )
    query_params = [*params, min_runs]
    return _run_report(
        database_path=database_path,
        name="repeated_test_failures_by_fingerprint",
        source_tables=("test_results", "pipeline_runs"),
        sql=f"SELECT * FROM ({grouped}) AS report ORDER BY run_count DESC, failure_fingerprint LIMIT ?",
        count_sql=f"SELECT COUNT(*) FROM ({grouped}) AS report",
        provenance_sql=_provenance_query(source, where, "r.run_date", "r.commit_sha"),
        params=query_params,
        count_params=query_params,
        provenance_params=params,
        filters={"test_id": test_id, "failure_fingerprint": failure_fingerprint,
                 "commit_sha": commit_sha, "platform": platform, "start_date": start_date,
                 "end_date": end_date, "min_runs": min_runs},
        requested_limit=requested,
        applied_limit=capped,
    )


def measurement_summary(
    database_path: str | Path,
    *,
    metric: str | None = None,
    stage: str | None = None,
    dimensions: Mapping[str, Any] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> QueryReport:
    """Summarize measurement values grouped by metric, stage, unit, dimensions."""
    metric = _optional_text(metric, "metric")
    stage = _optional_text(stage, "stage")
    start_date, end_date = _date_range(start_date, end_date)
    if dimensions is not None and not isinstance(dimensions, Mapping):
        raise TypeError("dimensions must be a mapping of scalar JSON values or None")
    dimension_filters: dict[str, Any] = {}
    if dimensions is not None:
        for key, value in dimensions.items():
            if not isinstance(key, str) or not _DIMENSION_KEY.fullmatch(key):
                raise ValueError("dimension keys must be simple identifier strings")
            if isinstance(value, (dict, list, tuple)):
                raise TypeError("dimension filter values must be JSON scalars")
            try:
                json.dumps(value, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise TypeError("dimension filter values must be finite JSON scalars") from exc
            dimension_filters[key] = value
    requested, capped = _limit(limit)
    conditions: list[tuple[str, Any]] = []
    if metric is not None:
        conditions.append(("m.metric = ?", metric))
    if stage is not None:
        conditions.append(("m.stage = ?", stage))
    if start_date is not None:
        conditions.append(("CAST(m.measured_at AS DATE) >= ?", start_date))
    if end_date is not None:
        conditions.append(("CAST(m.measured_at AS DATE) <= ?", end_date))
    params: list[Any] = []
    where, params = _where(conditions)
    for key, value in dimension_filters.items():
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        condition = "json_extract(m.dimensions, ?) = CAST(? AS JSON)"
        if not where:
            where = " WHERE " + condition
        else:
            where += " AND " + condition
        params.extend([f"$.{key}", encoded])
    source = "measurements AS m JOIN pipeline_runs AS r USING (run_id)"
    grouped = (
        "SELECT m.metric, m.stage, m.unit, CAST(m.dimensions AS VARCHAR) AS dimensions, "
        "COUNT(*) AS measurement_count, COUNT(m.value) AS value_count, AVG(m.value) AS mean_value, "
        "MIN(m.value) AS min_value, MAX(m.value) AS max_value, "
        "MIN(CAST(m.measured_at AS DATE)) AS date_start, MAX(CAST(m.measured_at AS DATE)) AS date_end "
        f"FROM {source}{where} GROUP BY m.metric, m.stage, m.unit, m.dimensions"
    )
    return _run_report(
        database_path=database_path,
        name="measurement_summary_by_metric_stage_dimensions",
        source_tables=("measurements", "pipeline_runs"),
        sql=f"SELECT * FROM ({grouped}) AS report ORDER BY metric, stage, unit, dimensions LIMIT ?",
        count_sql=f"SELECT COUNT(*) FROM ({grouped}) AS report",
        provenance_sql=_provenance_query(source, where, "m.measured_at", "r.commit_sha"),
        params=params,
        filters={"metric": metric, "stage": stage, "dimensions": dimension_filters,
                 "start_date": start_date, "end_date": end_date},
        requested_limit=requested,
        applied_limit=capped,
    )


def token_usage_summary(
    database_path: str | Path,
    *,
    stage: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    purpose: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> QueryReport:
    """Summarize llm_usage without imputing missing token counts or costs."""
    stage = _optional_text(stage, "stage")
    provider = _optional_text(provider, "provider")
    model = _optional_text(model, "model")
    purpose = _optional_text(purpose, "purpose")
    start_date, end_date = _date_range(start_date, end_date)
    requested, capped = _limit(limit)
    conditions: list[tuple[str, Any]] = []
    for field, value in (("u.stage", stage), ("u.provider", provider), ("u.model", model), ("u.purpose", purpose)):
        if value is not None:
            conditions.append((f"{field} = ?", value))
    if start_date is not None:
        conditions.append(("CAST(u.measured_at AS DATE) >= ?", start_date))
    if end_date is not None:
        conditions.append(("CAST(u.measured_at AS DATE) <= ?", end_date))
    where, params = _where(conditions)
    source = "llm_usage AS u JOIN pipeline_runs AS r USING (run_id)"
    grouped = (
        "SELECT u.stage, u.provider, u.model, u.purpose, COUNT(*) AS usage_records, "
        "COUNT(u.input_tokens) AS input_token_records, SUM(u.input_tokens) AS input_tokens, "
        "COUNT(u.output_tokens) AS output_token_records, SUM(u.output_tokens) AS output_tokens, "
        "COUNT(u.cached_input_tokens) AS cached_input_token_records, "
        "SUM(u.cached_input_tokens) AS cached_input_tokens, "
        "COUNT(u.estimated_cost_usd) AS cost_records, SUM(u.estimated_cost_usd) AS estimated_cost_usd, "
        "AVG(u.latency_ms) AS mean_latency_ms, MIN(CAST(u.measured_at AS DATE)) AS date_start, "
        "MAX(CAST(u.measured_at AS DATE)) AS date_end "
        f"FROM {source}{where} GROUP BY u.stage, u.provider, u.model, u.purpose"
    )
    return _run_report(
        database_path=database_path,
        name="llm_token_usage_summary",
        source_tables=("llm_usage", "pipeline_runs"),
        sql=f"SELECT * FROM ({grouped}) AS report ORDER BY stage, provider, model, purpose LIMIT ?",
        count_sql=f"SELECT COUNT(*) FROM ({grouped}) AS report",
        provenance_sql=_provenance_query(source, where, "u.measured_at", "r.commit_sha"),
        params=params,
        filters={"stage": stage, "provider": provider, "model": model, "purpose": purpose,
                 "start_date": start_date, "end_date": end_date},
        requested_limit=requested,
        applied_limit=capped,
    )
