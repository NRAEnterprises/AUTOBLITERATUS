"""Bounded, versioned JSONL importer for sanitized pipeline stage events."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autobliteratus.observability.telemetry_store import (
    _check_json_nesting,
    _duckdb,
    _strict_json_object,
    initialize_database,
    private_database_operation,
    redact_text,
)

SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024
MAX_RECORDS = 10_000
MAX_ATTRIBUTES = 16
MAX_ATTRIBUTE_BYTES = 4 * 1024
MAX_TEXT_LENGTH = 512

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_SEVERITIES = {"debug", "info", "warning", "error", "critical"}
_ATTRIBUTE_KEYS = {
    "status", "phase", "outcome", "count", "attempt", "retry_count",
    "backend", "metric", "unit", "cache_state", "model", "provider",
    "operation", "reason", "code", "source", "bytes",
}
_REQUIRED_FIELDS = {
    "schema_version", "event_id", "run_id", "occurred_at", "component",
    "severity", "event_type",
}
_OPTIONAL_FIELDS = {"stage", "message", "correlation_id", "duration_ms", "attributes"}


@dataclass(frozen=True)
class StageEventImportResult:
    records_seen: int
    records_inserted: int
    source_sha256: str


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("Stage event occurred_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Stage event occurred_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Stage event occurred_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _bounded_text(value: Any, field: str, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or not _ID_RE.fullmatch(value):
        raise ValueError(f"Stage event {field} is invalid")
    return value


def _safe_attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Stage event attributes must be an object")  # noqa: TRY004
    if len(value) > MAX_ATTRIBUTES:
        raise ValueError("Stage event attributes exceed the 16-key limit")
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in _ATTRIBUTE_KEYS:
            raise ValueError("Stage event contains an unsupported attribute key")
        if isinstance(item, str):
            if len(item) > MAX_TEXT_LENGTH or any(ord(char) < 32 and char != "\t" for char in item):
                raise ValueError(f"Stage event attribute {key} is invalid or too long")
            safe[key] = redact_text(item)
        elif isinstance(item, bool) or item is None:
            safe[key] = item
        elif isinstance(item, int):
            if not 0 <= item <= 2**63 - 1:
                raise ValueError(f"Stage event attribute {key} is out of range")
            safe[key] = item
        elif isinstance(item, float):
            if not math.isfinite(item) or item < 0:
                raise ValueError(f"Stage event attribute {key} must be finite and nonnegative")
            safe[key] = item
        else:
            # No arrays, nested objects, or arbitrary JSON values are persisted.
            raise ValueError(f"Stage event attribute {key} must be a scalar")
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(encoded.encode("utf-8")) > MAX_ATTRIBUTE_BYTES:
        raise ValueError("Stage event attributes exceed the 4096-byte limit")
    return safe


def _validate_record(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("Each stage event record must be a JSON object")
    keys = set(raw)
    if not _REQUIRED_FIELDS.issubset(keys) or keys - _REQUIRED_FIELDS - _OPTIONAL_FIELDS:
        raise ValueError("Stage event fields do not match the v1 contract")
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise ValueError("Unsupported stage event schema_version")
    event_id = _bounded_text(raw["event_id"], "event_id")
    run_id = _bounded_text(raw["run_id"], "run_id")
    occurred_at = _timestamp(raw["occurred_at"])
    component = raw["component"]
    if not isinstance(component, str) or not _NAME_RE.fullmatch(component):
        raise ValueError("Stage event component is invalid")
    stage = raw.get("stage")
    if stage is not None and (not isinstance(stage, str) or not _NAME_RE.fullmatch(stage)):
        raise ValueError("Stage event stage is invalid")
    severity = raw["severity"]
    if not isinstance(severity, str) or severity not in _SEVERITIES:
        raise ValueError("Stage event severity is invalid")
    event_type = raw["event_type"]
    if not isinstance(event_type, str) or not _NAME_RE.fullmatch(event_type):
        raise ValueError("Stage event event_type is invalid")
    message = raw.get("message")
    if message is not None:
        if (
            not isinstance(message, str)
            or len(message) > MAX_TEXT_LENGTH
            or any(ord(char) < 32 and char != "\t" for char in message)
        ):
            raise ValueError("Stage event message is invalid or too long")
        message = redact_text(message)
    correlation_id = raw.get("correlation_id")
    if correlation_id is not None:
        correlation_id = _bounded_text(correlation_id, "correlation_id")
    duration = raw.get("duration_ms")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("Stage event duration_ms must be a nonnegative finite number")
        try:
            duration = float(duration)
        except OverflowError as exc:
            raise ValueError("Stage event duration_ms is out of range") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Stage event duration_ms must be a nonnegative finite number")
    attributes = _safe_attributes(raw.get("attributes", {}))
    return {
        "event_id": event_id,
        "run_id": run_id,
        "occurred_at": occurred_at,
        "component": component,
        "stage": stage,
        "severity": severity,
        "event_type": event_type,
        "message_redacted": message,
        "correlation_id": correlation_id,
        "duration_ms": duration,
        "attributes": json.dumps(attributes, sort_keys=True, separators=(",", ":")),
    }


def _read_records(source_path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    with source_path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Stage event JSONL exceeds the 8 MiB input limit")
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Stage event JSONL has a truncated final record")
    records: list[dict[str, Any]] = []
    lines = raw.split(b"\n")
    for line_number, line in enumerate(lines[:-1], start=1):
        if not line:
            raise ValueError(f"Stage event JSONL contains a blank line at {line_number}")
        encoded_length = len(line) + 1
        if encoded_length > MAX_LINE_BYTES:
            raise ValueError(f"Stage event JSONL line {line_number} exceeds the 16 KiB limit")
        if len(records) >= MAX_RECORDS:
            raise ValueError("Stage event JSONL exceeds the 10000-record limit")
        try:
            _check_json_nesting(line, maximum_depth=16)
            text = line.decode("utf-8")
            raw_record = json.loads(
                text,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("Invalid JSON constant")),
            )
            records.append(_validate_record(raw_record))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"Stage event JSONL line {line_number} is malformed") from exc
        except ValueError as exc:
            raise ValueError(f"Stage event JSONL line {line_number}: {exc}") from exc
    return raw, records


@private_database_operation(1)
def import_stage_events(jsonl_path: str | Path, database_path: str | Path) -> StageEventImportResult:
    """Import strict v1 stage-event JSONL transactionally and idempotently.

    Each line has required ``schema_version``, ``event_id``, ``run_id``,
    ``occurred_at``, ``component``, ``severity`` and ``event_type`` fields.
    Optional fields are ``stage``, ``message``, ``correlation_id``,
    ``duration_ms`` and ``attributes``. Attributes are flat scalar mappings
    limited to the module's explicit allowlist. Unknown fields are rejected.
    """
    raw, records = _read_records(Path(jsonl_path))
    source_sha256 = hashlib.sha256(raw).hexdigest()
    database = initialize_database(database_path)
    connection = _duckdb().connect(str(database))
    inserted = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        for record in records:
            exists = connection.execute(
                "SELECT 1 FROM pipeline_runs WHERE run_id = ?", [record["run_id"]]
            ).fetchone()
            if exists is None:
                raise ValueError(f"Stage event references unknown run_id: {record['run_id']}")
            result = connection.execute(
                """INSERT INTO pipeline_events
                   (event_id, run_id, occurred_at, component, stage, severity,
                    event_type, message_redacted, correlation_id, duration_ms, attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON)
                   ON CONFLICT (event_id) DO NOTHING
                   RETURNING event_id""",
                [
                    record["event_id"], record["run_id"], record["occurred_at"],
                    record["component"], record["stage"], record["severity"],
                    record["event_type"], record["message_redacted"],
                    record["correlation_id"], record["duration_ms"], record["attributes"],
                ],
            ).fetchone()
            inserted += result is not None
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return StageEventImportResult(len(records), inserted, source_sha256)
