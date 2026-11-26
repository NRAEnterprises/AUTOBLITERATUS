"""Strict bounded v1 JSON importer for numeric quality outcomes.

Contract: the UTF-8 JSON source is exactly an object with ``schema_version``
(equal to 1), ``run_id`` (a safe identifier for an already-existing
``pipeline_runs`` row), and a non-empty ``measurements`` array. Every measurement
is exactly an object with required ``metric``, finite numeric ``value``,
explicit ``unit``, and explicit timezone-aware ISO-8601 ``measured_at``; its
only optional fields are ``stage`` and ``dimensions``. A stage is either absent,
null, or a short safe label. Dimensions are a flat mapping of at most 16 safe
keys to bounded string, finite numeric, boolean, or null scalar values. Missing
fields stay missing/NULL; this importer never supplies timestamps or units.
Unknown fields, duplicate JSON keys, non-standard/non-finite numbers, unsafe or
unbounded labels, malformed timestamps, nested/non-scalar dimensions, unknown
run IDs, nesting deeper than 8, input above 8 MiB, records above 16 KiB, and
reports above 10,000 measurements are rejected. No free-form text is accepted
or persisted, and raw prompts, logs, and other sensitive text are out of
contract. Each row is report-only: it writes only to ``measurements`` and does
not change a run, model, or pipeline state. Occurrence-qualified content IDs
preserve identical repeats within a report while making report replays idempotent.
"""

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
MAX_RECORD_BYTES = 16 * 1024
MAX_RECORDS = 10_000
MAX_DIMENSIONS = 16
MAX_DIMENSION_BYTES = 4 * 1024
MAX_LABEL_LENGTH = 64

_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+/-]{0,63}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_DIMENSION_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SENSITIVE_LABEL_RE = re.compile(
    r"(?i)(?:api[_-]?key|password|passwd|secret|authorization|bearer|credential|"
    r"raw[_-]?prompt|(?:^|[_-])prompt(?:$|[_-])|(?:^|[_-])(?:stdout|stderr|log)(?:$|[_-]))"
)
_SENSITIVE_DIMENSION_KEY_RE = re.compile(
    _SENSITIVE_LABEL_RE.pattern[4:] + r"|message|command|notes?|free[_-]?text",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityOutcomeImportResult:
    records_seen: int
    records_inserted: int
    source_sha256: str


def _label(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_LABEL_LENGTH
        or not _LABEL_RE.fullmatch(value)
        or redact_text(value) != value
        or _SENSITIVE_LABEL_RE.search(value)
    ):
        raise ValueError(f"Quality outcome {field} must be a short safe label")
    return value


def _run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _RUN_ID_RE.fullmatch(value)
        or redact_text(value) != value
        or _SENSITIVE_LABEL_RE.search(value)
    ):
        raise ValueError("Quality outcome run_id must be a safe identifier")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Quality outcome {field} must be a finite number")  # noqa: TRY004
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"Quality outcome {field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"Quality outcome {field} must be a finite number")
    return result


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("Quality outcome measured_at must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(
            "Quality outcome measured_at must be a timezone-aware ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Quality outcome measured_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _dimensions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Quality outcome dimensions must be an object")  # noqa: TRY004
    if len(value) > MAX_DIMENSIONS:
        raise ValueError("Quality outcome dimensions exceed the 16-key limit")
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not _DIMENSION_KEY_RE.fullmatch(key)
            or redact_text(key) != key
            or _SENSITIVE_DIMENSION_KEY_RE.search(key)
        ):
            raise ValueError("Quality outcome contains an unsafe dimension key")
        if item is None or isinstance(item, bool):
            safe[key] = item
        elif isinstance(item, str):
            if (
                not 1 <= len(item) <= MAX_LABEL_LENGTH
                or not _LABEL_RE.fullmatch(item)
                or redact_text(item) != item
                or _SENSITIVE_LABEL_RE.search(item)
            ):
                raise ValueError("Quality outcome dimension strings must be short safe labels")
            safe[key] = item
        elif isinstance(item, (int, float)):
            safe[key] = _number(item, f"dimensions.{key}")
        else:
            raise ValueError("Quality outcome dimensions must contain only scalar values")
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(encoded.encode("utf-8")) > MAX_DIMENSION_BYTES:
        raise ValueError("Quality outcome dimensions exceed the 4096-byte limit")
    return safe


def _measurement_record_lengths(text: str) -> list[int]:
    """Return UTF-8 byte lengths of the source measurement objects."""
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_json_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("Invalid JSON numeric constant")
        ),
    )

    def skip_space(position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    position = skip_space(0)
    if position >= len(text) or text[position] != "{":
        return []
    position += 1
    while True:
        position = skip_space(position)
        if position >= len(text) or text[position] == "}":
            return []
        key, position = decoder.raw_decode(text, position)
        position = skip_space(position)
        if position >= len(text) or text[position] != ":":
            return []
        position = skip_space(position + 1)
        if key != "measurements":
            _value, position = decoder.raw_decode(text, position)
        else:
            if position >= len(text) or text[position] != "[":
                return []
            lengths: list[int] = []
            position += 1
            while True:
                position = skip_space(position)
                if position >= len(text):
                    return lengths
                if text[position] == "]":
                    return lengths
                start = position
                _value, position = decoder.raw_decode(text, position)
                lengths.append(len(text[start:position].encode("utf-8")))
                position = skip_space(position)
                if position < len(text) and text[position] == ",":
                    position += 1
                    continue
                if position < len(text) and text[position] == "]":
                    return lengths
                return lengths
        position = skip_space(position)
        if position < len(text) and text[position] == ",":
            position += 1
            continue
        return []


def _validate_report(raw: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise ValueError("Quality outcome report must be a JSON object")  # noqa: TRY004
    required = {"schema_version", "run_id", "measurements"}
    if set(raw) != required:
        raise ValueError("Quality outcome report fields do not match the v1 contract")
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise ValueError("Unsupported quality outcome schema_version")
    run_id = _run_id(raw["run_id"])
    records = raw["measurements"]
    if not isinstance(records, list) or not records:
        raise ValueError("Quality outcome measurements must be a non-empty array")
    if len(records) > MAX_RECORDS:
        raise ValueError("Quality outcome report exceeds the 10000-measurement limit")

    validated: list[dict[str, Any]] = []
    expected = {"metric", "value", "unit", "measured_at"}
    allowed = expected | {"stage", "dimensions"}
    for record in records:
        if not isinstance(record, dict) or not expected.issubset(record) or set(record) - allowed:
            raise ValueError("Quality outcome measurement fields do not match the v1 contract")
        stage = record.get("stage")
        if stage is not None:
            stage = _label(stage, "stage")
        dimensions = _dimensions(record["dimensions"]) if "dimensions" in record else None
        validated.append(
            {
                "metric": _label(record["metric"], "metric"),
                "value": _number(record["value"], "value"),
                "unit": _label(record["unit"], "unit"),
                "measured_at": _timestamp(record["measured_at"]),
                "stage": stage,
                "dimensions": dimensions,
            }
        )
    return run_id, validated


def _read_report(source_path: Path) -> tuple[bytes, str, list[dict[str, Any]]]:
    with source_path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Quality outcome JSON exceeds the 8 MiB input limit")
    try:
        _check_json_nesting(raw, maximum_depth=8)
        decoded = raw.decode("utf-8")
        record_lengths = _measurement_record_lengths(decoded)
        report = json.loads(
            decoded,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("Invalid JSON numeric constant")
            ),
        )
        run_id, records = _validate_report(report)
        if len(record_lengths) != len(records):
            raise ValueError("Quality outcome measurement array is malformed")
        if any(length > MAX_RECORD_BYTES for length in record_lengths):
            raise ValueError("Quality outcome measurement exceeds the 16 KiB record limit")
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Quality outcome source must be strict UTF-8 JSON") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid quality outcome report: {exc}") from exc

    return raw, run_id, records


def _measurement_id(run_id: str, record: dict[str, Any], occurrence: int) -> str:
    identity = {
        "run_id": run_id,
        "stage": record["stage"],
        "metric": record["metric"],
        "value": record["value"],
        "unit": record["unit"],
        "measured_at": record["measured_at"].isoformat(),
        "dimensions": record["dimensions"],
        "occurrence": occurrence,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(("autobliteratus-quality-outcome-v1\0" + canonical).encode()).hexdigest()


@private_database_operation(1)
def import_quality_outcomes(
    json_path: str | Path, database_path: str | Path
) -> QualityOutcomeImportResult:
    """Import a strict quality-outcome JSON v1 report into ``measurements``."""
    raw, run_id, records = _read_report(Path(json_path))
    source_sha256 = hashlib.sha256(raw).hexdigest()
    database = initialize_database(database_path)
    connection = _duckdb().connect(str(database))
    inserted = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        if connection.execute(
            "SELECT 1 FROM pipeline_runs WHERE run_id = ?", [run_id]
        ).fetchone() is None:
            raise ValueError("Quality outcome report references an unknown run_id")
        occurrences: dict[str, int] = {}
        for record in records:
            identity = _measurement_id(run_id, record, 0)
            occurrence = occurrences.get(identity, 0)
            occurrences[identity] = occurrence + 1
            measurement_id = _measurement_id(run_id, record, occurrence)
            dimensions = (
                json.dumps(
                    record["dimensions"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                if record["dimensions"] is not None
                else None
            )
            result = connection.execute(
                """INSERT INTO measurements
                   (measurement_id, run_id, stage, metric, value, unit, measured_at, dimensions)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON)
                   ON CONFLICT (measurement_id) DO NOTHING
                   RETURNING measurement_id""",
                [
                    measurement_id,
                    run_id,
                    record["stage"],
                    record["metric"],
                    record["value"],
                    record["unit"],
                    record["measured_at"],
                    dimensions,
                ],
            ).fetchone()
            inserted += result is not None
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return QualityOutcomeImportResult(len(records), int(inserted), source_sha256)
