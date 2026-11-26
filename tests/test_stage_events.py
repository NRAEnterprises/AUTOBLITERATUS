from __future__ import annotations

import json

import pytest

from autobliteratus.observability.stage_events import import_stage_events
from autobliteratus.observability.telemetry_store import initialize_database

pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")


RUN_ID = "run-stage-events-001"


def _record(**overrides):
    record = {
        "schema_version": 1,
        "event_id": "event-stage-001",
        "run_id": RUN_ID,
        "occurred_at": "2026-11-25T18:40:00Z",
        "component": "pipeline",
        "stage": "compile",
        "severity": "info",
        "event_type": "stage_started",
        "message": "Compile started",
        "correlation_id": "corr-001",
        "duration_ms": 12.5,
        "attributes": {"phase": "prepare", "attempt": 1},
    }
    record.update(overrides)
    return record


def _write_jsonl(path, *records):
    path.write_bytes(b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    ))


def _database(tmp_path):
    path = tmp_path / "events.duckdb"
    initialize_database(path)
    import duckdb

    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "INSERT INTO pipeline_runs (run_id, status) VALUES (?, ?)",
            [RUN_ID, "running"],
        )
    finally:
        connection.close()
    return path


def _event_row(database, event_id="event-stage-001"):
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(
            """SELECT event_id, run_id, component, stage, severity, event_type,
                      message_redacted, correlation_id, duration_ms, attributes
               FROM pipeline_events WHERE event_id = ?""",
            [event_id],
        ).fetchone()
    finally:
        connection.close()


def test_valid_v1_import_writes_only_contract_fields(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "stage-events.jsonl"
    _write_jsonl(source, _record())

    result = import_stage_events(source, database)

    row = _event_row(database)
    assert result.records_seen == 1
    assert result.records_inserted == 1
    assert len(result.source_sha256) == 64
    assert row[:6] == (
        "event-stage-001", RUN_ID, "pipeline", "compile", "info", "stage_started"
    )
    assert row[6:9] == ("Compile started", "corr-001", 12.5)
    assert json.loads(row[9]) == {"attempt": 1, "phase": "prepare"}


def test_duplicate_replay_preserves_existing_row(tmp_path):
    database = _database(tmp_path)
    first = tmp_path / "first.jsonl"
    changed = tmp_path / "replay.jsonl"
    _write_jsonl(first, _record())
    _write_jsonl(changed, _record(message="new payload", severity="error"))

    assert import_stage_events(first, database).records_inserted == 1
    replay = import_stage_events(changed, database)

    assert replay.records_seen == 1
    assert replay.records_inserted == 0
    assert _event_row(database)[4] == "info"
    assert _event_row(database)[6] == "Compile started"


def test_unknown_run_is_rejected_and_transaction_does_not_insert(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "missing-run.jsonl"
    _write_jsonl(
        source,
        _record(),
        _record(event_id="event-stage-002", run_id="run-does-not-exist"),
    )

    with pytest.raises(ValueError, match="unknown run_id"):
        import_stage_events(source, database)
    assert _event_row(database) is None
    assert _event_row(database, "event-stage-002") is None


def test_version_unknown_fields_and_invalid_contract_values_rejected(tmp_path):
    database = _database(tmp_path)
    cases = [
        (_record(schema_version=2), "schema_version"),
        (_record(schema_version=True), "schema_version"),
        (_record(extra_payload="not allowed"), "fields"),
        (_record(event_id=""), "event_id"),
        (_record(occurred_at="2026-11-25T18:40:00"), "occurred_at"),
        (_record(severity="notice"), "severity"),
        (_record(duration_ms=-0.1), "duration_ms"),
        (_record(duration_ms=True), "duration_ms"),
    ]
    source = tmp_path / "invalid.jsonl"
    for record, message in cases:
        _write_jsonl(source, record)
        with pytest.raises(ValueError, match=message):
            import_stage_events(source, database)
    assert _event_row(database) is None


def test_malformed_duplicate_key_and_truncated_input_rejected(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "malformed.jsonl"
    source.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        import_stage_events(source, database)
    source.write_bytes(b'{"schema_version":1')
    with pytest.raises(ValueError, match="truncated"):
        import_stage_events(source, database)
    source.write_bytes(b"\n")
    with pytest.raises(ValueError, match="blank line"):
        import_stage_events(source, database)
    assert _event_row(database) is None


def test_oversized_line_and_total_input_rejected(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "oversized.jsonl"
    source.write_bytes((b" " * (16 * 1024)) + b"\n")
    with pytest.raises(ValueError, match="16 KiB"):
        import_stage_events(source, database)
    source.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="8 MiB"):
        import_stage_events(source, database)
    assert _event_row(database) is None


def test_credentials_are_redacted_from_message_and_attribute_values(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "redacted.jsonl"
    _write_jsonl(
        source,
        _record(
            message="request failed api_key=top-secret Bearer abc.def.token",
            attributes={"reason": "password='secret-value'"},
        ),
    )

    import_stage_events(source, database)

    row = _event_row(database)
    stored = row[6] + json.dumps(json.loads(row[9]), sort_keys=True)
    assert "top-secret" not in stored
    assert "abc.def.token" not in stored
    assert "secret-value" not in stored
    assert "[REDACTED]" in stored


def test_attributes_are_allowlisted_flat_and_bounded(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "bad-attributes.jsonl"
    cases = [
        ({f"key_{index}": index for index in range(17)}, "16-key"),
        ({"unknown": "value"}, "unsupported attribute"),
        ({"status": {"nested": "value"}}, "scalar"),
        ({"reason": "x" * 513}, "too long"),
    ]
    for attributes, message in cases:
        _write_jsonl(source, _record(attributes=attributes))
        with pytest.raises(ValueError, match=message):
            import_stage_events(source, database)
    assert _event_row(database) is None
