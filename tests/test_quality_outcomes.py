from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import duckdb
import pytest

from autobliteratus.observability import quality_outcomes as quality_importer
from autobliteratus.observability.telemetry_store import initialize_database


def _measurement(**changes):
    record = {
        "metric": "answer_quality",
        "value": 0.875,
        "unit": "score",
        "measured_at": "2026-11-25T12:30:00-07:00",
        "stage": "evaluation",
        "dimensions": {"dataset": "smoke-v1", "sample_count": 12},
    }
    record.update(changes)
    return record


def _report(*records, run_id="known-run"):
    return {"schema_version": 1, "run_id": run_id, "measurements": list(records)}


def _write_report(path, report):
    path.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    return path


def _seed_run(database, run_id="known-run"):
    initialize_database(database)
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            "INSERT INTO pipeline_runs (run_id, status) VALUES (?, 'passed')", [run_id]
        )
    finally:
        connection.close()


def test_import_maps_numeric_outcomes_into_measurements_and_preserves_missingness(tmp_path):
    without_dimensions = _measurement(
        metric="exact_match",
        value=1,
        unit="fraction",
        measured_at="2026-11-25T19:30:00Z",
        stage=None,
    )
    without_dimensions.pop("dimensions")
    source = _write_report(
        tmp_path / "quality.json",
        _report(_measurement(), without_dimensions),
    )
    database = tmp_path / "telemetry.duckdb"
    _seed_run(database)

    result = quality_importer.import_quality_outcomes(source, database)

    assert result.records_seen == 2
    assert result.records_inserted == 2
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            """SELECT run_id, stage, metric, value, unit, measured_at,
                      dimensions::VARCHAR, measurement_id
               FROM measurements ORDER BY metric"""
        ).fetchall()
    finally:
        connection.close()
    first, second = rows
    assert first[:5] == ("known-run", "evaluation", "answer_quality", 0.875, "score")
    assert first[5] == datetime(2026, 11, 25, 19, 30, tzinfo=timezone.utc)
    assert json.loads(first[6]) == {"dataset": "smoke-v1", "sample_count": 12.0}
    assert len(first[7]) == 64
    assert second[:5] == ("known-run", None, "exact_match", 1.0, "fraction")
    assert second[6] is None


def test_numeric_leading_hash_run_id_links_to_existing_run(tmp_path):
    run_id = "7" + "a" * 63
    database = tmp_path / "telemetry.duckdb"
    _seed_run(database, run_id)
    source = _write_report(
        tmp_path / "quality.json", _report(_measurement(), run_id=run_id)
    )

    result = quality_importer.import_quality_outcomes(source, database)

    assert result.records_inserted == 1
    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute(
            "SELECT run_id FROM measurements"
        ).fetchone()[0] == run_id
    finally:
        connection.close()


def test_identical_repeated_records_are_preserved_and_replays_are_idempotent(tmp_path):
    record = _measurement()
    source = _write_report(tmp_path / "quality.json", _report(record, record))
    database = tmp_path / "telemetry.duckdb"
    _seed_run(database)

    first = quality_importer.import_quality_outcomes(source, database)
    replay = quality_importer.import_quality_outcomes(source, database)

    assert (first.records_seen, first.records_inserted) == (2, 2)
    assert (replay.records_seen, replay.records_inserted) == (2, 0)
    assert replay.source_sha256 == first.source_sha256
    connection = duckdb.connect(str(database), read_only=True)
    try:
        count, distinct_ids = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT measurement_id) FROM measurements"
        ).fetchone()
    finally:
        connection.close()
    assert (count, distinct_ids) == (2, 2)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"run_id":"known-run","measurements":[],"extra":1}',
        '{"schema_version":2,"run_id":"known-run","measurements":[]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1,"unit":"score"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":true,"unit":"score","measured_at":"2026-01-01T00:00:00Z"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":NaN,"unit":"score","measured_at":"2026-01-01T00:00:00Z"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1e999,"unit":"score","measured_at":"2026-01-01T00:00:00Z"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1,"unit":"score","measured_at":"not-a-time"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00Z","raw_field":"not-allowed"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"api_key","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00Z"}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00Z","dimensions":{"prompt":"sensitive"}}]}',
        '{"schema_version":1,"run_id":"known-run","measurements":[{"metric":"q","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00Z","dimensions":{"status":{"nested":"value"}}}]}',
        '{"schema_version":1,"run_id":"known-run","run_id":"other","measurements":[]}',
    ],
)
def test_invalid_schema_values_and_duplicate_keys_are_rejected_before_database_creation(
    tmp_path, raw
):
    source = tmp_path / "invalid.json"
    source.write_text(raw, encoding="utf-8")
    database = tmp_path / "telemetry.duckdb"

    with pytest.raises(ValueError):
        quality_importer.import_quality_outcomes(source, database)

    assert not database.exists()


def test_unknown_run_id_is_rejected_without_inserting_rows(tmp_path):
    source = _write_report(tmp_path / "quality.json", _report(_measurement(), run_id="absent-run"))
    database = tmp_path / "telemetry.duckdb"
    _seed_run(database)

    with pytest.raises(ValueError, match="unknown run_id"):
        quality_importer.import_quality_outcomes(source, database)

    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 0
    finally:
        connection.close()


def test_prompt_like_free_text_is_rejected_and_never_persisted(tmp_path):
    record = _measurement(dimensions={"prompt": "private user prompt text"})
    source = _write_report(tmp_path / "private.json", _report(record))
    database = tmp_path / "telemetry.duckdb"
    _seed_run(database)

    with pytest.raises(ValueError, match="dimension key"):
        quality_importer.import_quality_outcomes(source, database)

    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 0
        contents = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='measurements'"
        ).fetchall()
        assert contents == [("measurements",)]
    finally:
        connection.close()
    assert "private user prompt text" not in database.read_bytes().decode("latin-1", errors="ignore")


def test_input_record_and_nesting_limits_are_enforced(tmp_path, monkeypatch):
    database = tmp_path / "oversized.duckdb"
    too_large = tmp_path / "too-large.json"
    too_large.write_bytes(b" " * (quality_importer.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="8 MiB"):
        quality_importer.import_quality_outcomes(too_large, database)
    assert not database.exists()

    oversized_record = tmp_path / "oversized-record.json"
    oversized_record.write_text(
        '{"schema_version":1,"run_id":"known-run","measurements":[{'
        + (" " * (quality_importer.MAX_RECORD_BYTES + 1))
        + '"metric":"q","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00Z"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="16 KiB"):
        quality_importer.import_quality_outcomes(oversized_record, database)

    nested = tmp_path / "nested.json"
    nested_dimensions = "{" + '"a":{' * 8 + '"x"' + "}" * 9
    nested.write_text(
        '{"schema_version":1,"run_id":"known-run","measurements":[{'
        '"metric":"q","value":1,"unit":"score","measured_at":"2026-01-01T00:00:00Z",'
        '"dimensions":' + nested_dimensions + '}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nesting limit"):
        quality_importer.import_quality_outcomes(nested, database)
    assert not database.exists()


def test_measurement_inserts_roll_back_as_one_transaction_on_write_failure(tmp_path, monkeypatch):
    source = _write_report(
        tmp_path / "quality.json",
        _report(_measurement(metric="first"), _measurement(metric="second")),
    )
    database = tmp_path / "telemetry.duckdb"
    _seed_run(database)
    actual_duckdb = duckdb

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection
            self.measurement_inserts = 0

        def execute(self, query, parameters=None):
            if query.lstrip().startswith("INSERT INTO measurements"):
                self.measurement_inserts += 1
                if self.measurement_inserts == 2:
                    raise RuntimeError("injected write failure")
            if parameters is None:
                return self.connection.execute(query)
            return self.connection.execute(query, parameters)

        def close(self):
            self.connection.close()

    def connect(path):
        return ConnectionProxy(actual_duckdb.connect(path))

    monkeypatch.setattr(quality_importer, "_duckdb", lambda: SimpleNamespace(connect=connect))

    with pytest.raises(RuntimeError, match="injected write failure"):
        quality_importer.import_quality_outcomes(source, database)

    connection = actual_duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 0
    finally:
        connection.close()
