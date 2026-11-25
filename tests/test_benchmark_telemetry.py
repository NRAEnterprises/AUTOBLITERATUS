from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import duckdb
import pytest

from autobliteratus.observability.telemetry_store import import_benchmark_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json"


def _csv_text(**changes: str) -> str:
    fields = [
        "schema_version", "workload_id", "engine", "engine_version", "run_id",
        "query_id", "latency_statistic", "correct", "latency_ms", "rows_returned",
        "input_tokens", "output_tokens", "context_bytes", "result_sha256", "cache_state",
        "hardware_fingerprint", "measured_at_utc",
    ]
    values = {
        "schema_version": "1", "workload_id": "workload-a", "engine": "duckdb",
        "engine_version": "2.0.1", "run_id": "known-run", "query_id": "q1",
        "latency_statistic": "median", "correct": "true", "latency_ms": "2.5",
        "rows_returned": "3", "input_tokens": "", "output_tokens": "",
        "context_bytes": "", "result_sha256": "", "cache_state": "unknown",
        "hardware_fingerprint": "", "measured_at_utc": "",
    }
    values.update(changes)
    fields.extend(key for key in changes if key not in fields)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(values)
    return stream.getvalue()


def test_import_json_flattens_summaries_and_preserves_source_time(tmp_path):
    database = tmp_path / "telemetry.duckdb"
    result = import_benchmark_report(FIXTURE, database)
    replay = import_benchmark_report(FIXTURE, database)
    assert result.inserted and not replay.inserted
    assert result.report_kind == "telemetry_engine_json_v1"
    assert result.records_seen == result.rows_inserted == 27
    assert replay.rows_inserted == 0

    connection = duckdb.connect(str(database), read_only=True)
    try:
        summary = connection.execute(
            "SELECT latency_statistic, COUNT(*) FROM benchmark_measurements "
            "GROUP BY latency_statistic ORDER BY latency_statistic"
        ).fetchall()
        sample = connection.execute(
            "SELECT run_id, measured_at, latency_statistic, cache_state, correct "
            "FROM benchmark_measurements WHERE engine='csv_python' "
            "AND query_id='q1_summary_totals' ORDER BY latency_statistic"
        ).fetchall()
        provenance = connection.execute(
            "SELECT source_sha256, record_count FROM benchmark_imports"
        ).fetchone()
    finally:
        connection.close()

    assert summary == [("first_after_parity", 9), ("median", 9), ("p95", 9)]
    assert all(row[0] is None and row[1] is not None for row in sample)
    first = next(row for row in sample if row[2] == "first_after_parity")
    assert first[3] == "after_parity" and first[4] is True
    assert provenance == (result.source_sha256, 27)


def test_import_csv_keeps_optional_metadata_null_and_links_only_existing_runs(tmp_path):
    source = tmp_path / "benchmark.csv"
    source.write_text(_csv_text(measured_at_utc="2026-11-20T15:00:00-07:00"), encoding="utf-8")
    database = tmp_path / "telemetry.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute((ROOT / "src/autobliteratus/observability/schema-v1.sql").read_text())
    connection.execute(
        "INSERT INTO pipeline_runs (run_id, status) VALUES ('known-run', 'passed')"
    )
    connection.close()

    result = import_benchmark_report(source, database)
    assert result.report_kind == "query_benchmark_csv_v1"
    assert result.records_seen == result.rows_inserted == 1
    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute(
            "SELECT run_id, measured_at, input_tokens, context_bytes, cache_state "
            "FROM benchmark_measurements"
        ).fetchone()
    finally:
        connection.close()
    assert row[0] == "known-run"
    assert row[1] is not None
    assert row[2:4] == (None, None)
    assert row[4] == "unknown"


def test_import_csv_does_not_fabricate_run_or_measurement_time(tmp_path):
    source = tmp_path / "benchmark.csv"
    source.write_text(_csv_text(run_id="missing-run", measured_at_utc=""), encoding="utf-8")
    database = tmp_path / "telemetry.duckdb"
    import_benchmark_report(source, database)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute("SELECT run_id, measured_at FROM benchmark_measurements").fetchone()
    finally:
        connection.close()
    assert row == (None, None)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"correct": "not-a-bool"}, "true or false"),
        ({"latency_ms": "nan"}, "finite non-negative"),
        ({"latency_statistic": "cold"}, "latency_statistic"),
        ({"schema_version": "2"}, "schema_version"),
        ({"secret_field": "should-not-import"}, "columns invalid"),
        ({"measured_at_utc": "2026-11-22T06:38:11"}, "include a timezone"),
    ],
)
def test_import_csv_rejects_invalid_contract_before_database_creation(tmp_path, change, message):
    source = tmp_path / "bad.csv"
    source.write_text(_csv_text(**change), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        import_benchmark_report(source, tmp_path / "telemetry.duckdb")
    assert not (tmp_path / "telemetry.duckdb").exists()


def test_import_json_requires_parity_consistency_and_hashes(tmp_path):
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["parity"]["all_engines_match"] = False
    path = tmp_path / "failed.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="parity validation"):
        import_benchmark_report(path, tmp_path / "telemetry.duckdb")

    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    query = report["engines"]["duckdb"]["queries"]["q1_summary_totals"]
    query["result_sha256"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hash conflicts"):
        import_benchmark_report(path, tmp_path / "telemetry.duckdb")

    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["engines"]["sqlite"]["queries"].pop("q3_filtered_run_detail_passed_gte_35")
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="query set conflicts"):
        import_benchmark_report(path, tmp_path / "telemetry.duckdb")
    assert not (tmp_path / "telemetry.duckdb").exists()


def test_import_rejects_duplicate_keys_malformed_csv_and_oversized_input(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"benchmark":"x","benchmark":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        import_benchmark_report(duplicate, tmp_path / "duplicate.duckdb")

    malformed = tmp_path / "malformed.csv"
    malformed.write_text(_csv_text() + '"unterminated\n', encoding="utf-8")
    with pytest.raises(ValueError):
        import_benchmark_report(malformed, tmp_path / "malformed.duckdb")

    large = tmp_path / "large.csv"
    large.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="16 MiB"):
        import_benchmark_report(large, tmp_path / "large.duckdb")


def test_import_does_not_store_unknown_json_fields_or_raw_warning_text(tmp_path):
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["warning"] = "api_key=not-to-persist"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    database = tmp_path / "telemetry.duckdb"
    import_benchmark_report(path, database)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute("SELECT * FROM benchmark_measurements").fetchall()
        columns = [item[0] for item in connection.description]
    finally:
        connection.close()
    persisted = json.dumps([columns, rows], default=str)
    assert "not-to-persist" not in persisted
    assert "warning" not in columns
