from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from autobliteratus.observability.telemetry_store import initialize_database

duckdb = pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "telemetry_db.py"


@pytest.fixture
def database(tmp_path):
    path = initialize_database(tmp_path / "cli-telemetry.duckdb")
    connection = duckdb.connect(str(path))
    connection.executemany(
        "INSERT INTO pipeline_runs (run_id, run_date, commit_sha, platform, command, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("run-a", date(2026, 11, 20), "commit-a", "linux", "pytest", "failed"),
            ("run-b", date(2026, 11, 21), "commit-a", "linux", "pytest", "passed"),
        ],
    )
    connection.executemany(
        "INSERT INTO test_summaries (run_id, passed, failed, skipped, total) VALUES (?, ?, ?, ?, ?)",
        [("run-a", 5, 1, 0, 6), ("run-b", 6, 0, 0, 6)],
    )
    connection.executemany(
        "INSERT INTO test_results (run_id, test_id, attempt, outcome, failure_type, failure_fingerprint) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("run-a", "tests.test_failure", 1, "failed", "AssertionError", "fp-shared"),
            ("run-b", "tests.test_failure", 1, "failed", "AssertionError", "fp-shared"),
        ],
    )
    connection.executemany(
        "INSERT INTO measurements (measurement_id, run_id, stage, metric, value, unit, measured_at, dimensions) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("measure-a", "run-a", "encode", "latency_ms", 2.0, "ms", datetime(2026, 11, 20, tzinfo=timezone.utc), '{"backend":"cpu"}'),
            ("measure-b", "run-b", "encode", "latency_ms", 4.0, "ms", datetime(2026, 11, 21, tzinfo=timezone.utc), '{"backend":"cpu"}'),
        ],
    )
    connection.execute(
        "INSERT INTO llm_usage (usage_id, run_id, stage, provider, model, purpose, input_tokens, "
        "output_tokens, cached_input_tokens, estimated_cost_usd, latency_ms, measured_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "usage-a", "run-a", "generation", "provider-x", "model-y", "summary",
            100, 25, 10, 0.01, 400.0, datetime(2026, 11, 20, tzinfo=timezone.utc),
        ],
    )
    connection.close()
    return path


def _run_cli(database: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(CLI), "--db", str(database), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _json_result(database: Path, *arguments: str) -> dict:
    completed = _run_cli(database, *arguments)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_cli_import_to_query_reporting_workflow(tmp_path):
    database = tmp_path / "workflow.duckdb"
    initialized = _run_cli(database, "init")
    assert initialized.returncode == 0, initialized.stderr

    csv_source = tmp_path / "results.csv"
    csv_source.write_text(
        "date,commit,platform,command,passed,failed,skipped,notes\n"
        "2026-11-25,commit-e2e,linux,pytest -q,3,0,1,workflow fixture\n",
        encoding="utf-8",
    )
    first_import = _json_result(database, "import-results", str(csv_source))
    replay_import = _json_result(database, "import-results", str(csv_source))
    assert first_import["rows_seen"] == 1
    assert first_import["rows_inserted"] == 1
    assert replay_import["rows_inserted"] == 0

    connection = duckdb.connect(str(database), read_only=True)
    try:
        run_id = connection.execute(
            "SELECT run_id FROM pipeline_runs WHERE commit_sha = ?", ["commit-e2e"]
        ).fetchone()[0]
    finally:
        connection.close()

    event_source = tmp_path / "stage-events.jsonl"
    event_source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_id": "event-workflow-001",
                "run_id": run_id,
                "occurred_at": "2026-11-25T18:00:00Z",
                "component": "pipeline",
                "stage": "compile",
                "severity": "info",
                "event_type": "stage_completed",
                "message": "compile completed",
                "attributes": {"outcome": "completed", "count": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event_import = _json_result(database, "import-stage-events", str(event_source))
    event_replay = _json_result(database, "import-stage-events", str(event_source))
    assert event_import["records_inserted"] == 1
    assert event_replay["records_inserted"] == 0

    report = _json_result(
        database,
        "run-outcomes",
        "--commit-sha", "commit-e2e",
        "--status", "partial",
        "--start-date", "2026-11-25",
    )
    assert report["name"] == "run_outcomes_by_commit_platform_status"
    assert report["metadata"]["coverage"]["source_rows"] == 1
    assert report["metadata"]["provenance"]["commits"] == ["commit-e2e"]
    assert report["rows"] == [
        {
            "commit_sha": "commit-e2e",
            "platform": "linux",
            "status": "partial",
            "run_count": 1,
            "passed": 3,
            "failed": 0,
            "skipped": 1,
            "total": 4,
            "date_start": "2026-11-25",
            "date_end": "2026-11-25",
        }
    ]

    summary = _json_result(database, "summary")
    assert summary == {"runs": 1, "tests": 4, "failed": 0, "skipped": 1}

    connection = duckdb.connect(str(database), read_only=True)
    try:
        persisted_event = connection.execute(
            "SELECT run_id, event_type FROM pipeline_events WHERE event_id = ?",
            ["event-workflow-001"],
        ).fetchone()
    finally:
        connection.close()
    assert persisted_event == (run_id, "stage_completed")


def test_import_stage_events_command_reports_insert_counts_and_redacts(tmp_path, database):
    source = tmp_path / "stage-events.jsonl"
    event = {
        "schema_version": 1,
        "event_id": "event-cli-001",
        "run_id": "run-a",
        "occurred_at": "2026-11-20T12:00:00Z",
        "component": "pipeline",
        "stage": "compile",
        "severity": "error",
        "event_type": "stage_failed",
        "message": "request failed api_key=top-secret",
        "attributes": {"reason": "password='private-value'"},
    }
    source.write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = _json_result(database, "import-stage-events", str(source))

    assert result["records_seen"] == 1
    assert result["records_inserted"] == 1
    assert len(result["source_sha256"]) == 64
    connection = duckdb.connect(str(database), read_only=True)
    try:
        message, attributes = connection.execute(
            "SELECT message_redacted, CAST(attributes AS VARCHAR) FROM pipeline_events WHERE event_id = ?",
            ["event-cli-001"],
        ).fetchone()
    finally:
        connection.close()
    assert "top-secret" not in message
    assert "private-value" not in attributes
    assert "[REDACTED]" in message and "[REDACTED]" in attributes


def test_run_outcomes_command_forwards_filters_and_outputs_provenance(database):
    report = _json_result(
        database,
        "run-outcomes",
        "--commit-sha", "commit-a",
        "--status", "failed",
        "--start-date", "2026-11-20",
    )

    assert report["name"] == "run_outcomes_by_commit_platform_status"
    assert len(report["rows"]) == 1
    assert report["rows"][0]["run_count"] == 1
    assert report["metadata"]["filters"]["commit_sha"] == "commit-a"
    assert report["metadata"]["provenance"]["commits"] == ["commit-a"]


def test_repeated_failures_command_forwards_fingerprint_filters(database):
    report = _json_result(
        database,
        "repeated-test-failures",
        "--test-id", "tests.test_failure",
        "--min-runs", "2",
    )

    assert report["name"] == "repeated_test_failures_by_fingerprint"
    assert report["rows"][0]["failure_fingerprint"] == "fp-shared"
    assert report["rows"][0]["run_count"] == 2
    assert report["metadata"]["filters"]["min_runs"] == 2


def test_measurement_summary_command_parses_json_dimensions(database):
    report = _json_result(
        database,
        "measurement-summary",
        "--metric", "latency_ms",
        "--stage", "encode",
        "--dimensions", '{"backend":"cpu"}',
    )

    assert report["name"] == "measurement_summary_by_metric_stage_dimensions"
    assert len(report["rows"]) == 1
    assert report["rows"][0]["mean_value"] == 3.0
    assert report["metadata"]["filters"]["dimensions"] == {"backend": "cpu"}


def test_token_usage_summary_command_preserves_usage_values(database):
    report = _json_result(
        database,
        "token-usage-summary",
        "--provider", "provider-x",
        "--model", "model-y",
        "--purpose", "summary",
    )

    assert report["name"] == "llm_token_usage_summary"
    row = report["rows"][0]
    assert row["usage_records"] == 1
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 25
    assert report["metadata"]["filters"]["provider"] == "provider-x"


def test_query_cli_rejects_invalid_date_and_non_object_dimensions(database):
    bad_date = _run_cli(database, "run-outcomes", "--start-date", "not-a-date")
    bad_dimensions = _run_cli(database, "measurement-summary", "--dimensions", "[]")

    assert bad_date.returncode == 2
    assert "YYYY-MM-DD" in bad_date.stderr
    assert bad_dimensions.returncode == 2
    assert "JSON object" in bad_dimensions.stderr
