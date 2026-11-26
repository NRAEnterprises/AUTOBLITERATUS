from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")

import duckdb

from autobliteratus.observability import query_pack
from autobliteratus.observability.query_pack import (
    MAX_REPORT_ROWS,
    measurement_summary,
    quality_latency_pareto_report,
    repeated_test_failures,
    run_outcomes,
    token_usage_summary,
)
from autobliteratus.observability.telemetry_store import initialize_database


@pytest.fixture
def database(tmp_path):
    path = initialize_database(tmp_path / "query-pack.duckdb")
    connection = duckdb.connect(str(path))
    runs = [
        ("run-a", date(2026, 11, 20), "commit-a", "linux", "pytest", "failed"),
        ("run-b", date(2026, 11, 21), "commit-a", "linux", "pytest", "passed"),
        ("run-c", date(2026, 11, 22), "commit-b", "macos", "pytest", "failed"),
    ]
    connection.executemany(
        "INSERT INTO pipeline_runs (run_id, run_date, commit_sha, platform, command, status) "
        "VALUES (?, ?, ?, ?, ?, ?)", runs
    )
    connection.executemany(
        "INSERT INTO test_summaries (run_id, passed, failed, skipped, total) VALUES (?, ?, ?, ?, ?)",
        [("run-a", 5, 1, 0, 6), ("run-b", 6, 0, 0, 6), ("run-c", 4, 2, 1, 7)],
    )
    connection.executemany(
        "INSERT INTO test_results (run_id, test_id, attempt, outcome, failure_type, failure_fingerprint) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("run-a", "tests.test_alpha", 1, "failed", "AssertionError", "fp-repeat"),
            ("run-b", "tests.test_alpha", 1, "failed", "AssertionError", "fp-repeat"),
            ("run-c", "tests.test_beta", 1, "failed", "Timeout", "fp-one-off"),
            ("run-c", "tests.test_ok", 1, "passed", None, None),
        ],
    )
    connection.executemany(
        "INSERT INTO measurements (measurement_id, run_id, stage, metric, value, unit, measured_at, dimensions) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("measure-a", "run-a", "encode", "latency_ms", 12.0, "ms", datetime(2026, 11, 20, tzinfo=timezone.utc), '{"backend":"cpu","batch":2}'),
            ("measure-b", "run-b", "encode", "latency_ms", 18.0, "ms", datetime(2026, 11, 21, tzinfo=timezone.utc), '{"backend":"cpu","batch":2}'),
            ("measure-c", "run-c", "decode", "latency_ms", None, "ms", datetime(2026, 11, 22, tzinfo=timezone.utc), '{"backend":"gpu","batch":4}'),
        ],
    )
    connection.executemany(
        "INSERT INTO llm_usage (usage_id, run_id, stage, provider, model, purpose, input_tokens, "
        "output_tokens, cached_input_tokens, estimated_cost_usd, latency_ms, measured_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("usage-a", "run-a", "generation", "provider-x", "model-y", "summary", 100, 25, 10, 0.01, 400.0, datetime(2026, 11, 20, tzinfo=timezone.utc)),
            ("usage-b", "run-b", "generation", "provider-x", "model-y", "summary", None, None, None, None, None, datetime(2026, 11, 21, tzinfo=timezone.utc)),
            ("usage-c", "run-c", "generation", "provider-z", "model-z", "completion", None, None, None, None, None, datetime(2026, 11, 22, tzinfo=timezone.utc)),
        ],
    )
    connection.close()
    return path


def test_run_outcomes_group_and_filter_with_provenance(database):
    report = run_outcomes(
        database, commit_sha="commit-a", platform="linux", start_date=date(2026, 11, 20)
    )

    assert report.name == "run_outcomes_by_commit_platform_status"
    assert len(report.rows) == 2
    failed = next(row for row in report.rows if row["status"] == "failed")
    assert (failed["run_count"], failed["passed"], failed["failed"], failed["total"]) == (1, 5, 1, 6)
    assert report.metadata["source_tables"] == ["pipeline_runs", "test_summaries"]
    assert report.metadata["coverage"]["source_rows"] == 2
    assert report.metadata["provenance"]["commits"] == ["commit-a"]
    assert report.metadata["provenance"]["date_start"] == date(2026, 11, 20)
    assert report.metadata["filters"]["platform"] == "linux"


def test_repeated_test_failures_uses_distinct_runs_and_fingerprints(database):
    report = repeated_test_failures(database, test_id="tests.test_alpha")

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row["failure_fingerprint"] == "fp-repeat"
    assert row["failure_occurrences"] == 2
    assert row["run_count"] == 2
    assert set(row["test_ids"]) == {"tests.test_alpha"}
    assert report.metadata["coverage"]["source_rows"] == 2
    assert report.metadata["provenance"]["commits"] == ["commit-a"]


def test_measurement_summary_groups_and_filters_dimensions(database):
    report = measurement_summary(
        database, metric="latency_ms", stage="encode", dimensions={"backend": "cpu", "batch": 2}
    )

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row["measurement_count"] == row["value_count"] == 2
    assert row["mean_value"] == 15.0
    assert row["min_value"] == 12.0 and row["max_value"] == 18.0
    assert report.metadata["coverage"]["source_rows"] == 2
    assert report.metadata["provenance"]["commits"] == ["commit-a"]
    unavailable = measurement_summary(database, metric="latency_ms", stage="decode").rows[0]
    assert unavailable["measurement_count"] == 1
    assert unavailable["value_count"] == 0
    assert unavailable["mean_value"] is None
    assert unavailable["min_value"] is None


def test_token_usage_preserves_unavailable_values_as_null(database):
    report = token_usage_summary(database, provider="provider-x")

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row["usage_records"] == 2
    assert row["input_token_records"] == 1
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 25
    assert row["cached_input_tokens"] == 10
    assert row["estimated_cost_usd"] == pytest.approx(0.01)
    null_only = token_usage_summary(database, provider="provider-z").rows[0]
    assert null_only["input_token_records"] == 0
    assert null_only["input_tokens"] is None
    assert null_only["output_tokens"] is None
    assert null_only["cached_input_tokens"] is None
    assert null_only["estimated_cost_usd"] is None


def test_parameter_values_cannot_inject_sql(database):
    malicious = "linux' OR 1=1 --"
    report = run_outcomes(database, platform=malicious)
    assert report.rows == ()
    assert report.metadata["coverage"]["source_rows"] == 0
    assert run_outcomes(database, status="failed' UNION SELECT 1 --").rows == ()
    assert measurement_summary(database, metric="latency_ms", dimensions={"backend": "cpu' OR true --"}).rows == ()


def test_limit_is_capped_and_reports_truncation(tmp_path):
    path = initialize_database(tmp_path / "many-runs.duckdb")
    connection = duckdb.connect(str(path))
    connection.executemany(
        "INSERT INTO pipeline_runs (run_id, commit_sha, platform, status) VALUES (?, ?, ?, ?)",
        [(f"run-{index:04}", f"commit-{index:04}", "linux", "passed") for index in range(MAX_REPORT_ROWS + 7)],
    )
    connection.close()

    report = run_outcomes(path, limit=MAX_REPORT_ROWS + 500)
    assert len(report.rows) == MAX_REPORT_ROWS
    coverage = report.metadata["coverage"]
    assert coverage["source_rows"] == MAX_REPORT_ROWS + 7
    assert coverage["report_rows"] == MAX_REPORT_ROWS + 7
    assert coverage["truncated"] is True
    assert coverage["applied_limit"] == MAX_REPORT_ROWS
    assert coverage["requested_limit"] == MAX_REPORT_ROWS + 500


def test_missing_data_returns_empty_report_and_validates_filters(database):
    empty = measurement_summary(database, metric="unknown_metric")
    assert empty.rows == ()
    assert empty.metadata["coverage"]["source_rows"] == 0
    assert empty.metadata["provenance"]["date_start"] is None
    with pytest.raises(TypeError, match="start_date"):
        run_outcomes(database, start_date="2026-11-20")
    with pytest.raises(ValueError, match="start_date"):
        run_outcomes(database, start_date=date(2026, 11, 22), end_date=date(2026, 11, 20))
    with pytest.raises(ValueError, match="dimension keys"):
        measurement_summary(database, dimensions={"backend'; DROP TABLE pipeline_runs; --": "cpu"})
    with pytest.raises(ValueError, match="limit"):
        token_usage_summary(database, limit=0)


def test_queries_open_read_only_and_do_not_mutate_database(database, monkeypatch):
    original_connect = duckdb.connect
    access_modes = []

    class DuckDBFacade:
        @staticmethod
        def connect(path, *, read_only=False):
            access_modes.append(read_only)
            return original_connect(path, read_only=read_only)

    monkeypatch.setattr(query_pack, "_duckdb", lambda: DuckDBFacade)
    before = run_outcomes(database).metadata["coverage"]["source_rows"]
    assert access_modes == [True]
    with pytest.raises(duckdb.InvalidInputException):
        connection = duckdb.connect(str(database), read_only=True)
        try:
            connection.execute(
                "INSERT INTO pipeline_runs (run_id, status) VALUES ('forbidden', 'passed')"
            )
        finally:
            connection.close()
    after = run_outcomes(database).metadata["coverage"]["source_rows"]
    assert before == after == 3


def _insert_pareto_fixture(database):
    connection = duckdb.connect(str(database))
    candidates = [
        ("fast-low", 0.80, 90.0, date(2026, 11, 20)),
        ("accurate-slow", 0.95, 150.0, date(2026, 11, 21)),
        ("dominated", 0.70, 180.0, date(2026, 11, 22)),
    ]
    connection.executemany(
        "INSERT INTO pipeline_runs (run_id, run_date, commit_sha, platform, command, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (f"pareto-{candidate}", day, f"sha-{candidate}", "linux", "benchmark", "passed")
            for candidate, _, _, day in candidates
        ],
    )
    measurements = []
    for candidate, quality, latency, day in candidates:
        measured_at = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
        dimensions = f'{{"workload_id":"workload-a","backend":"{candidate}"}}'
        for metric, stage, value, unit in (
            ("answer_quality", "evaluate", quality, "score"),
            ("latency_ms", "benchmark", latency, "ms"),
        ):
            measurements.append(
                (
                    f"{candidate}-{metric}", f"pareto-{candidate}", stage,
                    metric, value, unit, measured_at, dimensions,
                )
            )
    connection.executemany(
        "INSERT INTO measurements (measurement_id, run_id, stage, metric, value, unit, measured_at, dimensions) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        measurements,
    )
    connection.close()


def test_quality_latency_pareto_query_is_filtered_bounded_and_read_only(database, monkeypatch):
    _insert_pareto_fixture(database)
    original_connect = duckdb.connect
    access_modes = []

    class DuckDBFacade:
        @staticmethod
        def connect(path, *, read_only=False):
            access_modes.append(read_only)
            return original_connect(path, read_only=read_only)

    monkeypatch.setattr(query_pack, "_duckdb", lambda: DuckDBFacade)
    connection = original_connect(str(database), read_only=True)
    before = connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    connection.close()

    report = quality_latency_pareto_report(
        database,
        quality_metric="answer_quality",
        latency_metric="latency_ms",
        cohort_dimensions=["workload_id"],
        candidate_dimensions=["backend"],
        quality_stage="evaluate",
        latency_stage="benchmark",
        start_date=date(2026, 11, 20),
        end_date=date(2026, 11, 21),
    )

    assert access_modes == [True]
    assert [row["candidate"]["backend"] for row in report.rows] == [
        "accurate-slow", "fast-low"
    ]
    by_candidate = {row["candidate"]["backend"]: row for row in report.rows}
    assert by_candidate["fast-low"]["quality_value"] == 0.8
    assert by_candidate["fast-low"]["latency_value"] == 90.0
    assert all(row["pareto_efficient"] for row in report.rows)
    assert report.metadata["filters"]["quality_stage"] == "evaluate"
    assert report.metadata["filters"]["start_date"] == "2026-11-20"
    assert report.metadata["filters"]["end_date"] == "2026-11-21"
    assert report.metadata["coverage"]["quality_records_after_filters"] == 2
    assert report.metadata["coverage"]["latency_records_after_filters"] == 2

    connection = original_connect(str(database), read_only=True)
    after = connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    connection.close()
    assert before == after == 9


def test_quality_latency_pareto_query_fails_instead_of_truncating_input(
    database, monkeypatch
):
    _insert_pareto_fixture(database)
    monkeypatch.setattr(query_pack, "MAX_INPUT_OBSERVATIONS", 5)

    with pytest.raises(ValueError, match="input limit"):
        quality_latency_pareto_report(
            database,
            quality_metric="answer_quality",
            latency_metric="latency_ms",
            cohort_dimensions=["workload_id"],
            candidate_dimensions=["backend"],
            quality_stage="evaluate",
            latency_stage="benchmark",
        )


def test_quality_latency_pareto_binds_stage_values_in_fixed_sql(database, monkeypatch):
    _insert_pareto_fixture(database)
    original_connect = duckdb.connect
    executions = []

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, params=None):
            executions.append((sql, params))
            return self.connection.execute(sql, params)

        def close(self):
            self.connection.close()

    class DuckDBFacade:
        @staticmethod
        def connect(path, *, read_only=False):
            assert read_only is True
            return ConnectionProxy(original_connect(path, read_only=read_only))

    monkeypatch.setattr(query_pack, "_duckdb", lambda: DuckDBFacade)
    injected_stage = "evaluate' OR 1=1 --"

    report = quality_latency_pareto_report(
        database,
        quality_metric="answer_quality",
        latency_metric="latency_ms",
        cohort_dimensions=["workload_id"],
        candidate_dimensions=["backend"],
        quality_stage=injected_stage,
        latency_stage="benchmark",
    )

    assert report.rows == ()
    assert len(executions) == 1
    sql, params = executions[0]
    assert injected_stage not in sql
    assert injected_stage in params


@pytest.mark.parametrize(
    ("quality_metric", "latency_metric", "error"),
    [
        ("quality metric", "latency_ms", "quality_metric must be a short safe metric label"),
        ("answer_quality", "latency; DROP TABLE measurements", "latency_metric must be a short safe metric label"),
        ("same_metric", "same_metric", "quality_metric and latency_metric must differ"),
    ],
)
def test_quality_latency_pareto_validates_metric_labels_before_connect(
    database, monkeypatch, quality_metric, latency_metric, error
):
    def forbidden_connect(*args, **kwargs):
        pytest.fail("DuckDB must not be opened for invalid Pareto metric labels")

    monkeypatch.setattr(query_pack, "_duckdb", lambda: type("DuckDB", (), {"connect": forbidden_connect}))

    with pytest.raises(ValueError, match=error):
        quality_latency_pareto_report(
            database,
            quality_metric=quality_metric,
            latency_metric=latency_metric,
            cohort_dimensions=["workload_id"],
            candidate_dimensions=["backend"],
        )
