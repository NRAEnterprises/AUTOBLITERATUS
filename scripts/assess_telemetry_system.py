"""Assess practical telemetry-system paths using the actual project ledger.

No ledger rows are fabricated. Read concurrency uses the imported five-row
source; writer concurrency is deliberately not attempted because DuckDB's
embedded file has a single-writer-process policy. All generated databases and
restore copies live in a temporary directory and are deleted at exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Event
from typing import Any

from autobliteratus.observability.telemetry_store import (
    _duckdb,
    import_results_csv,
    redact_text,
    summarize_database,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile for an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    offset = (len(ordered) - 1) * quantile
    low = int(offset)
    high = min(low + 1, len(ordered) - 1)
    weight = offset - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _table_counts(connection: Any) -> dict[str, int]:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
        ).fetchall()
    ]
    return {
        name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in names
    }


def _read_concurrency(database_path: Path, readers: int, queries_each: int, expected: tuple[int, int]) -> dict[str, Any]:
    barrier = Barrier(readers + 1)
    start_event = Event()

    def worker() -> list[float]:
        connection = _duckdb().connect(str(database_path), read_only=True)
        connection.execute("SET threads = 1")
        latencies: list[float] = []
        try:
            barrier.wait(timeout=20)
            start_event.wait(timeout=20)
            for _ in range(queries_each):
                started = time.perf_counter_ns()
                actual = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(s.total), 0) "
                    "FROM pipeline_runs r JOIN test_summaries s USING (run_id)"
                ).fetchone()
                latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                if tuple(actual) != expected:
                    raise AssertionError(f"Reader parity failure: {actual!r} != {expected!r}")
        finally:
            connection.close()
        return latencies

    with ThreadPoolExecutor(max_workers=readers) as pool:
        futures = [pool.submit(worker) for _ in range(readers)]
        barrier.wait(timeout=20)
        started = time.perf_counter_ns()
        start_event.set()
        batches = [future.result() for future in futures]
    elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
    latencies = [latency for batch in batches for latency in batch]
    total_queries = len(latencies)
    return {
        "readers": readers,
        "queries_per_reader": queries_each,
        "queries_completed": total_queries,
        "errors": 0,
        "elapsed_s": elapsed_s,
        "aggregate_queries_per_second": total_queries / elapsed_s,
        "per_query_median_ms": statistics.median(latencies),
        "per_query_p95_ms": _percentile(latencies, 0.95),
        "results_exactly_match": True,
        "connections": "one independent read-only connection per reader thread",
        "database_threads_per_connection": 1,
    }


def assess(
    csv_path: Path,
    repeated_imports: int,
    queries_each_reader: int,
    reader_rounds: int,
) -> dict[str, Any]:
    source_bytes = csv_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_db = _duckdb()
    del source_db  # Fail early with the optional-dependency message if DuckDB is absent.

    with tempfile.TemporaryDirectory(prefix="autobliteratus-system-assessment-") as temp_name:
        directory = Path(temp_name)
        database = directory / "assessment.duckdb"

        import_started = time.perf_counter_ns()
        first_import = import_results_csv(csv_path, database)
        import_ms = (time.perf_counter_ns() - import_started) / 1_000_000
        if first_import.source_sha256 != source_hash:
            raise AssertionError("Importer source digest differs from the source file digest")
        if first_import.rows_inserted != first_import.rows_seen:
            raise AssertionError("Fresh-database import did not insert every source row")
        expected_summary = summarize_database(database)

        repeated_ms: list[float] = []
        repeated_inserted = 0
        for _ in range(repeated_imports):
            started = time.perf_counter_ns()
            result = import_results_csv(csv_path, database)
            repeated_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            repeated_inserted += result.rows_inserted
            if result.rows_inserted != 0 or result.source_sha256 != source_hash:
                raise AssertionError("Repeated import was not idempotent")
        if summarize_database(database) != expected_summary:
            raise AssertionError("Repeated imports changed the database summary")

        connection = _duckdb().connect(str(database), read_only=True)
        try:
            schema_version = connection.execute(
                "SELECT MAX(schema_version) FROM telemetry_schema"
            ).fetchone()[0]
            table_counts = _table_counts(connection)
            required_tables = {
                "telemetry_schema",
                "pipeline_runs",
                "test_summaries",
                "test_results",
                "pipeline_events",
                "measurements",
                "llm_usage",
                "artifact_refs",
                "query_benchmarks",
            }
            missing_tables = sorted(required_tables - set(table_counts))
            if missing_tables:
                raise AssertionError(f"Missing schema tables: {missing_tables}")
            summary_rows = connection.execute(
                "SELECT run_id, run_date, commit_sha, platform, status, passed, failed, skipped, total "
                "FROM v_run_outcomes ORDER BY run_id"
            ).fetchall()
            view_hash = _fingerprint(summary_rows)
            expected_read = tuple(
                connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(s.total), 0) "
                    "FROM pipeline_runs r JOIN test_summaries s USING (run_id)"
                ).fetchone()
            )
        finally:
            connection.close()

        read_concurrency = [
            {"round": round_index + 1, **_read_concurrency(database, readers, queries_each_reader, expected_read)}
            for readers in (1, 2, 4, 8)
            for round_index in range(reader_rounds)
        ]

        db_bytes = database.stat().st_size
        backup = directory / "closed-file-copy.duckdb"
        shutil.copy2(database, backup)
        restore_connection = _duckdb().connect(str(backup), read_only=True)
        try:
            restore_summary = summarize_database(backup)
            restore_rows = restore_connection.execute(
                "SELECT run_id, run_date, commit_sha, platform, status, passed, failed, skipped, total "
                "FROM v_run_outcomes ORDER BY run_id"
            ).fetchall()
        finally:
            restore_connection.close()
        restore_ok = restore_summary == expected_summary and _fingerprint(restore_rows) == view_hash
        if not restore_ok:
            raise AssertionError("Closed-file backup/restore parity check failed")

    privacy_cases = {
        "api_key_assignment": "api_key=BENCHMARK_FAKE_SECRET",
        "bare_bearer": "Bearer BENCHMARK_FAKE_SECRET",
        "authorization_bearer_header": "Authorization: Bearer BENCHMARK_FAKE_SECRET",
        "bare_token_assignment": "token=BENCHMARK_FAKE_SECRET",
        "password_assignment": "password=BENCHMARK_FAKE_SECRET",
        "ghp_like_prefix": "ghp_BENCHMARK_FAKE_SECRET",
    }
    privacy_probe = {}
    for name, sample in privacy_cases.items():
        redacted = redact_text(sample)
        privacy_probe[name] = {
            "secret_marker_removed": "BENCHMARK_FAKE_SECRET" not in redacted,
            "stored_sample": False,
        }

    return {
        "assessment": "autobliteratus_telemetry_system_assessment",
        "status": "completed_on_actual_ledger_with_replay_free_read_stress",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(csv_path),
            "rows": first_import.rows_seen,
            "bytes": len(source_bytes),
            "sha256": source_hash,
            "synthetic_rows": 0,
            "expected_summary": expected_summary,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "duckdb": _duckdb().__version__,
        },
        "functional_checks": {
            "first_import_inserted_all_rows": True,
            "repeated_imports": repeated_imports,
            "rows_inserted_by_repeated_imports": repeated_inserted,
            "idempotent": repeated_inserted == 0,
            "schema_version": schema_version,
            "missing_required_tables": missing_tables,
            "table_row_counts": table_counts,
            "actual_ledger_summary": expected_summary,
            "view_result_sha256": view_hash,
            "closed_file_copy_restore_parity": restore_ok,
            "database_file_bytes": db_bytes,
        },
        "ingestion": {
            "initial_import_ms": import_ms,
            "initial_import_rows_per_second": first_import.rows_seen / (import_ms / 1000),
            "idempotent_reimport_median_ms": statistics.median(repeated_ms),
            "idempotent_reimport_p95_ms": _percentile(repeated_ms, 0.95),
            "idempotent_reimport_runs": repeated_imports,
            "note": "Throughput uses only five real rows and includes DuckDB schema initialization plus CSV parsing; treat as a local smoke figure, not a capacity estimate.",
        },
        "read_concurrency": {
            "tests": read_concurrency,
            "rounds_per_reader_level": reader_rounds,
            "writer_concurrency": "not exercised; design requires one writer process and the pytest plugin does not implement worker spooling",
            "data_scale_caveat": "Every reader queried the same five-row ledger; aggregate qps is only a small-file read smoke test.",
        },
        "privacy_redaction_probes": {
            "cases": privacy_probe,
            "note": "Only boolean outcomes are persisted in this assessment; the seeded strings are synthetic redaction test markers, not telemetry evidence.",
        },
        "scope_not_implemented_or_not_measured": [
            "pytest-xdist/process-safe spool-and-single-collector workflow",
            "stage-event ingestion and log redaction at pipeline scale",
            "automatic schema migrations and downgrade/upgrade tests",
            "atomic online backup, scheduled backup, and restore automation",
            "Parquet export/rebuild path",
            "authoritative LLM provider token accounting and token-cost benchmark",
            "representative-volume retention, compaction, and resource use",
            "PostgreSQL/ClickHouse comparison",
            "crash injection during plugin session-finish batch write",
        ],
        "interpretation": "All positive checks apply only to the inspected five-row ledger and implemented paths. The read stress deliberately reuses real rows and does not create a larger synthetic dataset. No generalized database superiority or production capacity claim is supported.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("testing/results.csv"))
    parser.add_argument("--repeated-imports", type=int, default=20)
    parser.add_argument("--queries-each-reader", type=int, default=250)
    parser.add_argument("--reader-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeated_imports < 1 or args.queries_each_reader < 1 or args.reader_rounds < 1:
        parser.error("repeat counts must be positive")
    result = assess(args.csv, args.repeated_imports, args.queries_each_reader, args.reader_rounds)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote telemetry assessment: {args.output}", file=sys.stderr)
    else:
        print(output, end="")
    for item in result["read_concurrency"]["tests"]:
        print(
            f"readers={item['readers']} qps={item['aggregate_queries_per_second']:.2f} "
            f"p50_ms={item['per_query_median_ms']:.4f} p95_ms={item['per_query_p95_ms']:.4f} "
            f"errors={item['errors']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
