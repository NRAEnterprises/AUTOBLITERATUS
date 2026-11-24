"""Parity-first benchmark for CSV/Python, SQLite, and DuckDB telemetry queries.

The benchmark uses the real CSV ledger without synthesizing rows. It reports
first-touch latency (not a guaranteed cold-cache measurement) and repeated
steady-state query latency. SQLite and DuckDB database build/import time is
reported separately from query latency. Timings are descriptive and do not
qualify an engine for production or imply scaling behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from autobliteratus.observability.telemetry_store import import_results_csv


def _read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows: list[dict[str, Any]] = []
        for source in reader:
            rows.append(
                {
                    "run_date": date.fromisoformat(source["date"]).isoformat(),
                    "commit_sha": source["commit"],
                    "platform": source["platform"],
                    "passed": int(source["passed"]),
                    "failed": int(source["failed"]),
                    "skipped": int(source["skipped"]),
                }
            )
        return rows


def _canonical_hash(rows: list[tuple[Any, ...]]) -> str:
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _python_queries(csv_path: Path) -> dict[str, Callable[[], list[tuple[Any, ...]]]]:
    def summary() -> list[tuple[Any, ...]]:
        rows = _read_rows(csv_path)
        return [
            (
                len(rows),
                sum(row["passed"] for row in rows),
                sum(row["failed"] for row in rows),
                sum(row["skipped"] for row in rows),
                sum(row["passed"] + row["failed"] + row["skipped"] for row in rows),
            )
        ]

    def by_platform() -> list[tuple[Any, ...]]:
        groups: dict[str, list[int]] = {}
        for row in _read_rows(csv_path):
            totals = groups.setdefault(row["platform"], [0, 0, 0, 0])
            totals[0] += 1
            totals[1] += row["passed"]
            totals[2] += row["failed"]
            totals[3] += row["skipped"]
        return [
            (name, values[0], values[1], values[2], values[3])
            for name, values in sorted(groups.items())
        ]

    def selected_runs() -> list[tuple[Any, ...]]:
        rows = [row for row in _read_rows(csv_path) if row["passed"] >= 35]
        return [
            (
                row["run_date"],
                row["commit_sha"],
                row["platform"],
                row["passed"],
                row["failed"],
                row["skipped"],
            )
            for row in sorted(
                rows,
                key=lambda item: (
                    item["run_date"],
                    item["commit_sha"],
                    item["platform"],
                    item["passed"],
                ),
            )
        ]

    return {
        "q1_summary_totals": summary,
        "q2_group_by_platform": by_platform,
        "q3_filtered_run_detail_passed_gte_35": selected_runs,
    }


def _sqlite_queries(connection: sqlite3.Connection) -> dict[str, Callable[[], list[tuple[Any, ...]]]]:
    def run(sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in connection.execute(sql, params).fetchall()]

    return {
        "q1_summary_totals": lambda: run(
            "SELECT COUNT(*), SUM(s.passed), SUM(s.failed), SUM(s.skipped), "
            "SUM(s.passed + s.failed + s.skipped) "
            "FROM pipeline_runs r JOIN test_summaries s USING (run_id)"
        ),
        "q2_group_by_platform": lambda: run(
            "SELECT r.platform, COUNT(*), SUM(s.passed), SUM(s.failed), SUM(s.skipped) "
            "FROM pipeline_runs r JOIN test_summaries s USING (run_id) "
            "GROUP BY r.platform ORDER BY r.platform"
        ),
        "q3_filtered_run_detail_passed_gte_35": lambda: run(
            "SELECT r.run_date, r.commit_sha, r.platform, s.passed, s.failed, s.skipped "
            "FROM pipeline_runs r JOIN test_summaries s USING (run_id) "
            "WHERE s.passed >= ? "
            "ORDER BY r.run_date, r.commit_sha, r.platform, s.passed",
            (35,),
        ),
    }


def _duckdb_queries(connection: Any) -> dict[str, Callable[[], list[tuple[Any, ...]]]]:
    def run(sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        rows = connection.execute(sql, params).fetchall()
        return [tuple(value.isoformat() if isinstance(value, date) else value for value in row) for row in rows]

    return {
        "q1_summary_totals": lambda: run(
            "SELECT COUNT(*), SUM(s.passed), SUM(s.failed), SUM(s.skipped), "
            "SUM(s.total) FROM pipeline_runs r JOIN test_summaries s USING (run_id)"
        ),
        "q2_group_by_platform": lambda: run(
            "SELECT r.platform, COUNT(*), SUM(s.passed), SUM(s.failed), SUM(s.skipped) "
            "FROM pipeline_runs r JOIN test_summaries s USING (run_id) "
            "GROUP BY r.platform ORDER BY r.platform"
        ),
        "q3_filtered_run_detail_passed_gte_35": lambda: run(
            "SELECT r.run_date, r.commit_sha, r.platform, s.passed, s.failed, s.skipped "
            "FROM pipeline_runs r JOIN test_summaries s USING (run_id) "
            "WHERE s.passed >= ? "
            "ORDER BY r.run_date, r.commit_sha, r.platform, s.passed",
            (35,),
        ),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _measure(
    query_functions: dict[str, Callable[[], list[tuple[Any, ...]]]],
    expected_hashes: dict[str, str],
    repetitions: int,
    warmups: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for query_id, query in query_functions.items():
        first_start = time.perf_counter_ns()
        first_rows = query()
        first_ms = (time.perf_counter_ns() - first_start) / 1_000_000
        first_hash = _canonical_hash(first_rows)
        if first_hash != expected_hashes[query_id]:
            raise AssertionError(f"Parity failure on first-touch {query_id}")

        for _ in range(warmups):
            rows = query()
            if _canonical_hash(rows) != expected_hashes[query_id]:
                raise AssertionError(f"Parity failure during warmup: {query_id}")

        latencies_ms: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            rows = query()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if _canonical_hash(rows) != expected_hashes[query_id]:
                raise AssertionError(f"Parity failure during timed run: {query_id}")
            latencies_ms.append(elapsed_ms)

        results[query_id] = {
            "correct": True,
            "result_sha256": first_hash,
            "rows_returned": len(first_rows),
            "first_measured_after_parity_ms": first_ms,
            "warmup_runs": warmups,
            "timed_runs": repetitions,
            "median_ms": statistics.median(latencies_ms),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
        }
    return results


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def benchmark(csv_path: Path, repetitions: int, warmups: int) -> dict[str, Any]:
    raw_csv = csv_path.read_bytes()
    source_rows = _read_rows(csv_path)
    if not source_rows:
        raise ValueError("Refusing to benchmark an empty telemetry ledger")

    python_functions = _python_queries(csv_path)
    expected_rows = {name: query() for name, query in python_functions.items()}
    expected_hashes = {name: _canonical_hash(rows) for name, rows in expected_rows.items()}

    with tempfile.TemporaryDirectory(prefix="autobliteratus-ledger-bench-") as temp_name:
        temp_dir = Path(temp_name)
        sqlite_path = temp_dir / "ledger.sqlite3"
        sqlite_started = time.perf_counter_ns()
        sqlite_rows = _read_rows(csv_path)
        sqlite_connection = sqlite3.connect(sqlite_path)
        sqlite_connection.execute("PRAGMA foreign_keys = ON")
        sqlite_connection.execute(
            "CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, run_date TEXT NOT NULL, "
            "commit_sha TEXT, platform TEXT, status TEXT NOT NULL)"
        )
        sqlite_connection.execute(
            "CREATE TABLE test_summaries (run_id TEXT PRIMARY KEY REFERENCES pipeline_runs(run_id), "
            "passed INTEGER NOT NULL, failed INTEGER NOT NULL, skipped INTEGER NOT NULL, "
            "total INTEGER NOT NULL CHECK (total = passed + failed + skipped))"
        )
        sqlite_connection.execute("BEGIN")
        ledger_sha256 = hashlib.sha256(raw_csv).hexdigest()
        for index, row in enumerate(sqlite_rows):
            run_id = hashlib.sha256(f"{ledger_sha256}:{index}".encode("ascii")).hexdigest()
            status = (
                "failed"
                if row["failed"]
                else "passed"
                if row["passed"] and not row["skipped"]
                else "partial"
                if row["passed"] or row["skipped"]
                else "no_results"
            )
            sqlite_connection.execute(
                "INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?)",
                (run_id, row["run_date"], row["commit_sha"], row["platform"], status),
            )
            total = row["passed"] + row["failed"] + row["skipped"]
            sqlite_connection.execute(
                "INSERT INTO test_summaries VALUES (?, ?, ?, ?, ?)",
                (run_id, row["passed"], row["failed"], row["skipped"], total),
            )
        sqlite_connection.commit()
        sqlite_setup_ms = (time.perf_counter_ns() - sqlite_started) / 1_000_000

        sqlite_functions = _sqlite_queries(sqlite_connection)
        sqlite_journal_mode = sqlite_connection.execute("PRAGMA journal_mode").fetchone()[0]
        sqlite_initial_results = {name: query() for name, query in sqlite_functions.items()}
        sqlite_hashes = {name: _canonical_hash(rows) for name, rows in sqlite_initial_results.items()}
        if sqlite_hashes != expected_hashes:
            raise AssertionError(f"SQLite parity check failed: {sqlite_hashes} != {expected_hashes}")

        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Install the optional DuckDB extra to run this comparison") from exc

        duckdb_path = temp_dir / "ledger.duckdb"
        duckdb_started = time.perf_counter_ns()
        import_result = import_results_csv(csv_path, duckdb_path)
        duckdb_setup_ms = (time.perf_counter_ns() - duckdb_started) / 1_000_000
        duckdb_connection = duckdb.connect(str(duckdb_path), read_only=True)
        duckdb_connection.execute("SET threads = 1")
        duckdb_functions = _duckdb_queries(duckdb_connection)
        duckdb_initial_results = {name: query() for name, query in duckdb_functions.items()}
        duckdb_hashes = {name: _canonical_hash(rows) for name, rows in duckdb_initial_results.items()}
        if duckdb_hashes != expected_hashes:
            raise AssertionError(f"DuckDB parity check failed: {duckdb_hashes} != {expected_hashes}")

        engines: dict[str, Any] = {
            "csv_python": {
                "description": "Reopen and parse the source CSV for each query; aggregate in Python.",
                "build_or_import_ms": None,
                "persistent_bytes": len(raw_csv),
                "queries": _measure(python_functions, expected_hashes, repetitions, warmups),
            },
            "sqlite": {
            "description": "SQLite 3, single process, file-backed normalized runs/summaries tables, one reused connection.",
            "version": sqlite3.sqlite_version,
            "journal_mode": sqlite_journal_mode,
            "fts5_used": False,
                "build_or_import_ms": sqlite_setup_ms,
                "persistent_bytes": sqlite_path.stat().st_size,
                "queries": _measure(sqlite_functions, expected_hashes, repetitions, warmups),
            },
            "duckdb": {
                "description": "DuckDB file-backed database imported through the project telemetry importer; one thread, one reused read-only connection.",
                "version": duckdb.__version__,
                "import_rows_seen": import_result.rows_seen,
                "import_rows_inserted": import_result.rows_inserted,
                "build_or_import_ms": duckdb_setup_ms,
                "persistent_bytes": duckdb_path.stat().st_size,
                "queries": _measure(duckdb_functions, expected_hashes, repetitions, warmups),
            },
        }
        duckdb_connection.close()
        sqlite_connection.close()

    return {
        "benchmark": "autobliteratus_real_telemetry_ledger_parity_comparison",
        "status": "completed_parity_checked_smoke_scale",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _git_revision(),
        "source": {
            "path": str(csv_path),
            "bytes": len(raw_csv),
            "sha256": hashlib.sha256(raw_csv).hexdigest(),
            "ledger_rows": len(source_rows),
            "synthetic_rows": 0,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "duckdb": engines["duckdb"]["version"],
            "duckdb_threads": 1,
            "process_count": 1,
        },
        "method": {
            "query_cache_policy": "Every engine/query executes an untimed parity query before latency measurement, so first_measured_after_parity_ms is not a cold or first-use measurement. OS and database caches are not cleared.",
            "timed_query_runs_per_query_engine": repetitions,
            "warmups_per_query_engine": warmups,
            "connections_reused": True,
            "parity_gate": "Exact canonical result SHA-256 equality against CSV/Python for every query and every timed execution; timing is rejected if parity fails.",
            "queries": {
                "q1_summary_totals": "Count runs and sum passed/failed/skipped/total outcomes.",
                "q2_group_by_platform": "Group runs and outcome totals by platform.",
                "q3_filtered_run_detail_passed_gte_35": "Return run details with passed >= 35, sorted deterministically.",
            },
            "schema_notes": "SQLite uses the two normalized run/summary tables needed for these queries. DuckDB uses the complete versioned project schema. Storage sizes and import/setup time therefore include different schema/catalog footprints and are not same-schema efficiency comparisons.",
        },
        "parity": {
            "all_engines_match": True,
            "all_timed_executions_checked": True,
            "result_hashes": expected_hashes,
        },
        "engines": engines,
        "interpretation": "Five real ledger rows are insufficient for scaling conclusions. Timings are host- and cache-dependent and primarily characterize fixed per-query overhead at this size. No database or token-efficiency superiority claim is supported by this run.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("testing/results.csv"))
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None, help="write JSON report; stdout if omitted")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("--repetitions must be positive and --warmups non-negative")

    report = benchmark(args.csv, args.repetitions, args.warmups)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Wrote parity-checked telemetry benchmark report: {args.output}", file=sys.stderr)
    else:
        print(encoded, end="")
    for engine_name, engine in report["engines"].items():
        print(
            f"{engine_name}: build/import_ms={engine['build_or_import_ms']}, "
            f"bytes={engine['persistent_bytes']}",
            file=sys.stderr,
        )
        for query_id, result in engine["queries"].items():
            print(
                f"  {query_id}: median_ms={result['median_ms']:.6f}, "
                f"p95_ms={result['p95_ms']:.6f}, "
                f"first_measured_after_parity_ms={result['first_measured_after_parity_ms']:.6f}, "
                f"sha256={result['result_sha256']}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
