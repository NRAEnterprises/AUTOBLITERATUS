"""Opt-in pytest result capture for the local telemetry warehouse.

Usage::

    PYTHONPATH=src pytest -p autobliteratus.observability.pytest_plugin \
        --autobliteratus-db testing/autobliteratus_telemetry.duckdb

Test names and failure text are never stored. Stable hashes support aggregation
without leaking parameter values or exception content.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from autobliteratus.observability.telemetry_store import (
    _duckdb,
    initialize_database,
    redact_text,
)

_ACTIVE_CONFIG: Any | None = None


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("autobliteratus telemetry")
    group.addoption(
        "--autobliteratus-db",
        action="store",
        default=None,
        metavar="PATH",
        help="opt in to redacted per-test DuckDB telemetry at PATH",
    )


def pytest_configure(config: Any) -> None:
    global _ACTIVE_CONFIG
    db_path = config.getoption("--autobliteratus-db")
    if not db_path:
        return
    _ACTIVE_CONFIG = config
    path = initialize_database(db_path)
    config._autobliteratus_telemetry = {
        "db_path": str(path),
        "run_id": str(uuid.uuid4()),
        "results": {},
        "started_at": datetime.now(timezone.utc),
        "source": None,
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    config._autobliteratus_telemetry["source"] = revision


def pytest_runtest_logreport(report: Any) -> None:
    config = _ACTIVE_CONFIG
    if config is None:
        return
    telemetry = getattr(config, "_autobliteratus_telemetry", None)
    if telemetry is None:
        return

    results = telemetry["results"]
    record = results.setdefault(
        report.nodeid,
        {"phases": {}, "duration_ms": 0.0, "failure_fingerprints": []},
    )
    phase = report.when
    was_xfail = getattr(report, "wasxfail", None)
    outcome = ("xpassed" if report.outcome == "passed" else "xfail") if was_xfail else report.outcome
    record["phases"][phase] = outcome
    record["duration_ms"] += float(getattr(report, "duration", 0.0)) * 1000.0
    if report.failed:
        redacted_repr = redact_text(str(report.longrepr))
        fingerprint = hashlib.sha256(redacted_repr.encode("utf-8")).hexdigest()
        record["failure_fingerprints"].append(fingerprint)


def _outcome(record: dict[str, Any]) -> str:
    outcomes = set(record["phases"].values())
    if "failed" in outcomes:
        return "failed"
    if "xpassed" in outcomes:
        return "xpassed"
    if "xfail" in outcomes:
        return "xfail"
    if "skipped" in outcomes:
        return "skipped"
    if "passed" in outcomes:
        return "passed"
    return "incomplete"


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    global _ACTIVE_CONFIG
    config = session.config
    telemetry = getattr(config, "_autobliteratus_telemetry", None)
    if telemetry is None:
        return

    finished_at = datetime.now(timezone.utc)
    duration_ms = (finished_at - telemetry["started_at"]).total_seconds() * 1000.0
    revision = telemetry["source"]
    run_id = telemetry["run_id"]
    command = "pytest (opt-in telemetry; raw arguments omitted)"
    run_status = "passed" if exitstatus == 0 else "failed"
    duckdb = _duckdb()
    connection = duckdb.connect(telemetry["db_path"])
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """INSERT INTO pipeline_runs
               (run_id, run_date, commit_sha, platform, command, status, source_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [run_id, finished_at.date(), revision, platform.platform(), command, run_status, "pytest_plugin"],
        )
        counts = {"passed": 0, "failed": 0, "skipped": 0, "xfail": 0, "xpassed": 0, "incomplete": 0}
        for nodeid, record in telemetry["results"].items():
            outcome = _outcome(record)
            counts[outcome] = counts.get(outcome, 0) + 1
            test_id = hashlib.sha256(nodeid.encode("utf-8", errors="replace")).hexdigest()
            failures = record["failure_fingerprints"]
            failure_fingerprint = failures[0] if failures else None
            metadata = json.dumps({"phases": record["phases"]}, sort_keys=True)
            connection.execute(
                """INSERT INTO test_results
                   (run_id, test_id, attempt, outcome, duration_ms,
                    failure_type, failure_fingerprint, metadata)
                   VALUES (?, ?, 1, ?, ?, ?, ?, ?)""",
                [
                    run_id,
                    test_id,
                    outcome,
                    record["duration_ms"],
                    "pytest_failure" if failures else None,
                    failure_fingerprint,
                    metadata,
                ],
            )
        passed = counts["passed"]
        failed = counts["failed"] + counts["xpassed"]
        skipped = counts["skipped"] + counts["xfail"]
        total = sum(counts.values())
        connection.execute(
            """INSERT INTO test_summaries (run_id, passed, failed, skipped, total)
               VALUES (?, ?, ?, ?, ?)""",
            [run_id, passed, failed, skipped, total],
        )
        connection.execute(
            """INSERT INTO measurements
               (measurement_id, run_id, stage, metric, value, unit, measured_at, dimensions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                str(uuid.uuid4()),
                run_id,
                "pytest",
                "suite_duration",
                duration_ms,
                "milliseconds",
                finished_at,
                json.dumps({"exit_status": int(exitstatus), "python": sys.version.split()[0]}),
            ],
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
        _ACTIVE_CONFIG = None
