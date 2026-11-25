"""Opt-in pytest telemetry using worker-local JSONL spools.

Usage::

    PYTHONPATH=src pytest -p autobliteratus.observability.pytest_plugin \
        --autobliteratus-db testing/autobliteratus_telemetry.duckdb

Test names and failure text are never persisted. Each pytest process appends
sanitized events to its own spool; only the controller (or a later recovery run)
opens DuckDB. With distributed workers, the spool directory must be on a
filesystem shared by the controller and workers.
"""

from __future__ import annotations

import hashlib
import math
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from autobliteratus.observability.telemetry_spool import (
    ActiveSpoolError,
    acquire_run_lease,
    append_spool_event,
    cleanup_spool_run,
    collect_pending_spools,
    ensure_private_spool_directory,
    make_spool_event,
    release_run_lease,
    spool_root_for,
)
from autobliteratus.observability.telemetry_store import redact_text

_ACTIVE_CONFIG: Any | None = None
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("autobliteratus telemetry")
    group.addoption(
        "--autobliteratus-db",
        action="store",
        default=None,
        metavar="PATH",
        help="opt in to redacted pytest JSONL spooling and parent DuckDB collection",
    )


def _source_revision() -> str | None:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return revision if _COMMIT_RE.fullmatch(revision) else None


def _spool_path(database_path: str, run_id: str) -> Path:
    return spool_root_for(database_path) / run_id


def pytest_configure(config: Any) -> None:
    global _ACTIVE_CONFIG
    raw_database_path = config.getoption("--autobliteratus-db")
    if not raw_database_path:
        return

    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        shared = workerinput.get("autobliteratus_telemetry")
        if not isinstance(shared, dict):
            raise pytest.UsageError(
                "Telemetry xdist worker did not receive the controller spool configuration"
            )
        run_id = shared["run_id"]
        worker_id = str(workerinput.get("workerid", "worker"))
        try:
            lease_fd = acquire_run_lease(shared["spool_dir"], worker_id=worker_id)
        except ActiveSpoolError as exc:
            raise pytest.UsageError(
                "Telemetry xdist worker cannot join a run without its active controller"
            ) from exc
        state = {
            "db_path": shared["db_path"],
            "run_id": run_id,
            "spool_dir": shared["spool_dir"],
            "worker_id": worker_id,
            "role": "worker",
            "distributed": True,
            "lease_fd": lease_fd,
        }
    else:
        database_path = str(Path(raw_database_path).expanduser().resolve())
        spool_root = spool_root_for(database_path)
        # Recover runs left behind by a killed controller before starting a new
        # one. These calls are parent-only and internally acquire the DB lock.
        for recovered in collect_pending_spools(database_path):
            cleanup_spool_run(spool_root / recovered.run_id, database_path)

        run_id = str(uuid.uuid4())
        spool_dir = _spool_path(database_path, run_id)
        ensure_private_spool_directory(spool_dir, exist_ok=False)
        lease_fd = acquire_run_lease(spool_dir)
        state = {
            "db_path": database_path,
            "run_id": run_id,
            "spool_dir": str(spool_dir),
            "worker_id": "controller",
            "role": "controller",
            "distributed": False,
            "lease_fd": lease_fd,
        }
        started_at = datetime.now(timezone.utc).isoformat()
        append_spool_event(
            spool_dir,
            make_spool_event(
                run_id,
                "controller",
                "run_start",
                started_at=started_at,
                source_revision=_source_revision(),
                platform=redact_text(platform.platform())[:256],
                python=sys.version.split()[0][:64],
            ),
        )

    config._autobliteratus_telemetry = state
    _ACTIVE_CONFIG = config


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """Pass one run identity and shared spool location from xdist parent."""
    telemetry = getattr(node.config, "_autobliteratus_telemetry", None)
    if telemetry is None:
        return
    telemetry["distributed"] = True
    node.workerinput["autobliteratus_telemetry"] = {
        "db_path": telemetry["db_path"],
        "run_id": telemetry["run_id"],
        "spool_dir": telemetry["spool_dir"],
    }


def _test_id(nodeid: Any) -> str:
    return hashlib.sha256(str(nodeid).encode("utf-8", errors="replace")).hexdigest()


def _failure_fingerprint(report: Any) -> str | None:
    if not getattr(report, "failed", False):
        return None
    safe_text = redact_text(str(getattr(report, "longrepr", "")))
    return hashlib.sha256(safe_text.encode("utf-8", errors="replace")).hexdigest()


def _attempt(report: Any) -> int:
    rerun = getattr(report, "rerun", None)
    try:
        if rerun is not None:
            return max(1, int(rerun) + 1)
        attempt = int(getattr(report, "attempt", 1))
        return max(1, attempt)
    except (TypeError, ValueError):
        return 1


def _duration_ms(report: Any) -> float:
    try:
        value = float(getattr(report, "duration", 0.0)) * 1000.0
    except (TypeError, ValueError):
        return 0.0
    return min(1e9, max(0.0, value)) if math.isfinite(value) else 0.0


def _record_report(config: Any, report: Any) -> None:
    telemetry = getattr(config, "_autobliteratus_telemetry", None)
    if telemetry is None:
        return
    # In distributed mode, reports received by the controller duplicate those
    # already spooled by workers. Collection reports are handled separately and
    # have run-wide dedupe IDs.
    if telemetry["role"] == "controller" and telemetry["distributed"]:
        return

    phase = str(getattr(report, "when", ""))
    if phase not in {"setup", "call", "teardown"}:
        return
    report_outcome = str(getattr(report, "outcome", "failed"))
    was_xfail = getattr(report, "wasxfail", None)
    outcome = ("xpassed" if report_outcome == "passed" else "xfail") if was_xfail else report_outcome
    if outcome not in {"passed", "failed", "skipped", "xfail", "xpassed"}:
        outcome = "failed"
    test_id = _test_id(getattr(report, "nodeid", ""))
    worker_id = telemetry["worker_id"]
    event = make_spool_event(
        telemetry["run_id"],
        worker_id,
        "test_phase",
        test_id=test_id,
        attempt=_attempt(report),
        phase=phase,
        outcome=outcome,
        duration_ms=_duration_ms(report),
        failure_fingerprint=_failure_fingerprint(report),
        complete=phase == "teardown",
    )
    append_spool_event(telemetry["spool_dir"], event)


def pytest_runtest_logreport(report: Any) -> None:
    if _ACTIVE_CONFIG is not None:
        _record_report(_ACTIVE_CONFIG, report)


def pytest_collectreport(report: Any) -> None:
    config = _ACTIVE_CONFIG
    telemetry = getattr(config, "_autobliteratus_telemetry", None) if config else None
    if telemetry is None or getattr(report, "outcome", None) not in {"failed", "skipped"}:
        return
    event = make_spool_event(
        telemetry["run_id"],
        telemetry["worker_id"],
        "collection",
        test_id=_test_id(getattr(report, "nodeid", "")),
        outcome=str(report.outcome),
        failure_fingerprint=_failure_fingerprint(report),
    )
    append_spool_event(telemetry["spool_dir"], event)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    global _ACTIVE_CONFIG
    config = session.config
    telemetry = getattr(config, "_autobliteratus_telemetry", None)
    if telemetry is None:
        return
    finished_at = datetime.now(timezone.utc).isoformat()
    if telemetry["role"] == "worker":
        event = make_spool_event(
            telemetry["run_id"],
            telemetry["worker_id"],
            "worker_finish",
            finished_at=finished_at,
            exit_status=int(exitstatus),
        )
        append_spool_event(telemetry["spool_dir"], event)
        return

    run_status = "passed" if int(exitstatus) == 0 else "failed"
    event = make_spool_event(
        telemetry["run_id"],
        telemetry["worker_id"],
        "run_finish",
        finished_at=finished_at,
        exit_status=int(exitstatus),
        run_status=run_status,
    )
    append_spool_event(telemetry["spool_dir"], event)
    lease_fd = telemetry.get("lease_fd")
    if lease_fd is not None:
        release_run_lease(lease_fd)
        telemetry["lease_fd"] = None
    spool_root = spool_root_for(telemetry["db_path"])
    collected = collect_pending_spools(telemetry["db_path"])
    for result in collected:
        cleanup_spool_run(spool_root / result.run_id, telemetry["db_path"])
    _ACTIVE_CONFIG = None


def pytest_unconfigure(config: Any) -> None:
    global _ACTIVE_CONFIG
    telemetry = getattr(config, "_autobliteratus_telemetry", None)
    if telemetry is not None and telemetry.get("lease_fd") is not None:
        release_run_lease(telemetry["lease_fd"])
        telemetry["lease_fd"] = None
    if _ACTIVE_CONFIG is config:
        _ACTIVE_CONFIG = None
