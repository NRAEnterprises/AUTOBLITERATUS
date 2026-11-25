"""Process-safe JSONL spooling and single-writer telemetry collection.

Pytest processes only append small, sanitized events to per-process spool files.
The controller (or a later recovery run) serializes DuckDB writes with an OS
file lock and ingests those events transactionally.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autobliteratus.observability.telemetry_store import (
    _duckdb,
    database_writer_lock,
    initialize_database,
    private_database_operation,
)

SPOOL_SCHEMA_VERSION = 1
_MAX_EVENT_BYTES = 16 * 1024
_MAX_SPOOL_BYTES = 64 * 1024 * 1024
_MAX_SPOOL_FILES = 256
_MAX_SPOOL_RECORDS = 100_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_WORKER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_OUTCOMES = {"passed", "failed", "skipped", "xfail", "xpassed"}
_PHASES = {"setup", "call", "teardown"}
_EVENT_FIELDS: dict[str, set[str]] = {
    "run_start": {"started_at", "source_revision", "platform", "python"},
    "run_finish": {"finished_at", "exit_status", "run_status"},
    "worker_finish": {"finished_at", "exit_status"},
    "test_phase": {
        "test_id",
        "attempt",
        "phase",
        "outcome",
        "duration_ms",
        "failure_fingerprint",
        "complete",
    },
    "collection": {"test_id", "outcome", "failure_fingerprint"},
}
_COMMON_FIELDS = {
    "schema_version",
    "event_id",
    "run_id",
    "worker_id",
    "event_type",
    "occurred_at",
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionResult:
    run_id: str
    events_seen: int
    events_inserted: int
    tests_written: int
    malformed_lines: int
    run_status: str


class ActiveSpoolError(RuntimeError):
    """Raised when a controller or worker still owns a spool directory."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def spool_root_for(database_path: str | Path) -> Path:
    path = Path(database_path).expanduser().resolve()
    return path.with_name(path.name + ".spool")


def _safe_worker_id(worker_id: str) -> str:
    if not _WORKER_RE.fullmatch(worker_id):
        raise ValueError("Invalid telemetry worker identifier")
    return worker_id


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes where POSIX directory fsync is available."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    unsupported = {errno.EINVAL, errno.ENOSYS}
    if hasattr(errno, "ENOTSUP"):
        unsupported.add(errno.ENOTSUP)
    if hasattr(errno, "EOPNOTSUPP"):
        unsupported.add(errno.EOPNOTSUPP)
    try:
        fd = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(fd)


def ensure_private_spool_directory(directory: str | Path, *, exist_ok: bool = True) -> None:
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Refusing to use a telemetry spool directory symlink")
    missing = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=exist_ok)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Telemetry spool target is not a directory")
    if os.name == "posix":
        directory.chmod(0o700)
        if stat.S_IMODE(directory.stat().st_mode) != 0o700:
            raise PermissionError("Could not restrict telemetry spool directory permissions")
    for created in reversed(missing):
        _fsync_directory(created.parent)
        _fsync_directory(created)


def _canonical_event_id(event: dict[str, Any]) -> str:
    identity = {key: value for key, value in event.items() if key not in {"event_id", "occurred_at"}}
    # Identical collection errors reported by multiple xdist workers are one
    # logical event; worker identity remains part of other event identities.
    if event.get("event_type") == "collection":
        identity["worker_id"] = "collection"
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(("autobliteratus-pytest-spool-v1\0" + canonical).encode()).hexdigest()


def make_spool_event(
    run_id: str,
    worker_id: str,
    event_type: str,
    **fields: Any,
) -> dict[str, Any]:
    """Create a versioned event and compute its replay-stable identifier."""
    try:
        canonical_run_id = str(uuid.UUID(run_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid telemetry run identifier") from exc
    if event_type not in _EVENT_FIELDS:
        raise ValueError("Unsupported telemetry event type")
    extra = _EVENT_FIELDS[event_type]
    if set(fields) != extra:
        raise ValueError("Telemetry event fields do not match the event contract")
    event = {
        "schema_version": SPOOL_SCHEMA_VERSION,
        "run_id": canonical_run_id,
        "worker_id": _safe_worker_id(worker_id),
        "event_type": event_type,
        "occurred_at": utc_now(),
        **fields,
    }
    event["event_id"] = _canonical_event_id(event)
    # Exercise the same strict contract at the producer boundary.
    validated, reason = _validate_event(event, canonical_run_id)
    if validated is None:
        raise ValueError(f"Invalid telemetry event: {reason}")
    return validated


def append_spool_event(spool_dir: str | Path, event: dict[str, Any]) -> Path:
    """Append and fsync one compact JSON record, including new directory entries."""
    directory = Path(spool_dir)
    ensure_private_spool_directory(directory)
    worker_id = _safe_worker_id(str(event.get("worker_id", "")))
    path = directory / f"{worker_id}.jsonl"
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > _MAX_EVENT_BYTES:
        raise ValueError("Telemetry event exceeds the spool record size limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_existed = path.exists()
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("Short write to telemetry spool")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if not file_existed:
        _fsync_directory(directory)
    return path


def _iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _validate_event(raw: Any, expected_run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not_object"
    if any(not isinstance(key, str) for key in raw):
        return None, "field_name"
    event_type = raw.get("event_type")
    if not isinstance(event_type, str) or event_type not in _EVENT_FIELDS:
        return None, "event_type"
    expected_fields = _COMMON_FIELDS | _EVENT_FIELDS[event_type]
    if set(raw) != expected_fields:
        return None, "fields"
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SPOOL_SCHEMA_VERSION
    ):
        return None, "schema_version"
    if raw.get("run_id") != expected_run_id:
        return None, "run_id"
    if not isinstance(raw.get("worker_id"), str) or not _WORKER_RE.fullmatch(raw["worker_id"]):
        return None, "worker_id"
    if _iso_datetime(raw.get("occurred_at")) is None:
        return None, "occurred_at"
    if not isinstance(raw.get("event_id"), str) or not _SHA256_RE.fullmatch(raw["event_id"]):
        return None, "event_id"
    if raw["event_id"] != _canonical_event_id(raw):
        return None, "event_id_mismatch"

    event = dict(raw)
    if event_type in {"run_start", "run_finish", "worker_finish"}:
        field = "started_at" if event_type == "run_start" else "finished_at"
        if _iso_datetime(event.get(field)) is None:
            return None, field
        if event_type == "run_start":
            revision = event.get("source_revision")
            if revision is not None and (not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision)):
                return None, "source_revision"
            for text_field, limit in (("platform", 256), ("python", 64)):
                value = event.get(text_field)
                if not isinstance(value, str) or len(value) > limit:
                    return None, text_field
        else:
            exit_status = event.get("exit_status")
            if isinstance(exit_status, bool) or not isinstance(exit_status, int) or not -1 <= exit_status <= 255:
                return None, "exit_status"
            if event_type == "run_finish":
                run_status = event.get("run_status")
                if not isinstance(run_status, str) or run_status not in {"passed", "failed"}:
                    return None, "run_status"
    elif event_type == "test_phase":
        if not isinstance(event.get("test_id"), str) or not _SHA256_RE.fullmatch(event["test_id"]):
            return None, "test_id"
        attempt = event.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 10000:
            return None, "attempt"
        phase = event.get("phase")
        outcome = event.get("outcome")
        if (
            not isinstance(phase, str)
            or phase not in _PHASES
            or not isinstance(outcome, str)
            or outcome not in _OUTCOMES
        ):
            return None, "phase_or_outcome"
        duration = event.get("duration_ms")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            return None, "duration_ms"
        try:
            finite_duration = math.isfinite(float(duration))
        except (OverflowError, ValueError):
            return None, "duration_ms"
        if not finite_duration or not 0 <= duration <= 1e9:
            return None, "duration_ms"
        fingerprint = event.get("failure_fingerprint")
        if fingerprint is not None and (not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint)):
            return None, "failure_fingerprint"
        if not isinstance(event.get("complete"), bool):
            return None, "complete"
    elif event_type == "collection":
        if not isinstance(event.get("test_id"), str) or not _SHA256_RE.fullmatch(event["test_id"]):
            return None, "test_id"
        outcome = event.get("outcome")
        if not isinstance(outcome, str) or outcome not in {"failed", "skipped"}:
            return None, "outcome"
        fingerprint = event.get("failure_fingerprint")
        if fingerprint is not None and (not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint)):
            return None, "failure_fingerprint"
    return event, None


def _read_events(spool_dir: Path, run_id: str) -> tuple[list[dict[str, Any]], int]:
    accepted: dict[str, dict[str, Any]] = {}
    malformed = 0
    paths = []
    for path in spool_dir.glob("*.jsonl"):
        if len(paths) >= _MAX_SPOOL_FILES:
            malformed += 1
            break
        paths.append(path)
    total_bytes = 0
    records_seen = 0
    record_limit_reached = False
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            malformed += 1
            continue
        try:
            file_bytes = path.stat().st_size
            if file_bytes > _MAX_SPOOL_BYTES - total_bytes:
                malformed += 1
                continue
            with path.open("rb") as handle:
                while True:
                    if records_seen >= _MAX_SPOOL_RECORDS:
                        try:
                            unread_records = path.stat().st_size > handle.tell()
                        except OSError:
                            unread_records = True
                        if unread_records:
                            malformed += 1
                            record_limit_reached = True
                        break
                    remaining = _MAX_SPOOL_BYTES - total_bytes
                    if remaining <= 0:
                        try:
                            unread_bytes = path.stat().st_size > handle.tell()
                        except OSError:
                            unread_bytes = True
                        if unread_bytes:
                            malformed += 1
                            record_limit_reached = True
                        break
                    line = handle.readline(min(_MAX_EVENT_BYTES + 1, remaining))
                    if not line:
                        break
                    total_bytes += len(line)
                    records_seen += 1
                    if len(line) > _MAX_EVENT_BYTES:
                        malformed += 1
                        while line and not line.endswith(b"\n"):
                            remaining = _MAX_SPOOL_BYTES - total_bytes
                            if remaining <= 0:
                                try:
                                    unread_bytes = path.stat().st_size > handle.tell()
                                except OSError:
                                    unread_bytes = True
                                if unread_bytes:
                                    malformed += 1
                                    record_limit_reached = True
                                break
                            line = handle.readline(min(_MAX_EVENT_BYTES + 1, remaining))
                            total_bytes += len(line)
                            if total_bytes > _MAX_SPOOL_BYTES:
                                malformed += 1
                                record_limit_reached = True
                                break
                        if record_limit_reached:
                            break
                        continue
                    if not line.endswith(b"\n"):
                        # A worker can be killed while writing its last line.
                        malformed += 1
                        continue
                    try:
                        decoded = json.loads(line)
                    except (ValueError, UnicodeDecodeError, RecursionError):
                        malformed += 1
                        continue
                    try:
                        event, _reason = _validate_event(decoded, run_id)
                    except (TypeError, ValueError, OverflowError, RecursionError):
                        malformed += 1
                        continue
                    if event is None:
                        malformed += 1
                        continue
                    accepted.setdefault(event["event_id"], event)
        except OSError:
            malformed += 1
        if record_limit_reached:
            break
    return list(accepted.values()), malformed


def _lock_fd(fd: int, timeout_seconds: float) -> bool:
    start = time.monotonic()
    while True:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() - start >= timeout_seconds:
                return False
            time.sleep(0.05)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _open_locked_file(path: Path, timeout_seconds: float) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    if not _lock_fd(fd, timeout_seconds):
        os.close(fd)
        raise TimeoutError("Timed out waiting for a telemetry spool lock")
    return fd


@contextlib.contextmanager
def _named_file_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    fd = _open_locked_file(path, timeout_seconds)
    try:
        yield
    finally:
        release_run_lease(fd)


def _file_lock_is_held(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return True
    try:
        if not _lock_fd(fd, 0):
            return True
        _unlock_fd(fd)
        return False
    finally:
        os.close(fd)


def _active_lease_held(directory: Path) -> bool:
    if _file_lock_is_held(directory / ".controller.lock"):
        return True
    return any(_file_lock_is_held(path) for path in directory.glob(".worker-*.lock"))


def acquire_run_lease(
    spool_dir: str | Path,
    timeout_seconds: float = 60.0,
    *,
    create: bool = True,
    worker_id: str | None = None,
) -> int:
    """Hold a controller or worker lease on a run spool."""
    directory = Path(spool_dir)
    if create:
        ensure_private_spool_directory(directory)
    elif not directory.is_dir():
        raise FileNotFoundError(directory)
    if worker_id is None:
        return _open_locked_file(directory / ".controller.lock", timeout_seconds)
    worker_path = directory / f".worker-{_safe_worker_id(worker_id)}.lock"
    # The collector gate makes checking the controller lease and publishing a
    # worker lease atomic with respect to recovery. If the controller died,
    # recovery wins the gate and this worker must not append to a collected run.
    with _named_file_lock(directory / ".collector.lock", timeout_seconds):
        if not _file_lock_is_held(directory / ".controller.lock"):
            raise ActiveSpoolError("Telemetry controller lease is no longer active")
        return _open_locked_file(worker_path, timeout_seconds)


def release_run_lease(fd: int) -> None:
    try:
        _unlock_fd(fd)
    finally:
        os.close(fd)


def spool_is_active(spool_dir: str | Path) -> bool:
    """Return true when a controller or worker still owns the spool."""
    directory = Path(spool_dir)
    if not directory.is_dir():
        return False
    # A controller creates its run directory immediately before taking the
    # lease and fsyncing run_start. Never mistake that short window for an
    # orphan that can be collected or removed.
    if not any(directory.glob("*.jsonl")):
        return True
    return _active_lease_held(directory)


@contextlib.contextmanager
def _spool_read_lock(spool_dir: Path) -> Iterator[None]:
    with _named_file_lock(spool_dir / ".collector.lock", 60.0):
        if _active_lease_held(spool_dir):
            raise ActiveSpoolError("Telemetry spool still has an active controller or worker")
        yield


@contextlib.contextmanager
def _single_writer_lock(database_path: Path, timeout_seconds: float = 60.0) -> Iterator[None]:
    with database_writer_lock(database_path, timeout_seconds):
        yield


def _event_severity(event: dict[str, Any]) -> tuple[str, str]:
    event_type = event["event_type"]
    outcome = event.get("outcome")
    if event_type == "test_phase":
        message = "Pytest phase report"
    elif event_type == "collection":
        message = "Pytest collection issue"
    elif event_type in {"run_finish", "worker_finish"}:
        message = "Pytest process finished"
    else:
        message = "Pytest telemetry run started"
    severity = "error" if outcome == "failed" or event.get("run_status") == "failed" else "info"
    if outcome in {"skipped", "xfail", "xpassed"}:
        severity = "warning"
    return severity, message


def _attempt_rows(connection: Any, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT attributes FROM pipeline_events
           WHERE run_id = ? AND event_type = 'pytest_phase'""",
        [run_id],
    ).fetchall()
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for (attributes,) in rows:
        data = json.loads(str(attributes))
        key = (data["test_id"], int(data["attempt"]))
        item = grouped.setdefault(
            key,
            {"phases": {}, "duration_ms": 0.0, "failures": [], "complete": False},
        )
        item["phases"][data["phase"]] = data["outcome"]
        item["duration_ms"] += float(data["duration_ms"])
        item["complete"] = item["complete"] or bool(data["complete"])
        if data.get("failure_fingerprint"):
            item["failures"].append(data["failure_fingerprint"])

    results: list[dict[str, Any]] = []
    for (test_id, attempt), record in sorted(grouped.items()):
        phases = set(record["phases"].values())
        if "failed" in phases:
            outcome = "failed"
        elif "xpassed" in phases:
            outcome = "xpassed"
        elif "xfail" in phases:
            outcome = "xfail"
        elif "skipped" in phases:
            outcome = "skipped"
        elif not record["complete"]:
            outcome = "incomplete"
        elif "passed" in phases:
            outcome = "passed"
        else:
            outcome = "incomplete"
        failures = sorted(set(record["failures"]))
        results.append(
            {
                "test_id": test_id,
                "attempt": attempt,
                "outcome": outcome,
                "duration_ms": record["duration_ms"],
                "failure_type": "pytest_failure" if failures else None,
                "failure_fingerprint": failures[0] if failures else None,
                "metadata": json.dumps(
                    {"phases": record["phases"], "complete": record["complete"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return results


@private_database_operation(1)
def collect_spool_run(spool_dir: str | Path, database_path: str | Path) -> CollectionResult:
    """Transactionally collect one run directory; safe to replay after interruption."""
    spool_dir = Path(spool_dir).resolve()
    try:
        run_id = str(uuid.UUID(spool_dir.name))
    except ValueError as exc:
        raise ValueError("Telemetry spool directory must be named with its run UUID") from exc
    with _spool_read_lock(spool_dir):
        events, malformed = _read_events(spool_dir, run_id)
    starts = [event for event in events if event["event_type"] == "run_start"]
    start_event = min(starts, key=lambda item: item["occurred_at"]) if starts else None
    if start_event is not None:
        start_at = _iso_datetime(start_event["started_at"])
        assert start_at is not None
    else:
        observed_times = [_iso_datetime(event["occurred_at"]) for event in events]
        valid_times = [value for value in observed_times if value is not None]
        if not valid_times:
            raise ValueError("Telemetry spool has no recoverable events; it was retained")
        # This is an observed event date, not an inferred suite start time.
        # Missing run metadata remains NULL and the run is explicitly incomplete.
        start_at = min(valid_times)
        malformed += 1
    finishes = [event for event in events if event["event_type"] == "run_finish"]
    finish_event = max(finishes, key=lambda item: item["occurred_at"]) if finishes else None
    finish_at = _iso_datetime(finish_event["finished_at"]) if finish_event else None
    run_status = (
        finish_event["run_status"]
        if finish_event and start_event is not None
        else "incomplete"
    )
    if finish_event and start_event is not None and finish_event["exit_status"] != 0:
        run_status = "failed"

    database = Path(database_path).expanduser().resolve()
    with _single_writer_lock(database):
        path = initialize_database(database)
        duckdb = _duckdb()
        connection = duckdb.connect(str(path))
        events_inserted = 0
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """INSERT INTO pipeline_runs
                   (run_id, run_date, commit_sha, platform, command, status, source_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (run_id) DO NOTHING""",
                [
                    run_id,
                    start_at.date(),
                    start_event["source_revision"] if start_event else None,
                    start_event["platform"] if start_event else None,
                    "pytest (opt-in telemetry; raw arguments omitted)",
                    run_status,
                    "pytest_jsonl_spool_v1",
                ],
            )
            for event in events:
                severity, message = _event_severity(event)
                event_type = {
                    "test_phase": "pytest_phase",
                    "run_start": "pytest_run_start",
                    "run_finish": "pytest_run_finish",
                    "worker_finish": "pytest_worker_finish",
                    "collection": "pytest_collection",
                }[event["event_type"]]
                attributes = {
                    key: value
                    for key, value in event.items()
                    if key not in _COMMON_FIELDS and key not in {"started_at", "finished_at"}
                }
                result = connection.execute(
                    """INSERT INTO pipeline_events
                       (event_id, run_id, occurred_at, component, stage, severity,
                        event_type, message_redacted, correlation_id, duration_ms, attributes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON)
                       ON CONFLICT (event_id) DO NOTHING
                       RETURNING event_id""",
                    [
                        event["event_id"],
                        run_id,
                        _iso_datetime(event["occurred_at"]),
                        "pytest",
                        "test" if event["event_type"] == "test_phase" else "session",
                        severity,
                        event_type,
                        message,
                        event.get("test_id"),
                        event.get("duration_ms"),
                        json.dumps(attributes, sort_keys=True, separators=(",", ":")),
                    ],
                ).fetchone()
                events_inserted += result is not None

            test_rows = _attempt_rows(connection, run_id)
            for row in test_rows:
                connection.execute(
                    """INSERT INTO test_results
                       (run_id, test_id, attempt, outcome, duration_ms,
                        failure_type, failure_fingerprint, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON)
                       ON CONFLICT (run_id, test_id, attempt) DO UPDATE SET
                         outcome = excluded.outcome,
                         duration_ms = excluded.duration_ms,
                         failure_type = excluded.failure_type,
                         failure_fingerprint = excluded.failure_fingerprint,
                         metadata = excluded.metadata""",
                    [
                        run_id,
                        row["test_id"],
                        row["attempt"],
                        row["outcome"],
                        row["duration_ms"],
                        row["failure_type"],
                        row["failure_fingerprint"],
                        row["metadata"],
                    ],
                )

            latest_attempts: dict[str, dict[str, Any]] = {}
            for row in test_rows:
                current = latest_attempts.get(row["test_id"])
                if current is None or row["attempt"] > current["attempt"]:
                    latest_attempts[row["test_id"]] = row
            counts = {name: 0 for name in ("passed", "failed", "skipped", "xfail", "xpassed", "incomplete")}
            for row in latest_attempts.values():
                counts[row["outcome"]] += 1
            passed = counts["passed"]
            failed = counts["failed"] + counts["xpassed"] + counts["incomplete"]
            skipped = counts["skipped"] + counts["xfail"]
            total = passed + failed + skipped
            connection.execute(
                """INSERT INTO test_summaries (run_id, passed, failed, skipped, total)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (run_id) DO UPDATE SET
                     passed = excluded.passed, failed = excluded.failed,
                     skipped = excluded.skipped, total = excluded.total""",
                [run_id, passed, failed, skipped, total],
            )
            if finish_at is not None and start_event is not None:
                duration_ms = max(0.0, (finish_at - start_at).total_seconds() * 1000.0)
                measurement_id = hashlib.sha256(f"pytest-suite-duration-v1:{run_id}".encode()).hexdigest()
                connection.execute(
                    """INSERT INTO measurements
                       (measurement_id, run_id, stage, metric, value, unit, measured_at, dimensions)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON)
                       ON CONFLICT (measurement_id) DO UPDATE SET
                         value = excluded.value, measured_at = excluded.measured_at,
                         dimensions = excluded.dimensions""",
                    [
                        measurement_id,
                        run_id,
                        "pytest",
                        "suite_duration",
                        duration_ms,
                        "milliseconds",
                        finish_at,
                        json.dumps(
                            {
                                "exit_status": int(finish_event["exit_status"]),
                                "python": start_event["python"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ],
                )
            if malformed:
                malformed_id = hashlib.sha256(
                    f"pytest-malformed-spool-v1:{run_id}:{malformed}".encode()
                ).hexdigest()
                connection.execute(
                    """INSERT INTO pipeline_events
                       (event_id, run_id, occurred_at, component, stage, severity,
                        event_type, message_redacted, attributes)
                       VALUES (?, ?, ?, 'pytest', 'collection', 'warning',
                               'pytest_spool_records_rejected',
                               'Malformed or truncated telemetry records were ignored',
                               ?::JSON)
                       ON CONFLICT (event_id) DO NOTHING""",
                    [
                        malformed_id,
                        run_id,
                        datetime.now(timezone.utc),
                        json.dumps({"rejected_record_count": malformed}, separators=(",", ":")),
                    ],
                )
            # A successfully completed suite cannot conceal an attempt that had
            # no teardown report (for example, a killed xdist worker).
            if counts["incomplete"] and run_status == "passed":
                run_status = "incomplete"
            connection.execute(
                "UPDATE pipeline_runs SET status = ? WHERE run_id = ?",
                [run_status, run_id],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return CollectionResult(
        run_id=run_id,
        events_seen=len(events),
        events_inserted=int(events_inserted),
        tests_written=len(test_rows),
        malformed_lines=malformed,
        run_status=run_status,
    )


def collect_pending_spools(database_path: str | Path) -> list[CollectionResult]:
    """Collect every recoverable orphan/current run spool beneath this database."""
    root = spool_root_for(database_path)
    if not root.is_dir():
        return []
    results: list[CollectionResult] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if spool_is_active(candidate):
            continue
        try:
            uuid.UUID(candidate.name)
        except ValueError:
            continue
        try:
            results.append(collect_spool_run(candidate, database_path))
        except ActiveSpoolError:
            continue
        except FileNotFoundError:
            # Another collector may have completed and cleaned it after the
            # directory scan but before this process acquired its read lease.
            continue
        except ValueError:
            # Keep an unrecoverable spool for manual inspection, but do not let
            # it prevent ingestion of other runs or the next pytest session.
            _LOGGER.warning(
                "Skipped an unrecoverable telemetry spool; its contents were retained locally."
            )
    return results


def cleanup_spool_run(spool_dir: str | Path, database_path: str | Path) -> bool:
    """Remove a successfully collected spool without racing another reader."""
    spool_dir = Path(spool_dir)
    database = Path(database_path).expanduser().resolve()
    if not spool_dir.is_dir() or spool_dir.is_symlink():
        return False
    try:
        with _single_writer_lock(database), _spool_read_lock(spool_dir):
            shutil.rmtree(spool_dir)
    except (ActiveSpoolError, FileNotFoundError):
        return False
    except OSError:
        # On platforms that disallow unlinking a locked file, leave the
        # idempotent spool for the next recovery pass rather than risking loss.
        return False
    return True
