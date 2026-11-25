from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")

from autobliteratus.observability.telemetry_spool import (
    append_spool_event,
    collect_spool_run,
    make_spool_event,
    utc_now,
)


def _run_spool(directory, run_id):
    append_spool_event(
        directory,
        make_spool_event(
            run_id,
            "controller",
            "run_start",
            started_at=utc_now(),
            source_revision="abc1234",
            platform="test-platform",
            python="3.12.0",
        ),
    )


def _phase(directory, run_id, phase, outcome="passed", *, complete=False, duration_ms=1.0):
    append_spool_event(
        directory,
        make_spool_event(
            run_id,
            "gw0",
            "test_phase",
            test_id=hashlib.sha256(b"opaque test id").hexdigest(),
            attempt=1,
            phase=phase,
            outcome=outcome,
            duration_ms=duration_ms,
            failure_fingerprint=None,
            complete=complete,
        ),
    )


def test_collector_is_idempotent_and_deduplicates_replayed_phase_events(tmp_path):
    run_id = "b0c37352-7194-40f0-9768-2c420a86c64d"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    _phase(spool, run_id, "setup", duration_ms=1.5)
    _phase(spool, run_id, "call", duration_ms=2.5)
    # Simulate a repeated delivery of the same report. Its event ID is stable
    # even though the append timestamp differs.
    _phase(spool, run_id, "call", duration_ms=2.5)
    _phase(spool, run_id, "teardown", complete=True, duration_ms=0.5)
    append_spool_event(
        spool,
        make_spool_event(
            run_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=0,
            run_status="passed",
        ),
    )

    first = collect_spool_run(spool, database)
    second = collect_spool_run(spool, database)
    assert first.events_seen == 5
    assert first.events_inserted == 5
    assert first.tests_written == 1
    assert second.events_inserted == 0
    assert second.run_status == "passed"

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        runs = connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        phase_events = connection.execute(
            "SELECT COUNT(*) FROM pipeline_events WHERE event_type = 'pytest_phase'"
        ).fetchone()[0]
        result = connection.execute(
            "SELECT outcome, attempt, duration_ms FROM test_results"
        ).fetchone()
        summary = connection.execute(
            "SELECT passed, failed, skipped, total FROM test_summaries"
        ).fetchone()
    finally:
        connection.close()
    assert runs == 1
    assert phase_events == 3
    assert result[:2] == ("passed", 1)
    assert result[2] == pytest.approx(4.5)
    assert summary == (1, 0, 0, 1)


def test_collector_marks_orphaned_attempt_incomplete_and_ignores_partial_tail(tmp_path):
    run_id = "8a17436a-f63b-4e93-98ca-6451ea74a14b"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    _phase(spool, run_id, "setup")
    _phase(spool, run_id, "call")
    with (spool / "gw0.jsonl").open("ab") as handle:
        handle.write(b'{"partial":')

    result = collect_spool_run(spool, database)
    assert result.run_status == "incomplete"
    assert result.malformed_lines == 1

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        outcome, = connection.execute("SELECT outcome FROM test_results").fetchone()
        status, = connection.execute("SELECT status FROM pipeline_runs").fetchone()
        rejected, = connection.execute(
            "SELECT COUNT(*) FROM pipeline_events WHERE event_type = 'pytest_spool_records_rejected'"
        ).fetchone()
        summary = connection.execute(
            "SELECT passed, failed, skipped, total FROM test_summaries"
        ).fetchone()
    finally:
        connection.close()
    assert outcome == "incomplete"
    assert status == "incomplete"
    assert rejected == 1
    assert summary == (0, 1, 0, 1)


def test_concurrent_collectors_serialize_on_database_lock(tmp_path):
    run_id = "c2b81991-f04e-46b2-9ce0-f41bb1764bc7"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    _phase(spool, run_id, "setup")
    _phase(spool, run_id, "call")
    _phase(spool, run_id, "teardown", complete=True)
    append_spool_event(
        spool,
        make_spool_event(
            run_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=0,
            run_status="passed",
        ),
    )
    project_root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "src")
    script = (
        "import sys; from autobliteratus.observability.telemetry_spool import "
        "collect_spool_run; collect_spool_run(sys.argv[1], sys.argv[2])"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(spool), str(database)],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [process.communicate(timeout=30) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM test_results").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM pipeline_events").fetchone()[0] == 5
    finally:
        connection.close()



def test_pytest_hook_tracks_retry_attempts_without_persisting_raw_text(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autobliteratus.observability import pytest_plugin

    run_id = "f31b03b1-2112-4e5c-a6c3-c9674f44f181"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    config = SimpleNamespace(
        _autobliteratus_telemetry={
            "run_id": run_id,
            "spool_dir": str(spool),
            "worker_id": "gw0",
            "role": "worker",
            "distributed": True,
        }
    )
    monkeypatch.setattr(pytest_plugin, "_ACTIVE_CONFIG", config)

    def report(when, outcome, rerun):
        return SimpleNamespace(
            nodeid="test_private[api_key=do-not-write]",
            when=when,
            outcome=outcome,
            duration=0.001,
            failed=outcome == "failed",
            longrepr="token=do-not-write" if outcome == "failed" else "",
            wasxfail=None,
            rerun=rerun,
        )

    for phase, outcome in (("setup", "passed"), ("call", "failed"), ("teardown", "passed")):
        pytest_plugin.pytest_runtest_logreport(report(phase, outcome, 0))
    for phase, outcome in (("setup", "passed"), ("call", "passed"), ("teardown", "passed")):
        pytest_plugin.pytest_runtest_logreport(report(phase, outcome, 1))
    append_spool_event(
        spool,
        make_spool_event(
            run_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=0,
            run_status="passed",
        ),
    )

    spool_text = "".join(path.read_text(encoding="utf-8") for path in spool.glob("*.jsonl"))
    assert "do-not-write" not in spool_text
    result = collect_spool_run(spool, database)
    assert result.tests_written == 2

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            "SELECT attempt, outcome FROM test_results ORDER BY attempt"
        ).fetchall()
        summary = connection.execute(
            "SELECT passed, failed, skipped, total FROM test_summaries"
        ).fetchone()
        persisted = str(connection.execute("SELECT * FROM pipeline_events").fetchall())
    finally:
        connection.close()
    assert rows == [(1, "failed"), (2, "passed")]
    assert summary == (1, 0, 0, 1)
    assert "do-not-write" not in persisted


def test_collection_failures_are_persisted_only_as_safe_events(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autobliteratus.observability import pytest_plugin

    run_id = "0c472833-470b-4533-a2f6-9b7469bfc7b0"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    config = SimpleNamespace(
        _autobliteratus_telemetry={
            "run_id": run_id,
            "spool_dir": str(spool),
            "worker_id": "gw0",
            "role": "worker",
            "distributed": True,
        }
    )
    monkeypatch.setattr(pytest_plugin, "_ACTIVE_CONFIG", config)
    report = SimpleNamespace(
        outcome="failed",
        nodeid="test_private_collection::test_secret_parameter",
        failed=True,
        longrepr="OPENAI_API_KEY=collection-canary",
    )
    pytest_plugin.pytest_collectreport(report)
    append_spool_event(
        spool,
        make_spool_event(
            run_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=2,
            run_status="failed",
        ),
    )

    spool_text = "".join(path.read_text(encoding="utf-8") for path in spool.glob("*.jsonl"))
    assert "collection-canary" not in spool_text
    assert "test_secret_parameter" not in spool_text
    collect_spool_run(spool, database)

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        event = connection.execute(
            "SELECT event_type, severity, message_redacted, attributes FROM pipeline_events "
            "WHERE event_type = 'pytest_collection'"
        ).fetchone()
        results = connection.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]
    finally:
        connection.close()
    assert event[:3] == ("pytest_collection", "error", "Pytest collection issue")
    assert "collection-canary" not in str(event)
    assert "test_secret_parameter" not in str(event)
    assert results == 0


def test_pending_recovery_skips_a_live_worker_lease(tmp_path):
    from autobliteratus.observability.telemetry_spool import (
        acquire_run_lease,
        collect_pending_spools,
        release_run_lease,
        spool_is_active,
    )

    run_id = "a9f4dd33-9912-436d-b44f-9055742a21db"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    append_spool_event(
        spool,
        make_spool_event(
            run_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=0,
            run_status="passed",
        ),
    )
    controller_lease = acquire_run_lease(spool)
    worker_lease = acquire_run_lease(spool, worker_id="gw0")
    release_run_lease(controller_lease)
    try:
        assert spool_is_active(spool)
        assert collect_pending_spools(database) == []
    finally:
        release_run_lease(worker_lease)

    assert not spool_is_active(spool)
    collected = collect_pending_spools(database)
    assert [item.run_id for item in collected] == [run_id]
    assert collected[0].run_status == "passed"


def test_torn_run_start_does_not_block_other_or_new_spools(tmp_path):
    from autobliteratus.observability.telemetry_spool import collect_pending_spools

    database = tmp_path / "telemetry.duckdb"
    root = tmp_path / "telemetry.duckdb.spool"
    damaged_id = "5d749b27-9eb9-4dd1-8d61-401d83bed9c5"
    damaged = root / damaged_id
    damaged.mkdir(parents=True)
    with (damaged / "controller.jsonl").open("wb") as handle:
        handle.write(b'{"schema_version":1,"event_type":"run_start"')
    _phase(damaged, damaged_id, "setup")
    _phase(damaged, damaged_id, "call")
    _phase(damaged, damaged_id, "teardown", complete=True)

    valid_id = "8cefec72-225c-4f50-9f23-ae71b2ec2870"
    valid = root / valid_id
    valid.mkdir(parents=True)
    _run_spool(valid, valid_id)
    append_spool_event(
        valid,
        make_spool_event(
            valid_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=0,
            run_status="passed",
        ),
    )

    results = collect_pending_spools(database)
    assert {item.run_id for item in results} == {damaged_id, valid_id}
    damaged_result = next(item for item in results if item.run_id == damaged_id)
    assert damaged_result.run_status == "incomplete"
    assert damaged_result.tests_written == 1
    assert damaged_result.malformed_lines >= 1

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        damaged_row = connection.execute(
            "SELECT status, commit_sha, platform FROM pipeline_runs WHERE run_id = ?",
            [damaged_id],
        ).fetchone()
        run_count = connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    finally:
        connection.close()
    assert damaged_row == ("incomplete", None, None)
    assert run_count == 2


def test_oversized_spool_line_is_bounded_and_later_events_survive(tmp_path):
    run_id = "e224fc50-9797-41ac-afb4-b1808b101e07"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    with (spool / "gw0.jsonl").open("wb") as handle:
        handle.write(b"X" * (1024 * 1024) + b"\n")
    _phase(spool, run_id, "setup")
    _phase(spool, run_id, "call")
    _phase(spool, run_id, "teardown", complete=True)

    result = collect_spool_run(spool, database)
    assert result.malformed_lines == 1
    assert result.tests_written == 1


def test_collector_wal_permissions_are_private_during_write(tmp_path, monkeypatch):
    import stat
    from pathlib import Path
    from types import SimpleNamespace

    if os.name == "nt":
        pytest.skip("POSIX file mode bits are not portable to Windows")
    import duckdb

    from autobliteratus.observability import telemetry_spool

    run_id = "c2d705a0-4cbb-4e13-9d9f-b51d81247b13"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    database = tmp_path / "telemetry.duckdb"
    _run_spool(spool, run_id)
    _phase(spool, run_id, "setup")
    _phase(spool, run_id, "call")
    _phase(spool, run_id, "teardown", complete=True)
    append_spool_event(
        spool,
        make_spool_event(
            run_id,
            "controller",
            "run_finish",
            finished_at=utc_now(),
            exit_status=0,
            run_status="passed",
        ),
    )

    observed_wal_modes = []

    class MonitoringConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, *args, **kwargs):
            result = self.connection.execute(*args, **kwargs)
            wal_path = Path(f"{database}.wal")
            if wal_path.exists():
                observed_wal_modes.append(stat.S_IMODE(wal_path.stat().st_mode))
            return result

        def close(self):
            self.connection.close()

    def connect(*args, **kwargs):
        return MonitoringConnection(duckdb.connect(*args, **kwargs))

    monkeypatch.setattr(telemetry_spool, "_duckdb", lambda: SimpleNamespace(connect=connect))
    collect_spool_run(spool, database)

    assert observed_wal_modes
    assert set(observed_wal_modes) == {0o600}


def test_unparseable_json_records_are_counted_without_poisoning_recovery(tmp_path):
    import json

    from autobliteratus.observability import telemetry_spool

    run_id = "da1d3985-cc80-4b54-a0fa-2a3f18e93f50"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    _run_spool(spool, run_id)
    _phase(spool, run_id, "call")
    with (spool / "gw1.jsonl").open("wb") as handle:
        handle.write(b'{"value":' + b"9" * 5000 + b"}\n")
        handle.write(b"[" * 1200 + b"0" + b"]" * 1200 + b"\n")
        handle.write(b'{"event_type": []}\n')
        bad_version = telemetry_spool.make_spool_event(
            run_id,
            "gw1",
            "worker_finish",
            finished_at=utc_now(),
            exit_status=0,
        )
        bad_version["schema_version"] = True
        bad_version["event_id"] = telemetry_spool._canonical_event_id(bad_version)
        handle.write((json.dumps(bad_version, separators=(",", ":")) + "\n").encode())

    events, malformed = telemetry_spool._read_events(spool, run_id)

    assert malformed == 4
    assert {event["event_type"] for event in events} == {"run_start", "test_phase"}


def test_aggregate_spool_byte_limit_preserves_other_small_files(tmp_path, monkeypatch):
    from autobliteratus.observability import telemetry_spool

    run_id = "6f3626c2-53a3-4aa4-9bc2-53d52114aa56"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    _run_spool(spool, run_id)
    _phase(spool, run_id, "call")
    (spool / "a-over-limit.jsonl").write_bytes(b"X" * 1025)
    monkeypatch.setattr(telemetry_spool, "_MAX_SPOOL_BYTES", 1024)

    events, malformed = telemetry_spool._read_events(spool, run_id)

    assert malformed == 1
    assert {event["event_type"] for event in events} == {"run_start", "test_phase"}


def test_actual_bytes_remain_capped_if_a_spool_grows_after_stat(tmp_path, monkeypatch):
    from pathlib import Path

    from autobliteratus.observability import telemetry_spool

    run_id = "3ab01077-f4fc-4fb1-ad10-1bfc18854d79"
    spool = tmp_path / "telemetry.duckdb.spool" / run_id
    spool.mkdir(parents=True)
    (spool / "a-growing.jsonl").write_bytes(b"X" * 4096)
    _run_spool(spool, run_id)
    _phase(spool, run_id, "call")
    monkeypatch.setattr(telemetry_spool, "_MAX_SPOOL_BYTES", 512)
    original_stat = Path.stat
    original_open = Path.open
    bytes_consumed = 0

    class CountingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def readline(self, size=-1):
            nonlocal bytes_consumed
            result = self.handle.readline(size)
            bytes_consumed += len(result)
            return result

        def read(self, size=-1):
            nonlocal bytes_consumed
            result = self.handle.read(size)
            bytes_consumed += len(result)
            return result

        def tell(self):
            return self.handle.tell()

    def smaller_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path.name == "a-growing.jsonl":
            from types import SimpleNamespace

            return SimpleNamespace(st_size=1, st_mode=result.st_mode)
        return result

    def counted_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return CountingReader(handle) if path.name == "a-growing.jsonl" else handle

    monkeypatch.setattr(Path, "stat", smaller_stat)
    monkeypatch.setattr(Path, "open", counted_open)
    events, malformed = telemetry_spool._read_events(spool, run_id)

    assert events == []
    assert malformed >= 1
    assert bytes_consumed <= 512


def test_spool_directory_permission_failure_fails_closed(tmp_path, monkeypatch):
    from pathlib import Path

    from autobliteratus.observability import telemetry_spool

    if os.name == "nt":
        pytest.skip("POSIX mode enforcement is not portable to Windows")
    target = tmp_path / "private-spool"
    original_chmod = Path.chmod

    def fail_target_chmod(path, mode):
        if path == target:
            raise PermissionError("simulated chmod failure")
        return original_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", fail_target_chmod)
    with pytest.raises(PermissionError, match="simulated chmod failure"):
        telemetry_spool.ensure_private_spool_directory(target)
