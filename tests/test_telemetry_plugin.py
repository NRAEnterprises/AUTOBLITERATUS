from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")


def test_opt_in_pytest_plugin_records_redacted_per_test_results(tmp_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    test_file = tmp_path / "test_sample_private_parameter.py"
    database = tmp_path / "telemetry.duckdb"
    test_file.write_text(
        """import pytest


def test_passed():
    assert 2 + 2 == 4


@pytest.mark.parametrize('credential', ['api_key=example-secret-value'])
def test_failure_fingerprint_only(credential):
    assert False, credential


def test_skipped():
    pytest.skip('skip details must not be persisted')
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "src")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "autobliteratus.observability.pytest_plugin",
            "--autobliteratus-db",
            str(database),
            str(test_file),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 1, process.stdout + process.stderr

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        outcomes = connection.execute(
            "SELECT outcome, COUNT(*) FROM test_results GROUP BY outcome ORDER BY outcome"
        ).fetchall()
        names = connection.execute("SELECT test_id FROM test_results").fetchall()
        combined = " ".join(
            str(row)
            for table in ("pipeline_runs", "test_results", "measurements")
            for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        )
        summary = connection.execute("SELECT passed, failed, skipped, total FROM test_summaries").fetchone()
    finally:
        connection.close()

    assert dict(outcomes) == {"failed": 1, "passed": 1, "skipped": 1}
    assert summary == (1, 1, 1, 3)
    assert len(names) == 3
    assert all(len(row[0]) == 64 for row in names)
    assert "example-secret-value" not in combined
    assert "test_failure_fingerprint_only" not in combined
    assert "skip details must not be persisted" not in combined


def test_xdist_workers_spool_and_controller_collects_once(tmp_path):
    pytest.importorskip("xdist", reason="pytest-xdist is optional")
    project_root = os.path.dirname(os.path.dirname(__file__))
    test_file = tmp_path / "test_distributed_samples.py"
    database = tmp_path / "distributed.duckdb"
    test_file.write_text(
        """import pytest


@pytest.mark.parametrize('value', range(8))
def test_value(value):
    assert value >= 0
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "src")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "2",
            "-p",
            "autobliteratus.observability.pytest_plugin",
            "--autobliteratus-db",
            str(database),
            str(test_file),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM test_results").fetchone()[0] == 8
        assert connection.execute(
            "SELECT passed, failed, skipped, total FROM test_summaries"
        ).fetchone() == (8, 0, 0, 8)
        worker_finishes = connection.execute(
            "SELECT COUNT(*) FROM pipeline_events WHERE event_type = 'pytest_worker_finish'"
        ).fetchone()[0]
        assert worker_finishes == 2
    finally:
        connection.close()

    spool_root = database.with_name(database.name + ".spool")
    assert not spool_root.exists() or not any(child.is_dir() for child in spool_root.iterdir())


def test_overlapping_pytest_sessions_share_database_safely(tmp_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    database = tmp_path / "shared.duckdb"
    test_files = [tmp_path / f"test_parallel_{index}.py" for index in range(2)]
    for index, test_file in enumerate(test_files):
        test_file.write_text(
            f"""import time


def test_sample_{index}():
    time.sleep(0.2)
    assert True


def test_second_{index}():
    time.sleep(0.2)
    assert True
""",
            encoding="utf-8",
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "src")
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "autobliteratus.observability.pytest_plugin",
                "--autobliteratus-db",
                str(database),
                str(test_file),
            ],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for test_file in test_files
    ]
    outputs = [process.communicate(timeout=60) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM test_results").fetchone()[0] == 4
        assert connection.execute(
            "SELECT COUNT(*) FROM test_summaries WHERE total = 2 AND passed = 2"
        ).fetchone()[0] == 2
    finally:
        connection.close()


def test_unrecoverable_orphan_does_not_block_a_new_session(tmp_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    database = tmp_path / "recovery.duckdb"
    broken_run = tmp_path / "recovery.duckdb.spool" / "ce75fda0-a5aa-4a31-baf4-bd594670ef3e"
    broken_run.mkdir(parents=True)
    (broken_run / "controller.jsonl").write_bytes(b"not-json\n")
    test_file = tmp_path / "test_after_orphan.py"
    test_file.write_text("def test_runs_after_bad_spool():\n    assert True\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "src")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "autobliteratus.observability.pytest_plugin",
            "--autobliteratus-db",
            str(database),
            str(test_file),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "unrecoverable telemetry spool" in process.stderr
    assert broken_run.exists()

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 1
        assert connection.execute("SELECT passed, total FROM test_summaries").fetchone() == (1, 1)
    finally:
        connection.close()
