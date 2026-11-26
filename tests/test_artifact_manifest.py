from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import duckdb
import pytest

from autobliteratus.observability import artifact_manifest
from autobliteratus.observability.artifact_manifest import import_artifact_manifest
from autobliteratus.observability.telemetry_store import initialize_database

RUN_ID = "run-artifact-manifest-001"


def _record(**overrides):
    record = {
        "run_id": RUN_ID,
        "artifact_sha256": "a" * 64,
        "artifact_kind": "model_weights",
        "uri": "s3://model-bucket/releases/model.bin",
        "size_bytes": 1234,
        "content_type": "application/octet-stream",
        "model_id": "example/model",
        "model_revision": "revision-001",
    }
    record.update(overrides)
    return record


def _write_manifest(path, *records):
    payload = {"schema_version": 1, "artifacts": list(records)}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _database(tmp_path):
    path = tmp_path / "artifacts.duckdb"
    initialize_database(path)
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "INSERT INTO pipeline_runs (run_id, status) VALUES (?, ?)",
            [RUN_ID, "completed"],
        )
    finally:
        connection.close()
    return path


def _artifact_rows(database):
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(
            "SELECT artifact_sha256, run_id, artifact_kind, uri, size_bytes, "
            "content_type, model_id, model_revision, manifest_sha256 "
            "FROM artifact_refs ORDER BY artifact_sha256"
        ).fetchall()
    finally:
        connection.close()


def test_valid_v1_import_stores_metadata_and_manifest_provenance(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "artifacts.json"
    _write_manifest(source, _record())

    result = import_artifact_manifest(source, database)

    row = _artifact_rows(database)[0]
    assert result.records_seen == 1
    assert result.records_inserted == 1
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row == (
        "a" * 64, RUN_ID, "model_weights", "s3://model-bucket/releases/model.bin",
        1234, "application/octet-stream", "example/model", "revision-001",
        result.source_sha256,
    )


@pytest.mark.parametrize("version", [True, 1.0])
def test_schema_version_must_be_an_integer_not_a_coercible_value(tmp_path, version):
    source = tmp_path / "bad-version.json"
    source.write_text(
        json.dumps({"schema_version": version, "artifacts": []}), encoding="utf-8"
    )
    database = tmp_path / "not-created.duckdb"

    with pytest.raises(ValueError, match="v1 top-level contract"):
        import_artifact_manifest(source, database)

    assert not database.exists()


def test_identical_replay_is_idempotent_and_preserves_original_provenance(tmp_path):
    database = _database(tmp_path)
    first_source = tmp_path / "first.json"
    replay_source = tmp_path / "replay.json"
    _write_manifest(first_source, _record())
    _write_manifest(replay_source, _record())

    first = import_artifact_manifest(first_source, database)
    replay = import_artifact_manifest(replay_source, database)

    assert first.records_inserted == 1
    assert replay.records_seen == 1
    assert replay.records_inserted == 0
    assert _artifact_rows(database)[0][-1] == first.source_sha256
    assert len(_artifact_rows(database)) == 1


def test_conflicting_metadata_for_existing_digest_is_rejected(tmp_path):
    database = _database(tmp_path)
    initial = tmp_path / "initial.json"
    conflict = tmp_path / "conflict.json"
    _write_manifest(initial, _record())
    _write_manifest(conflict, _record(uri="https://cdn.example.net/different.bin"))
    import_artifact_manifest(initial, database)
    before = _artifact_rows(database)

    with pytest.raises(ValueError, match="conflicting metadata"):
        import_artifact_manifest(conflict, database)

    assert _artifact_rows(database) == before


def test_conflicting_duplicate_digests_inside_manifest_are_rejected(tmp_path):
    source = tmp_path / "duplicate-digest.json"
    _write_manifest(source, _record(), _record(uri="https://cdn.example.net/other.bin"))

    with pytest.raises(ValueError, match="conflicting metadata"):
        import_artifact_manifest(source, tmp_path / "not-created.duckdb")
    assert not (tmp_path / "not-created.duckdb").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_sha256", "A" * 64, "lowercase SHA-256"),
        ("artifact_sha256", "not-a-digest", "lowercase SHA-256"),
        ("uri", "http://example.com/model.bin", "unsupported scheme"),
        ("uri", "https://user:password@example.com/file", "userinfo"),
        ("uri", "https://example.com/file?token=secret", "query or fragment"),
        ("uri", "https://example.com/file#section", "query or fragment"),
        ("uri", "https://example.com/path/api_key=secret", "credentials"),
        ("uri", "https://example.com/path/%0Asecret", "encoded whitespace or controls"),
        ("uri", "file:relative/path", "absolute local path"),
        ("uri", "s3://bucket/", "object key"),
        ("uri", "https://example.com/a\nb", "control character"),
        ("uri", "https://example.com/" + "x" * 2040, "bounded non-empty"),
    ],
)
def test_invalid_uri_or_digest_rejected_before_database_creation(
    tmp_path, field, value, message
):
    source = tmp_path / "invalid.json"
    _write_manifest(source, _record(**{field: value}))
    database = tmp_path / "not-created.duckdb"

    with pytest.raises(ValueError, match=message):
        import_artifact_manifest(source, database)
    assert not database.exists()


def test_unknown_run_rejects_whole_manifest_without_inserting_any_artifact(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "unknown-run.json"
    _write_manifest(
        source,
        _record(),
        _record(
            run_id="run-not-present",
            artifact_sha256="b" * 64,
            uri="https://cdn.example.net/other.bin",
        ),
    )

    with pytest.raises(ValueError, match="unknown run_id"):
        import_artifact_manifest(source, database)
    assert _artifact_rows(database) == []


def test_malformed_duplicate_key_invalid_utf8_and_deep_json_rejected(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "malformed.json"
    cases = [
        (b'{"schema_version":1,"schema_version":1,"artifacts":[]}', "strict UTF-8 JSON"),
        (b'{"schema_version":1,"artifacts":[', "strict UTF-8 JSON"),
        (b'{"schema_version":1,"artifacts":[]}\xff', "strict UTF-8 JSON"),
        (
            b'{"schema_version":1,"artifacts":' + b"[" * 10 + b"0" + b"]" * 10 + b"}",
            "strict UTF-8 JSON",
        ),
        (
            b'{"schema_version":1,"artifacts":[{"run_id":"x","run_id":"y"}]}',
            "strict UTF-8 JSON",
        ),
    ]
    for raw, message in cases:
        source.write_bytes(raw)
        with pytest.raises(ValueError, match=message):
            import_artifact_manifest(source, database)
    assert _artifact_rows(database) == []


def test_input_record_and_artifact_size_limits_rejected(tmp_path):
    database = _database(tmp_path)
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        import_artifact_manifest(source, database)

    _write_manifest(source, _record(size_bytes=2**63))
    with pytest.raises(ValueError, match="size_bytes"):
        import_artifact_manifest(source, database)

    source.write_text(
        '{"schema_version":1,"artifacts":['
        + ",".join(json.dumps(_record(artifact_sha256=f"{index:064x}")) for index in range(10_001))
        + "]}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="10000-record"):
        import_artifact_manifest(source, database)
    assert _artifact_rows(database) == []


def test_database_error_rolls_back_earlier_inserts(tmp_path, monkeypatch):
    database = _database(tmp_path)
    source = tmp_path / "two-artifacts.json"
    second = _record(
        artifact_sha256="b" * 64,
        uri="https://cdn.example.net/second.bin",
    )
    _write_manifest(source, _record(), second)
    real_connection = duckdb.connect(str(database))

    class FailingConnection:
        def __init__(self, connection):
            self.connection = connection
            self.artifact_inserts = 0

        def execute(self, query, parameters=None):
            if query.lstrip().startswith("INSERT INTO artifact_refs"):
                self.artifact_inserts += 1
                if self.artifact_inserts == 2:
                    raise RuntimeError("simulated insert failure")
            return self.connection.execute(query, parameters or [])

        def close(self):
            self.connection.close()

    monkeypatch.setattr(
        artifact_manifest,
        "_duckdb",
        lambda: SimpleNamespace(connect=lambda _path: FailingConnection(real_connection)),
    )

    with pytest.raises(RuntimeError, match="simulated insert failure"):
        import_artifact_manifest(source, database)

    assert _artifact_rows(database) == []
