"""Strict v1 JSON importer for asserted external artifact metadata.

Contract: the top-level object has exactly ``schema_version`` (integer 1) and
``artifacts`` (an array of at most 10,000 objects). Each object requires
``run_id``, ``artifact_sha256``, ``artifact_kind``, and ``uri``; the only
optional keys are ``size_bytes``, ``content_type``, ``model_id``, and
``model_revision``. The artifact digest is lowercase hexadecimal SHA-256;
``size_bytes`` is a nonnegative signed 64-bit integer. All text is ASCII and
field lengths/formats are bounded below. Unknown keys, duplicate JSON keys,
non-finite JSON values, invalid UTF-8, and JSON nesting deeper than 8 are
rejected. The complete source is limited to 4 MiB.

This is an importer of *asserted manifest metadata only*. It neither fetches
nor hashes nor verifies external artifact bytes, and the manifest's digest is
not proof that the referenced URI contains those bytes. URIs are untrusted;
only ``file``, ``https``, ``s3``, and ``gs`` forms are accepted, with no
userinfo, query, fragment, control characters, whitespace, or obvious
credential-bearing text. No prompts, commands, logs, or arbitrary source JSON
are stored. Identical artifact metadata is idempotent by artifact digest;
existing rows with conflicting metadata are rejected without overwrite.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from autobliteratus.observability.telemetry_store import (
    _check_json_nesting,
    _duckdb,
    _strict_json_object,
    initialize_database,
    private_database_operation,
)

SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 10_000
MAX_URI_BYTES = 2_048
MAX_TEXT_BYTES = 256

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_ARTIFACT_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_TYPE_RE = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]{1,63}/[A-Za-z0-9!#$&^_.+-]{1,63}$"
)
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_CREDENTIAL_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"authorization|password|passwd|client[_-]?secret|private[_-]?key|secret)"
    r"\s*[:=]|\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+|"
    r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|"
    r"glpat-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9-]+)"
)
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_REQUIRED_KEYS = {"run_id", "artifact_sha256", "artifact_kind", "uri"}
_OPTIONAL_KEYS = {"size_bytes", "content_type", "model_id", "model_revision"}


@dataclass(frozen=True)
class ArtifactManifestImportResult:
    records_seen: int
    records_inserted: int
    source_sha256: str


def _bounded_ascii(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"Artifact manifest {field} must be a bounded non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Artifact manifest {field} must contain ASCII only") from exc
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"Artifact manifest {field} contains a control character")
    return value


def _safe_uri(value: Any) -> str:
    uri = _bounded_ascii(value, "uri", MAX_URI_BYTES)
    if any(char.isspace() for char in uri):
        raise ValueError("Artifact manifest uri must not contain whitespace")
    if "?" in uri or "#" in uri:
        raise ValueError("Artifact manifest uri must not contain a query or fragment")
    decoded_uri = unquote(uri)
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in decoded_uri):
        raise ValueError("Artifact manifest uri must not contain encoded whitespace or controls")
    if _CREDENTIAL_RE.search(decoded_uri):
        raise ValueError("Artifact manifest uri appears to contain credentials")
    try:
        parsed = urlsplit(uri)
        # Accessing .port validates malformed/out-of-range port syntax.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Artifact manifest uri is malformed") from exc
    if parsed.scheme not in {"file", "https", "s3", "gs"}:
        raise ValueError("Artifact manifest uri uses an unsupported scheme")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("Artifact manifest uri must not contain userinfo")
    if parsed.scheme == "https":
        if not parsed.hostname or "%" in parsed.netloc:
            raise ValueError("Artifact manifest https uri must have a valid host")
    elif parsed.scheme == "file":
        if parsed.netloc or not parsed.path.startswith("/"):
            raise ValueError("Artifact manifest file uri must use an absolute local path")
    else:
        if not _BUCKET_RE.fullmatch(parsed.netloc) or not parsed.path.startswith("/"):
            raise ValueError("Artifact manifest object uri must name a safe bucket and key")
        if len(parsed.path) == 1:
            raise ValueError("Artifact manifest object uri must include an object key")
    return uri


def _optional_text(
    value: Any, field: str, pattern: re.Pattern[str], maximum: int
) -> str | None:
    if value is None:
        return None
    text = _bounded_ascii(value, field, maximum)
    if not pattern.fullmatch(text) or _CREDENTIAL_RE.search(text):
        raise ValueError(f"Artifact manifest {field} has an unsupported format")
    return text


def _validate_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Each artifact manifest record must be an object")
    keys = set(value)
    if not _REQUIRED_KEYS.issubset(keys) or keys - _REQUIRED_KEYS - _OPTIONAL_KEYS:
        raise ValueError("Artifact manifest record keys do not match the v1 contract")

    run_id = _bounded_ascii(value["run_id"], "run_id", 128)
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Artifact manifest run_id has an unsupported format")
    digest = value["artifact_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("Artifact manifest artifact_sha256 must be lowercase SHA-256")
    kind = _bounded_ascii(value["artifact_kind"], "artifact_kind", 64)
    if not _ARTIFACT_KIND_RE.fullmatch(kind):
        raise ValueError("Artifact manifest artifact_kind has an unsupported format")
    uri = _safe_uri(value["uri"])

    size = value.get("size_bytes")
    if size is not None and (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 <= size <= 2**63 - 1
    ):
        raise ValueError("Artifact manifest size_bytes must be a nonnegative signed 64-bit integer")
    content_type = _optional_text(
        value.get("content_type"), "content_type", _CONTENT_TYPE_RE, 127
    )
    model_id = _optional_text(value.get("model_id"), "model_id", _MODEL_ID_RE, 256)
    model_revision = _optional_text(
        value.get("model_revision"), "model_revision", _MODEL_ID_RE, 256
    )
    return {
        "run_id": run_id,
        "artifact_sha256": digest,
        "artifact_kind": kind,
        "uri": uri,
        "size_bytes": size,
        "content_type": content_type,
        "model_id": model_id,
        "model_revision": model_revision,
    }


def _read_manifest(source_path: Path) -> tuple[bytes, list[dict[str, Any]], int]:
    with source_path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Artifact manifest exceeds the 4 MiB input limit")
    try:
        _check_json_nesting(raw, maximum_depth=8)
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("Artifact manifest is not valid strict UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "artifacts"}
        or isinstance(payload.get("schema_version"), bool)
        or not isinstance(payload.get("schema_version"), int)
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Artifact manifest must match the v1 top-level contract")
    raw_records = payload["artifacts"]
    if not isinstance(raw_records, list):
        raise TypeError("Artifact manifest artifacts must be an array")
    if len(raw_records) > MAX_RECORDS:
        raise ValueError("Artifact manifest exceeds the 10000-record limit")
    records = [_validate_record(record) for record in raw_records]

    # One digest names one artifact. Identical repeats in one manifest collapse;
    # contradictory assertions are rejected before any database operation.
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        prior = unique.get(record["artifact_sha256"])
        if prior is not None and prior != record:
            raise ValueError("Artifact manifest has conflicting metadata for one digest")
        unique[record["artifact_sha256"]] = record
    return raw, list(unique.values()), len(records)


@private_database_operation(1)
def import_artifact_manifest(
    manifest_path: str | Path, database_path: str | Path
) -> ArtifactManifestImportResult:
    """Import strict artifact-manifest v1 metadata transactionally and idempotently.

    The source is fully decoded and validated before initialization or writes.
    Every referenced run must exist. Identical digest/metadata replays are
    no-ops; changing metadata for an already-known digest is an explicit error.
    """
    raw, records, records_seen = _read_manifest(Path(manifest_path).expanduser())
    source_sha256 = hashlib.sha256(raw).hexdigest()
    database = initialize_database(database_path)
    connection = _duckdb().connect(str(database))
    inserted = 0
    try:
        connection.execute("BEGIN TRANSACTION")
        run_ids = sorted({record["run_id"] for record in records})
        if run_ids:
            existing_runs = {
                row[0]
                for row in connection.execute(
                    "SELECT run_id FROM pipeline_runs WHERE run_id IN (SELECT UNNEST(?))",
                    [run_ids],
                ).fetchall()
            }
        else:
            existing_runs = set()
        unknown_runs = set(run_ids) - existing_runs
        if unknown_runs:
            raise ValueError(
                "Artifact manifest references unknown run_id: "
                + ", ".join(sorted(unknown_runs))
            )

        metadata_columns = (
            "run_id", "artifact_kind", "uri", "size_bytes", "content_type",
            "model_id", "model_revision",
        )
        to_insert: list[dict[str, Any]] = []
        for record in records:
            existing = connection.execute(
                "SELECT run_id, artifact_kind, uri, size_bytes, content_type, "
                "model_id, model_revision FROM artifact_refs WHERE artifact_sha256 = ?",
                [record["artifact_sha256"]],
            ).fetchone()
            expected = tuple(record[column] for column in metadata_columns)
            if existing is not None:
                if tuple(existing) != expected:
                    raise ValueError(
                        "Artifact digest already exists with conflicting metadata: "
                        + record["artifact_sha256"]
                    )
            else:
                to_insert.append(record)

        for record in to_insert:
            connection.execute(
                """INSERT INTO artifact_refs
                   (artifact_sha256, run_id, artifact_kind, uri, size_bytes,
                    content_type, model_id, model_revision, manifest_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    record["artifact_sha256"], record["run_id"],
                    record["artifact_kind"], record["uri"], record["size_bytes"],
                    record["content_type"], record["model_id"],
                    record["model_revision"], source_sha256,
                ],
            )
            inserted += 1
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return ArtifactManifestImportResult(
        records_seen=records_seen, records_inserted=inserted,
        source_sha256=source_sha256,
    )
