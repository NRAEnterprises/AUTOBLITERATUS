"""Build and verify weight-delta artifacts without granting mutation authority.

This module deliberately has no module-level ``run(adapter, context)`` entrypoint,
so the prototype is not discoverable as an active pipeline stage. Callers must
supply both baseline and transformed model/state objects explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA_VERSION = 1
_DELTA_FILENAME = "delta.safetensors"
_MANIFEST_FILENAME = "autobliteratus_delta.json"


def _state_dict(value: Any) -> Mapping[str, Any]:
    state = value.state_dict() if callable(getattr(value, "state_dict", None)) else value
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Model state must be a non-empty mapping.")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError("Model state keys must be non-empty strings.")
    return state


def _contiguous_preserving_shape(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array).reshape(array.shape)


def _array_and_dtype(value: Any) -> tuple[np.ndarray, str]:
    """Return a CPU NumPy view and the source dtype label for a tensor-like value."""
    if isinstance(value, np.ndarray):
        array = value
        source_dtype = array.dtype.name
    elif callable(getattr(value, "detach", None)):
        tensor = value.detach().to(device="cpu").contiguous()
        source_dtype = str(tensor.dtype).removeprefix("torch.")
        try:
            array = tensor.numpy()
        except (TypeError, RuntimeError):
            if source_dtype != "bfloat16":
                raise
            # NumPy has no native bfloat16 representation. Every bfloat16 value
            # is exactly representable in float32, which is used only as the
            # arithmetic/serialization representation for the delta.
            import torch

            array = tensor.to(dtype=torch.float32).numpy()
    else:
        array = np.asarray(value)
        source_dtype = array.dtype.name

    array = _contiguous_preserving_shape(array)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported tensor dtype {source_dtype!r}; expected a numeric tensor.")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"Non-finite values are not allowed in tensor {source_dtype!r}.")
    return array, source_dtype


def _normalize_state(value: Any) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    state = _state_dict(value)
    arrays: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    for name in sorted(state):
        arrays[name], dtypes[name] = _array_and_dtype(state[name])
    return arrays, dtypes


def _fingerprint(arrays: Mapping[str, np.ndarray], dtypes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = _contiguous_preserving_shape(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(dtypes[name].encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_values(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def _make_delta(base: np.ndarray, updated: np.ndarray, name: str) -> np.ndarray:
    if base.dtype.kind != "f" or updated.dtype.kind != "f":
        raise ValueError(f"Changed non-floating tensor {name!r} cannot be represented as an additive delta.")

    # First use the smallest practical representation and verify that adding
    # it back reproduces the exact stored model values. Fall back to float64
    # arithmetic for cases where float32 subtraction loses round-trip fidelity.
    work_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
    for work_dtype in work_dtypes:
        delta = updated.astype(work_dtype) - base.astype(work_dtype)
        restored = (base.astype(work_dtype) + delta).astype(base.dtype)
        if np.array_equal(restored, updated):
            return _contiguous_preserving_shape(delta)

    raise ValueError(
        f"Changed tensor {name!r} cannot be reconstructed exactly from an additive delta."
    )


def _validate_pair(
    base_arrays: Mapping[str, np.ndarray],
    base_dtypes: Mapping[str, str],
    updated_arrays: Mapping[str, np.ndarray],
    updated_dtypes: Mapping[str, str],
) -> None:
    if set(base_arrays) != set(updated_arrays):
        missing = sorted(set(base_arrays) - set(updated_arrays))
        extra = sorted(set(updated_arrays) - set(base_arrays))
        raise ValueError(f"Model state keys differ (missing={missing[:5]}, extra={extra[:5]}).")
    for name in base_arrays:
        if base_arrays[name].shape != updated_arrays[name].shape:
            raise ValueError(f"Tensor shape changed for {name!r}.")
        if base_dtypes[name] != updated_dtypes[name]:
            raise ValueError(f"Tensor dtype changed for {name!r}.")


def reconstruct_weight_delta(base_state: Any, artifact_dir: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Reload a delta artifact, verify its integrity and base binding, and reconstruct state."""
    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / _MANIFEST_FILENAME
    weights_path = root / _DELTA_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Delta manifest is missing or unreadable.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed delta manifest schema.")
    if manifest.get("artifact_kind") != "autobliteratus_weight_delta":
        raise ValueError("Manifest artifact kind is invalid.")
    if not weights_path.is_file() or _sha256_file(weights_path) != manifest.get("delta_sha256"):
        raise ValueError("Delta artifact is missing or its checksum does not match the manifest.")

    base_arrays, base_dtypes = _normalize_state(base_state)
    if _fingerprint(base_arrays, base_dtypes) != manifest.get("base_fingerprint"):
        raise ValueError("Delta artifact is bound to a different baseline model state.")

    entries = manifest.get("tensors")
    if not isinstance(entries, dict) or set(entries) != set(base_arrays):
        raise ValueError("Delta manifest tensor inventory does not match the baseline state.")
    try:
        from safetensors.numpy import load_file

        deltas = load_file(str(weights_path))
    except Exception as exc:
        raise ValueError("Delta safetensors file is unreadable.") from exc

    changed_names = {name for name, item in entries.items() if isinstance(item, dict) and item.get("changed") is True}
    if set(deltas) != changed_names:
        raise ValueError("Delta tensor keys do not match the manifest's changed-tensor inventory.")

    reconstructed = {name: array.copy() for name, array in base_arrays.items()}
    for name in changed_names:
        item = entries[name]
        base = base_arrays[name]
        delta = np.asarray(deltas[name])
        if list(base.shape) != item.get("shape") or base_dtypes[name] != item.get("dtype"):
            raise ValueError(f"Delta manifest metadata differs for tensor {name!r}.")
        if list(delta.shape) != item.get("shape") or delta.dtype.kind != "f":
            raise ValueError(f"Delta tensor shape or dtype is invalid for {name!r}.")
        restored = (base.astype(delta.dtype) + delta).astype(base.dtype)
        if not np.isfinite(restored).all():
            raise ValueError(f"Reconstructed tensor {name!r} contains non-finite values.")
        reconstructed[name] = _contiguous_preserving_shape(restored)

    for name, item in entries.items():
        if not isinstance(item, dict):
            raise TypeError(f"Delta manifest entry is invalid for {name!r}.")
        if list(base_arrays[name].shape) != item.get("shape") or base_dtypes[name] != item.get("dtype"):
            raise ValueError(f"Delta manifest metadata differs for tensor {name!r}.")
    if _fingerprint(reconstructed, base_dtypes) != manifest.get("result_fingerprint"):
        raise ValueError("Reloaded delta does not reproduce the recorded transformed state.")
    return reconstructed


def export_weight_delta(
    base_state: Any,
    updated_state: Any,
    output_dir: str | os.PathLike[str],
    *,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write a safetensors delta and verify exact reconstruction before publish.

    The result is an integrity-checked artifact, not a signed manifest and not
    permission to mutate or deploy a model. Existing output paths are never replaced.
    """
    base_arrays, base_dtypes = _normalize_state(base_state)
    updated_arrays, updated_dtypes = _normalize_state(updated_state)
    _validate_pair(base_arrays, base_dtypes, updated_arrays, updated_dtypes)

    delta_tensors: dict[str, np.ndarray] = {}
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(base_arrays):
        base = base_arrays[name]
        updated = updated_arrays[name]
        changed = not _same_values(base, updated)
        entries[name] = {
            "shape": list(base.shape),
            "dtype": base_dtypes[name],
            "changed": changed,
        }
        if not changed:
            continue
        delta_tensors[name] = _make_delta(base, updated, name)

    if not delta_tensors:
        raise ValueError("No changed tensors were found; refusing to publish an empty delta.")
    try:
        identity = json.loads(json.dumps(model_identity, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("model_identity must be finite JSON-compatible data.") from exc
    if not isinstance(identity, dict):
        raise TypeError("model_identity must be a JSON object.")
    identity_digest = identity.get("identity_sha256")
    if not isinstance(identity_digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", identity_digest) is None:
        raise ValueError("model_identity must contain a 64-character identity_sha256.")

    requested_target = Path(output_dir).expanduser()
    if requested_target.exists() or requested_target.is_symlink():
        raise FileExistsError(f"Delta output already exists: {requested_target}")
    target = requested_target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(target.parent, os.W_OK | os.X_OK):
        raise PermissionError(f"Delta output parent is not writable: {target.parent}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent)))
    published = False
    try:
        from safetensors.numpy import save_file

        delta_path = staging / _DELTA_FILENAME
        save_file(delta_tensors, str(delta_path), metadata={"format": "autobliteratus-delta-v1"})
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "artifact_kind": "autobliteratus_weight_delta",
            "model_identity": identity,
            "base_fingerprint": _fingerprint(base_arrays, base_dtypes),
            "result_fingerprint": _fingerprint(updated_arrays, updated_dtypes),
            "changed_tensor_count": len(delta_tensors),
            "tensor_count": len(entries),
            "tensors": entries,
            "delta_file": _DELTA_FILENAME,
            "delta_sha256": _sha256_file(delta_path),
            "manifest_signed": False,
            "mutation_authorized": False,
        }
        (staging / _MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        reconstructed = reconstruct_weight_delta(base_state, staging)
        if _fingerprint(reconstructed, base_dtypes) != manifest["result_fingerprint"]:
            raise RuntimeError("Pre-publication delta reconstruction verification failed.")

        from autobliteratus.tuning._96_model_export import _publish_without_replacing

        _publish_without_replacing(staging, target)
        published = True
        published_state = reconstruct_weight_delta(base_state, target)
        if _fingerprint(published_state, base_dtypes) != manifest["result_fingerprint"]:
            raise RuntimeError("Post-publication delta reconstruction verification failed.")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(target, ignore_errors=True)
        raise

    return {
        "status": "validated",
        "path": str(target),
        "delta_file": str(target / _DELTA_FILENAME),
        "manifest_file": str(target / _MANIFEST_FILENAME),
        "changed_tensor_count": len(delta_tensors),
        "tensor_count": len(entries),
        "base_fingerprint": manifest["base_fingerprint"],
        "result_fingerprint": manifest["result_fingerprint"],
        "delta_sha256": manifest["delta_sha256"],
        "reconstruction_verified": True,
        "manifest_signed": False,
        "mutation_authorized": False,
    }


class DeltaWeightExportAutoTuning:
    """Legacy direct-call wrapper; remains disabled and not a discovered pipeline stage."""

    def __init__(self, work_dir: str = "./ablated_model/delta", enabled: bool = False):
        self.work_dir = Path(work_dir)
        self.enabled = enabled

    def run(
        self,
        base_model: Any,
        ablated_model: Any,
        *,
        model_identity: Mapping[str, Any],
    ) -> Path:
        if not self.enabled:
            return Path()
        report = export_weight_delta(
            base_model,
            ablated_model,
            self.work_dir,
            model_identity=model_identity,
        )
        return Path(report["delta_file"])
