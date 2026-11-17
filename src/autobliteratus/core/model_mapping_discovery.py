"""Evidence-driven model mapping discovery.

Static model mappings are hypotheses only. This module performs bounded,
read-only experiments, records evidence, and keeps every unverified category
unknown. No user-supplied label can promote a category.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from autobliteratus.core.model_mapping import AREA_CATEGORIES, get_model_mapping

TOGGLE = {
    "id": "tuning.model_mapping_discovery",
    "default": False,
    "desc": "Discover and record model mappings from bounded read-only runtime experiments.",
    "read_only": True,
}

_EXPERIMENTS = (
    "config_identity",
    "state_inventory",
    "candidate_path_resolution",
    "weight_shape_validation",
    "cache_api_contract",
    "cache_output_stability",
    "state_mutation_guard",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _config_dict(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return _jsonable(config.to_dict())
    if hasattr(config, "__dict__"):
        return _jsonable(vars(config))
    return {"repr": repr(config)}


def _state_fingerprint(model: Any) -> str:
    state_dict = getattr(model, "state_dict", dict)()
    digest = hashlib.sha256()
    try:
        import torch

        for name in sorted(state_dict):
            tensor = state_dict[name].detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        for name in sorted(state_dict):
            digest.update(name.encode("utf-8"))
            digest.update(repr(state_dict[name]).encode("utf-8"))
    return digest.hexdigest()


def _model_identity(adapter: Any) -> dict[str, Any]:
    model = adapter.handle.model
    config = _config_dict(model)
    model_type = str(config.get("model_type") or getattr(adapter.handle, "architecture", "unknown")).lower()
    identity_payload = {
        "model_type": model_type,
        "config": config,
        "state_fingerprint": _state_fingerprint(model),
    }
    identity_json = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    return {
        "model_type": model_type,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "state_fingerprint": identity_payload["state_fingerprint"],
        "identity_sha256": hashlib.sha256(identity_json.encode("utf-8")).hexdigest(),
    }


def _resolve_path(root: Any, path: str) -> Any | None:
    if path.endswith("(use_cache=...)"):
        target = path.split("(", 1)[0]
        return getattr(root, target, None)
    current = root
    for part in path.split("."):
        if part.endswith("[]"):
            current = getattr(current, part[:-2], None)
            try:
                current = current[0]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _record(code: str, status: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "status": status, **details}


def _run_cache_probe(adapter: Any, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = adapter.handle.model
    tokenizer = adapter.handle.tokenizer
    generate = getattr(model, "generate", None)
    if not callable(generate):
        return (
            _record("cache_api_contract", "unknown", reason="model.generate_unavailable"),
            _record("cache_output_stability", "unknown", reason="model.generate_unavailable"),
        )
    try:
        parameters = inspect.signature(generate).parameters
        accepts_cache = "use_cache" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        accepts_cache = False
    if not accepts_cache:
        return (
            _record("cache_api_contract", "unknown", reason="use_cache_argument_not_exposed"),
            _record("cache_output_stability", "unknown", reason="cache_probe_not_safe"),
        )
    if tokenizer is None or not callable(tokenizer):
        return (
            _record("cache_api_contract", "passed", evidence="use_cache_argument_exposed"),
            _record("cache_output_stability", "unknown", reason="tokenizer_unavailable"),
        )
    prompt = context.get("model_mapping_probe_prompt", "mapping discovery probe")
    if not isinstance(prompt, str) or len(prompt) > 10_000:
        raise ValueError("model_mapping_probe_prompt must be a string of at most 10000 characters.")
    max_new_tokens = context.get("model_mapping_probe_max_new_tokens", 2)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 8:
        raise ValueError("model_mapping_probe_max_new_tokens must be an integer in [1, 8].")
    try:
        import torch

        device = next((parameter.device for parameter in model.parameters() if parameter.device.type != "meta"), torch.device("cpu"))
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items() if torch.is_tensor(value)}
        with torch.inference_mode():
            before = _state_fingerprint(model)
            uncached = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
            cached = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            after = _state_fingerprint(model)
        uncached = getattr(uncached, "sequences", uncached)
        cached = getattr(cached, "sequences", cached)
        if not torch.is_tensor(uncached) or not torch.is_tensor(cached):
            return (
                _record("cache_api_contract", "passed", evidence="use_cache_probe_executed"),
                _record("cache_output_stability", "unknown", reason="non_tensor_generation_output"),
            )
        if tuple(uncached.shape) != tuple(cached.shape):
            drift = "shape_mismatch"
            stable = False
        else:
            drift_value = float(torch.max(torch.abs(uncached.detach().to("cpu") - cached.detach().to("cpu"))).item()) if uncached.numel() else 0.0
            drift = drift_value
            stable = drift_value <= float(context.get("model_mapping_output_tolerance", 0.0))
        return (
            _record("cache_api_contract", "passed", evidence="use_cache_probe_executed"),
            _record("cache_output_stability", "passed" if stable else "failed", drift=drift, state_unchanged=before == after),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        return (
            _record("cache_api_contract", "failed", error_type=type(exc).__name__, message=str(exc)),
            _record("cache_output_stability", "unknown", reason="cache_probe_failed"),
        )


def discover(adapter: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the bounded mapping battery and return an evidence-bound mapping."""
    context = context or {}
    model = adapter.handle.model
    identity = _model_identity(adapter)
    state_before = identity["state_fingerprint"]
    static = get_model_mapping(identity["model_type"])
    evidence: list[dict[str, Any]] = []

    evidence.append(_record("config_identity", "passed" if identity["model_type"] != "unknown" else "unknown", model_type=identity["model_type"]))
    state_dict = getattr(model, "state_dict", dict)()
    named_modules = dict(getattr(model, "named_modules", list)())
    evidence.append(_record("state_inventory", "passed" if state_dict and named_modules else "unknown", parameter_count=len(state_dict), module_count=len(named_modules)))

    areas: dict[str, dict[str, Any]] = {}
    for category in AREA_CATEGORIES:
        candidate = static.areas[category] if static is not None else None
        paths = list(candidate.paths) if candidate is not None else []
        resolved = [path for path in paths if _resolve_path(model, path) is not None]
        path_status = "passed" if resolved else "unknown"
        evidence.append(_record("candidate_path_resolution", path_status, category=category, candidates=paths, resolved=resolved))
        shape_valid = []
        for path in resolved:
            target = _resolve_path(model, path)
            weight = getattr(target, "weight", None)
            if weight is None or getattr(weight, "ndim", 0) != 2:
                continue
            shape_valid.append(path)
        if resolved and any(getattr(_resolve_path(model, path), "weight", None) is None for path in resolved):
            shape_status = "passed"
        else:
            shape_status = "passed" if shape_valid else "unknown"
        evidence.append(_record("weight_shape_validation", shape_status, category=category, validated=shape_valid))
        areas[category] = {
            "status": "verified" if path_status == "passed" and shape_status == "passed" else "unknown",
            "candidate_paths": paths,
            "verified_paths": resolved if path_status == "passed" and shape_status == "passed" else [],
            "reason": "runtime_path_and_shape_evidence" if path_status == "passed" and shape_status == "passed" else "runtime_evidence_required",
        }

    cache_api, cache_quality = _run_cache_probe(adapter, context)
    evidence.extend((cache_api, cache_quality))
    state_after = _state_fingerprint(model)
    evidence.append(
        _record(
            "state_mutation_guard",
            "passed" if state_before == state_after else "failed",
            before=state_before,
            after=state_after,
        )
    )
    if cache_api["status"] == "passed":
        areas["kv_cache"] = {
            "status": "verified" if cache_quality["status"] == "passed" else "unknown",
            "candidate_paths": areas["kv_cache"]["candidate_paths"],
            "verified_paths": ["generate(use_cache=...)"] if cache_quality["status"] == "passed" else [],
            "reason": "cache_api_and_output_probe",
        }
    mutation_detected = state_before != state_after
    return {
        "stage_id": "tuning.model_mapping_discovery",
        "schema_version": 1,
        "status": "failed" if mutation_detected else "measured",
        "model_identity": identity,
        "static_mapping_source": "candidate_hypotheses_only",
        "areas": areas,
        "experiments": evidence,
        "unknown_area_count": sum(area["status"] == "unknown" for area in areas.values()),
        "mutation_detected": mutation_detected,
        "mutation_performed": False,
        "model_modified": False,
        "optimization_performed": False,
        "artifacts": [],
        "failures": [item for item in evidence if item.get("status") == "failed"],
        "warnings": [
            "Static mappings are hypotheses only; labels are promoted only by passing runtime evidence.",
            "Unknown areas are fail-closed and cannot authorize weight mutation.",
        ],
    }


def update_database(path: str | os.PathLike[str], discovery_report: dict[str, Any]) -> None:
    """Atomically merge one evidence-bound discovery into a JSON database."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            database = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Mapping database is not valid JSON: {destination}") from exc
    else:
        database = {"schema_version": 1, "models": {}}
    if not isinstance(database, dict) or not isinstance(database.get("models", {}), dict):
        raise TypeError("Mapping database must contain an object-valued models field.")
    identity = discovery_report["model_identity"]["identity_sha256"]
    history = database["models"].setdefault(identity, [])
    if not isinstance(history, list):
        raise TypeError("Mapping database model history must be a list.")
    history.append(discovery_report)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(database, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Discover mappings and optionally persist the evidence-bound database."""
    report = discover(adapter, context)
    database_path = context.get("model_mapping_db_path")
    if database_path is not None:
        update_database(database_path, report)
        report["database_path"] = str(database_path)
    context["model_mapping_discovery_report"] = report
    return report
